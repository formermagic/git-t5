import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.serialization import from_bytes, to_bytes
from flax.training.train_state import TrainState
from git_t5.utils import rank_zero_only
from omegaconf import MISSING
from tqdm import tqdm

if TYPE_CHECKING:
    from .data_module import T5DataModule
    from .logger import BaseLogger
    from .model import T5ModelForPreTraining
else:
    T5DataModule = Any
    T5ModelForPreTraining = Any
    BaseLogger = Any


@dataclass
class TrainerConfig:
    pass


@dataclass
class T5TrainerConfig(TrainerConfig):
    output_dir: str = MISSING
    max_epochs: int = 3
    logging_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 2000
    push_to_hub: bool = False
    push_to_hub_model_id: Optional[str] = None
    push_to_hub_organization: Optional[str] = None
    push_to_hub_token: Optional[str] = None
    checkpoint_dir: Optional[str] = None


@dataclass
class T5Trainer:
    config: T5TrainerConfig
    model: T5ModelForPreTraining
    data_module: T5DataModule
    logger: BaseLogger
    current_epoch: int = 0
    global_step: int = 0

    def __post_init__(self) -> None:
        self.model.trainer = self
        self.data_module.trainer = self
        self.logger.trainer = self

    def fit(self) -> None:
        self.model.setup()
        self.data_module.setup()

        num_epochs = self.config.max_epochs
        train_dataloader = self.data_module.train_dataloader()
        train_samples = len(train_dataloader)

        optimizer, scheduler_fn = self.model.configure_optimizers(self)
        state = self.create_state(optimizer, self.config.checkpoint_dir)
        rng = jax.random.PRNGKey(self.model.config.seed)
        dropout_rng = jax.random.split(rng, jax.device_count())

        train_time = 0

        for epoch in range(num_epochs):
            running_loss = jnp.array(0, dtype=jnp.float32)
            current_step = 0
            train_start_time = time.time()
            self.current_epoch = epoch

            for batch in tqdm(
                train_dataloader,
                total=train_samples,
                desc=f"Training epoch={epoch}",
            ):
                state, metrics, dropout_rng = self.model.train_step(
                    state, dropout_rng, batch
                )

                running_loss += jax_utils.unreplicate(metrics["loss"])
                current_step += 1
                self.global_step += 1

                if current_step % self.config.logging_steps == 0:
                    state_step = jax_utils.unreplicate(state.step)
                    train_loss = running_loss / current_step
                    train_lr = scheduler_fn(state_step - 1)
                    train_time += time.time() - train_start_time

                    metrics = {
                        "step": state_step.item(),
                        "train_loss": train_loss.item(),
                        "train_lr": train_lr.item(),  # type: ignore
                        "train_time": train_time,
                    }

                    self.logger.log_metrics(metrics, step=self.global_step)

                if current_step % self.config.eval_steps == 0:
                    metrics = self.validate(state)
                    self.logger.log_metrics(metrics, step=self.global_step)

                if current_step % self.config.save_steps == 0:
                    self.save_checkpoint(state, self.config.output_dir)

    def validate(self, state: TrainState) -> Dict[str, float]:
        valid_dataloader = self.data_module.valid_dataloader()
        valid_samples = len(valid_dataloader)
        running_loss = jnp.array(0, dtype=jnp.float32)
        running_accuracy = jnp.array(0, dtype=jnp.float32)
        current_step = 0

        for batch in tqdm(
            valid_dataloader,
            total=valid_samples,
            desc="Validating...",
        ):
            metrics = self.model.valid_step(state, batch)
            running_loss += jax_utils.unreplicate(metrics["loss"])
            running_accuracy += jax_utils.unreplicate(metrics["accuracy"])
            current_step += 1

        loss = running_loss / current_step
        accuracy = running_accuracy / current_step
        metrics = {
            "valid_loss": loss.item(),
            "valid_accuracy": accuracy.item(),
        }

        return metrics

    @rank_zero_only
    def save_checkpoint(self, state: TrainState, save_dir: str) -> None:
        # save pretrained flax model
        state = jax_utils.unreplicate(state)
        flax_model = self.model.model
        flax_model.save_pretrained(
            save_dir,
            params=state.params,
            push_to_hub=self.config.push_to_hub,
            commit_message=f"Saving weights and logs of step {state.step}",
        )

        # save optimizer weights
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))  # type: ignore

        # save the training state
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            training_state = {
                "step": state.step,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
            }

            json.dump(training_state, f)

    def restore_checkpoint(
        self, state: TrainState, save_dir: str
    ) -> Tuple[Dict[str, Any], optax.OptState, Dict[str, Any]]:
        # restore pretrained flax model weights
        with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
            params = from_bytes(state.params, f.read())
            assert isinstance(params, dict)

        # restore optimizer weights
        with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
            opt_state = from_bytes(state.opt_state, f.read())
            assert isinstance(opt_state, optax.OptState)

        # restore the training state
        with open(os.path.join(save_dir, "training_state.json"), "r") as f:
            training_state = json.load(f)
            assert isinstance(training_state, dict)

        return params, opt_state, training_state

    def create_state(
        self,
        optimizer: optax.GradientTransformation,
        checkpoint_dir: Optional[str] = None,
    ) -> TrainState:
        model = self.model.model
        state = TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=optimizer,
        )

        if checkpoint_dir is not None:
            params, opt_state, training_state = self.restore_checkpoint(
                state, checkpoint_dir
            )

            step = training_state["step"]
            self.current_epoch = training_state["current_epoch"]
            self.global_step = training_state["global_step"]

            state = TrainState(
                step=step,
                apply_fn=model.__call__,
                params=params,
                tx=optimizer,
                opt_state=opt_state,
            )

        state = jax_utils.replicate(state)
        return state
