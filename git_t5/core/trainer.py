import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.serialization import from_bytes, to_bytes
from flax.training.train_state import TrainState
from git_t5.core import AutoOptimizer, AutoScheduler, OptimizerConfig
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
    optimizer_config: OptimizerConfig
    logger: BaseLogger
    current_epoch: int = 0
    global_step: int = 0

    def __post_init__(self) -> None:
        self.model.trainer = self
        self.data_module.trainer = self
        self.logger.trainer = self

    def fit(self) -> None:
        num_epochs = self.config.max_epochs
        train_dataloader = self.data_module.train_dataloader()
        train_samples = len(train_dataloader)

        optimizer, scheduler_fn = self.configure_optimizers()
        state = self.create_state(optimizer, self.config.checkpoint_dir)
        rng = jax.random.PRNGKey(self.model.config.seed)
        dropout_rng = jax.random.split(rng, jax.device_count())

        train_time = 0

        for epoch in range(num_epochs):
            train_metrics = []
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

                # accumulate train metrics
                train_metrics.append(metrics)
                current_step += 1
                self.global_step += 1

                if current_step % self.config.logging_steps == 0:
                    state_step = jax_utils.unreplicate(state.step)
                    train_lr = scheduler_fn(state_step - 1)
                    train_time = time.time() - train_start_time

                    common_metrics = {
                        "epoch": epoch,
                        "step": state_step.item(),
                        "train_lr": train_lr.item(),  # type: ignore
                        "train_time": train_time,
                    }

                    # write train metrics
                    self.log_metrics(
                        train_metrics,
                        step=self.global_step,
                        prefix="train",
                    )
                    # write common metrics
                    self.logger.log_metrics(common_metrics, step=self.global_step)
                    # clear train metrics buffer
                    train_metrics = []

                if current_step % self.config.eval_steps == 0:
                    metrics = self.validate(state)
                    self.log_metrics(
                        metrics,
                        step=self.global_step,
                        prefix="valid",
                    )

                if current_step % self.config.save_steps == 0:
                    self.save_checkpoint(state, self.config.output_dir)

    def validate(self, state: TrainState) -> List[Dict[str, jnp.ndarray]]:
        valid_dataloader = self.data_module.valid_dataloader()
        valid_samples = len(valid_dataloader)
        valid_metrics = []

        for batch in tqdm(
            valid_dataloader,
            total=valid_samples,
            desc="Validating...",
        ):
            # accumulate validation metrics
            metrics = self.model.valid_step(state, batch)
            valid_metrics.append(metrics)

        return valid_metrics

    def configure_optimizers(
        self,
    ) -> Tuple[optax.GradientTransformation, optax.Schedule]:
        def training_steps() -> int:
            total_steps = len(self.data_module.dataset["train"])
            batch_size = self.data_module.config.train_batch_size * jax.device_count()
            num_epochs = self.config.max_epochs
            num_train_steps = (total_steps // batch_size) * num_epochs
            return num_train_steps

        cfg = self.optimizer_config
        cfg.scheduler.train_steps = training_steps()
        optimizer = AutoOptimizer.from_config(cfg)
        scheduler = AutoScheduler.from_config(cfg.scheduler)

        return optimizer, scheduler

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

        # unwrap state step scalar
        state_step = state.step
        if isinstance(state_step, jnp.ndarray):
            state_step = state_step.item()

        # save optimizer weights
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))  # type: ignore

        # save the training state
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            training_state = {
                "step": state_step,
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
            assert isinstance(opt_state, type(state.opt_state))

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

    def log_metrics(
        self,
        metrics: List[Dict[str, jnp.ndarray]],
        step: int,
        prefix: str,
    ) -> None:
        for idx, metric in enumerate(metrics):
            current_step = step - len(metrics) + idx + 1
            metric = self.prepare_metrics(metric, prefix)
            self.logger.log_metrics(metric, step=current_step)

    def prepare_metrics(
        self,
        metrics: Dict[str, jnp.ndarray],
        prefix: str,
    ) -> Dict[str, float]:
        return {f"{prefix}_{name}": value.item() for name, value in metrics.items()}
