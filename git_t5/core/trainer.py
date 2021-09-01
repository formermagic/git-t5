import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.serialization import from_bytes, to_bytes
from flax.training.common_utils import get_metrics
from flax.training.train_state import TrainState
from git_t5.core import AutoOptimizer, AutoScheduler
from git_t5.utils import rank_zero_only
from tqdm import tqdm

if TYPE_CHECKING:
    from git_t5.cli.train_model import Config
    from git_t5.core.data_module import T5DataModule
    from git_t5.core.logger import BaseLogger
    from git_t5.core.model import T5ModelForPreTraining
else:
    Config = Any
    T5DataModule = Any
    T5ModelForPreTraining = Any
    BaseLogger = Any


@dataclass
class TrainerConfig:
    pass


@dataclass
class T5TrainerConfig(TrainerConfig):
    max_epochs: int = 3
    logging_steps: int = 500
    save_steps: int = 500
    valid_steps: int = 2000


@dataclass
class T5Trainer:
    config: Config
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
        num_epochs = self.config.trainer.max_epochs
        train_dataloader = self.data_module.train_dataloader()
        train_samples = len(train_dataloader)

        optimizer, scheduler_fn = self.configure_optimizers()
        state = self.create_state(optimizer, self.config.training.checkpoint_dir)
        rng = jax.random.PRNGKey(self.config.training.seed)
        dropout_rng = jax.random.split(rng, jax.device_count())

        train_time = 0
        train_start_time = time.time()
        train_metrics = []

        for epoch in range(num_epochs):
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

                # increase the global train step
                self.global_step += 1

                if self.global_step % self.config.trainer.logging_steps == 0:
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    state_step = jax_utils.unreplicate(state.step)
                    train_lr = scheduler_fn(state_step - 1)
                    train_time = time.time() - train_start_time

                    common_metrics = {
                        "epoch": epoch,
                        "step": state_step,
                        "train_lr": train_lr,
                        "train_time": train_time,
                    }

                    # write train metrics
                    self.log_metrics(
                        train_metrics,
                        step=self.global_step,
                        prefix="train",
                    )

                    # write common metrics
                    self.log_metrics(
                        common_metrics,
                        step=self.global_step,
                        prefix=None,
                    )

                    # clear train metrics buffer
                    train_metrics = []

                if self.global_step % self.config.trainer.valid_steps == 0:
                    self.validate(state)

                if self.global_step % self.config.trainer.save_steps == 0:
                    self.save_checkpoint(state, self.config.training.output_dir)

    def validate(self, state: TrainState) -> None:
        valid_dataloader = self.data_module.valid_dataloader()
        valid_samples = len(valid_dataloader)
        valid_metrics = []

        for batch in tqdm(
            valid_dataloader,
            total=valid_samples,
            desc="Validating...",
        ):
            metrics = self.model.valid_step(state, batch)
            valid_metrics.append(metrics)

        valid_metrics = get_metrics(valid_metrics)
        valid_metrics = jax.tree_map(jnp.mean, valid_metrics)

        self.log_metrics(
            valid_metrics,
            step=self.global_step,
            prefix="valid",
        )

    def configure_optimizers(
        self,
    ) -> Tuple[optax.GradientTransformation, optax.Schedule]:
        def training_steps() -> int:
            total_steps = len(self.data_module.datasets["train"])
            batch_size = self.data_module.config.data.train_batch_size
            total_batch_size = batch_size * jax.device_count()
            num_epochs = self.config.trainer.max_epochs
            num_train_steps = (total_steps // total_batch_size) * num_epochs
            return num_train_steps

        cfg = self.config.optimizer
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
            push_to_hub=self.config.training.push_to_hub,
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
        metrics: Union[List[Dict[str, float]], Dict[str, float]],
        step: int,
        prefix: Optional[str] = None,
    ) -> None:
        device_metrics = metrics
        if not isinstance(device_metrics, list):
            device_metrics = [device_metrics]

        for idx, scalar_metrics in enumerate(device_metrics):
            current_step = step - len(device_metrics) + idx + 1
            scalar_metrics = self.prepare_metrics(scalar_metrics, prefix)
            self.logger.log_metrics(scalar_metrics, step=current_step)

    def prepare_metrics(
        self,
        metrics: Dict[str, float],
        prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        prefix = prefix + "_" if prefix else ""
        return {f"{prefix}{name}": value for name, value in metrics.items()}
