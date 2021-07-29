import os
from abc import abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar

import wandb
from git_t5.utils import rank_zero_only
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:
    from .trainer import T5Trainer
else:
    T5Trainer = Any

T = TypeVar("T", bound=Callable)


def rank_zero_experiment(fn: T) -> T:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment  # type: ignore


class DummyExperiment:
    """Dummy experiment"""

    def nop(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Any:
        return self.nop

    def __getitem__(self, idx: int) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self


class BaseLogger:
    trainer: Optional[T5Trainer] = None

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        raise NotImplementedError()

    def _add_prefix(
        self, metrics: Dict[str, float], prefix: Optional[str]
    ) -> Dict[str, float]:
        if prefix is not None:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        return metrics


class WandbLogger(BaseLogger):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        run_id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        experiment: Optional[Run] = None,
        prefix: Optional[str] = "",
        **kwargs,
    ) -> None:
        anonymous_map = {
            True: "allow",
            False: None,
            None: None,
        }

        self._offline = offline
        self._experiment = experiment
        self._prefix = prefix
        self._wandb_init = dict(
            name=name,
            project=project,
            id=version or run_id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_map.get(anonymous),
        )

        self._wandb_init.update(**kwargs)

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            if wandb.run is None:
                self._experiment = wandb.init(**self._wandb_init)
            else:
                self._experiment = wandb.run

        self._experiment.define_metric("trainer/global_step")
        self._experiment.define_metric(
            "*",
            step_metric="trainer/global_step",
            step_sync=True,
        )

        return self._experiment

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        metrics = self._add_prefix(metrics, prefix=self._prefix)
        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step})
        else:
            self.experiment.log(metrics)


class DummyLogger(BaseLogger):
    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        print(metrics)
