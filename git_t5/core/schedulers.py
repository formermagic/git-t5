from dataclasses import dataclass
from typing import Optional

import chex
import jax.numpy as jnp
import optax
from git_t5.utils import resolve_object


def inverse_square_root_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> optax.Schedule:
    """Constructs a schedule with inverse square root transition from init to end value.
    Args:
      init_value: initial value for the scalar to be annealed.
      end_value: end value of the scalar to be annealed.
      transition_steps: number of steps over which annealing takes place,
        the scalar starts changing at `transition_begin` steps and completes
        the transition by `transition_begin + transition_steps` steps.
        If `transition_steps <= 0`, then the entire annealing process is disabled
        and the value is held fixed at `init_value`.
      transition_begin: must be positive. After how many steps to start annealing
        (before this many steps the scalar value is held fixed at `init_value`).
    Returns:
      schedule: A function that maps step counts to values.
    """

    if transition_steps <= 0:
        return lambda _: init_value

    if transition_begin < 0:
        transition_begin = 0

    decay_factor = (init_value - end_value) * transition_begin ** 0.5

    def schedule(count: chex.Numeric) -> chex.Numeric:
        return decay_factor * 1 / jnp.sqrt(count + transition_begin)

    return schedule


@dataclass
class SchedulerConfig:
    train_steps: Optional[int] = None
    learning_rate: Optional[float] = None


@dataclass
class PolynomialSchedulerConfig(SchedulerConfig):
    init_learning_rate: float = 0.0
    warmup_steps: int = 0
    power: float = 1.0


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    init_learning_rate: float = 0.0
    warmup_steps: int = 0


@dataclass
class InverseSquareRootSchedulerConfig(SchedulerConfig):
    warmup_steps: int = 0


@dataclass
class ConstantSchedulerConfig(SchedulerConfig):
    init_learning_rate: float = 0.0
    warmup_steps: int = 0


def polynomial_scheduler(config: PolynomialSchedulerConfig) -> optax.Schedule:
    if config.train_steps is None:
        raise ValueError("`train_steps` must be specified.")
    if config.learning_rate is None:
        raise ValueError("`learning_rate` must be specified.")

    warmup_fn = optax.linear_schedule(
        init_value=config.init_learning_rate,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.polynomial_schedule(
        init_value=config.learning_rate,
        end_value=config.init_learning_rate,
        power=config.power,
        transition_steps=config.train_steps - config.warmup_steps,
    )
    scheduler_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    return scheduler_fn


def linear_scheduler(config: LinearSchedulerConfig) -> optax.Schedule:
    if config.train_steps is None:
        raise ValueError("`train_steps` must be specified.")
    if config.learning_rate is None:
        raise ValueError("`learning_rate` must be specified.")

    warmup_fn = optax.linear_schedule(
        init_value=config.init_learning_rate,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.init_learning_rate,
        transition_steps=config.train_steps - config.warmup_steps,
    )
    scheduler_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    return scheduler_fn


def inverse_square_root_scheduler(
    config: InverseSquareRootSchedulerConfig,
) -> optax.Schedule:
    if config.train_steps is None:
        raise ValueError("`train_steps` must be specified.")
    if config.learning_rate is None:
        raise ValueError("`learning_rate` must be specified.")

    warmup_fn = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = inverse_square_root_schedule(
        init_value=config.learning_rate,
        end_value=0.0,
        transition_steps=config.train_steps,
        transition_begin=config.warmup_steps,
    )
    scheduler_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    return scheduler_fn


def constant_scheduler(config: ConstantSchedulerConfig) -> optax.Schedule:
    if config.train_steps is None:
        raise ValueError("`train_steps` must be specified.")
    if config.learning_rate is None:
        raise ValueError("`learning_rate` must be specified.")

    warmup_fn = optax.linear_schedule(
        init_value=config.init_learning_rate,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.learning_rate,
        transition_steps=config.train_steps - config.warmup_steps,
    )
    scheduler_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    return scheduler_fn


class AutoScheduler:
    @staticmethod
    def from_config(config: SchedulerConfig) -> optax.Schedule:
        config = resolve_object(config)
        scheduler: optax.Schedule
        if isinstance(config, PolynomialSchedulerConfig):
            scheduler = polynomial_scheduler(config)
        elif isinstance(config, LinearSchedulerConfig):
            scheduler = linear_scheduler(config)
        elif isinstance(config, InverseSquareRootSchedulerConfig):
            scheduler = inverse_square_root_scheduler(config)
        elif isinstance(config, ConstantSchedulerConfig):
            scheduler = constant_scheduler(config)
        else:
            raise ValueError(f"Unknown scheduler type: {type(config)}.")
        return scheduler
