from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import optax
from git_t5.utils import resolve_object

from .schedulers import AutoScheduler, SchedulerConfig


@dataclass
class OptimizerConfig:
    scheduler: SchedulerConfig = SchedulerConfig()


@dataclass
class AdamConfig(OptimizerConfig):
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.001


@dataclass
class AdamWConfig(OptimizerConfig):
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.001
    weight_decay: float = 1e-4


@dataclass
class AdafactorConfig(OptimizerConfig):
    min_dim_size_to_factor: int = 128
    decay_rate: float = 0.8
    decay_offset: int = 0
    multiply_by_parameter_scale: bool = True
    clipping_threshold: Optional[float] = 1.0
    momentum: Optional[float] = None
    weight_decay_rate: Optional[float] = None
    dtype_momentum: str = "float32"
    eps: float = 1e-30
    factored: bool = True


@dataclass
class AdagradConfig(OptimizerConfig):
    initial_accumulator_value: float = 0.1
    eps: float = 1e-7


def adam(config: AdamConfig) -> optax.GradientTransformation:
    scheduler = AutoScheduler.from_config(config.scheduler)
    return optax.adam(
        learning_rate=scheduler,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
    )


def adamw(config: AdamWConfig) -> optax.GradientTransformation:
    scheduler = AutoScheduler.from_config(config.scheduler)
    return optax.adamw(
        learning_rate=scheduler,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        weight_decay=config.weight_decay,
    )


def adafactor(config: AdafactorConfig) -> optax.GradientTransformation:
    scheduler = AutoScheduler.from_config(config.scheduler)
    return optax.adafactor(
        learning_rate=scheduler,
        min_dim_size_to_factor=config.min_dim_size_to_factor,
        decay_rate=config.decay_rate,
        decay_offset=config.decay_offset,
        multiply_by_parameter_scale=config.multiply_by_parameter_scale,  # type: ignore
        clipping_threshold=config.clipping_threshold,
        momentum=config.momentum,
        dtype_momentum=getattr(jnp, config.dtype_momentum),
        weight_decay_rate=config.weight_decay_rate,
        eps=config.eps,
        factored=config.factored,
    )


def adagrad(config: AdagradConfig) -> optax.GradientTransformation:
    scheduler = AutoScheduler.from_config(config.scheduler)
    return optax.adagrad(
        learning_rate=scheduler,
        initial_accumulator_value=config.initial_accumulator_value,
        eps=config.eps,
    )


class AutoOptimizer:
    @staticmethod
    def from_config(config: OptimizerConfig) -> optax.GradientTransformation:
        config = resolve_object(config)
        optimizer: optax.GradientTransformation
        if isinstance(config, AdamConfig):
            optimizer = adam(config)
        elif isinstance(config, AdamWConfig):
            optimizer = adamw(config)
        elif isinstance(config, AdafactorConfig):
            optimizer = adafactor(config)
        elif isinstance(config, AdagradConfig):
            optimizer = adagrad(config)
        else:
            raise ValueError(f"Unknown optimizer type: {type(config)}.")
        return optimizer
