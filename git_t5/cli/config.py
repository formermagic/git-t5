from dataclasses import dataclass

from git_t5.core import (
    DataModuleConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TokenizerTrainerConfig,
    TrainerConfig,
)
from hydra.core.config_store import ConfigStore


@dataclass
class DefaultConfig:
    tokenizer: TokenizerConfig = TokenizerConfig()
    tokenizer_trainer: TokenizerTrainerConfig = TokenizerTrainerConfig()
    model: ModelConfig = ModelConfig()
    data: DataModuleConfig = DataModuleConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerConfig = TrainerConfig()
    logger: LoggerConfig = LoggerConfig()


def register_base_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultConfig)
    cs.store(
        group="tokenizer",
        name="base_tokenizer",
        node=TokenizerConfig,
    )
    cs.store(
        group="tokenizer_trainer",
        name="base_tokenizer_trainer",
        node=TokenizerTrainerConfig,
    )
    cs.store(
        group="model",
        name="base_model",
        node=ModelConfig,
    )
    cs.store(
        group="data",
        name="base_data",
        node=DataModuleConfig,
    )
    cs.store(
        group="optimizer",
        name="base_optimizer",
        node=OptimizerConfig,
    )
    cs.store(
        group="optimizer/scheduler",
        name="base_scheduler",
        node=SchedulerConfig,
    )
    cs.store(
        group="trainer",
        name="base_trainer",
        node=TrainerConfig,
    )
    cs.store(
        group="logger",
        name="base_logger",
        node=LoggerConfig,
    )
