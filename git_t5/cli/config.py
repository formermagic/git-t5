from dataclasses import dataclass

from git_t5.core import (
    DataModuleConfig,
    DatasetConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    PreTrainedTokenizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TokenizerTrainerConfig,
    TrainerConfig,
    TrainingConfig,
)
from hydra.core.config_store import ConfigStore


@dataclass
class DefaultConfig:
    tokenizer_trainer: TokenizerTrainerConfig = TokenizerTrainerConfig()
    tokenizer: PreTrainedTokenizerConfig = PreTrainedTokenizerConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    data: DataModuleConfig = DataModuleConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerConfig = TrainerConfig()
    training: TrainingConfig = TrainingConfig()
    logger: LoggerConfig = LoggerConfig()


def register_base_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultConfig)
    cs.store(
        group="tokenizer_trainer",
        name="base_tokenizer_trainer",
        node=TokenizerTrainerConfig,
    )
    cs.store(
        group="tokenizer_trainer/tokenizer",
        name="base_tokenizer",
        node=TokenizerConfig,
    )
    cs.store(
        group="tokenizer",
        name="base_pretrained_tokenizer",
        node=PreTrainedTokenizerConfig,
    )
    cs.store(
        group="dataset",
        name="base_dataset",
        node=DatasetConfig,
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
        group="training",
        name="base_training",
        node=TrainingConfig,
    )
    cs.store(
        group="logger",
        name="base_logger",
        node=LoggerConfig,
    )
