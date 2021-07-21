from dataclasses import dataclass

from git_t5.tokenizer_model import TokenizerConfig
from git_t5.trainer import DataConfig, ModelConfig, TrainingConfig
from hydra.core.config_store import ConfigStore


@dataclass
class DefaultArgumentsConfig:
    pass


@dataclass
class DefaultConfig:
    arguments: DefaultArgumentsConfig = DefaultArgumentsConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()


def register_base_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultConfig)
    cs.store(
        group="tokenizer",
        name="base_tokenizer",
        node=TokenizerConfig,
    )
    cs.store(
        group="arguments",
        name="base_arguments",
        node=DefaultArgumentsConfig,
    )
    cs.store(
        group="model",
        name="base_model",
        node=ModelConfig,
    )
    cs.store(
        group="data",
        name="base_data",
        node=DataConfig,
    )
    cs.store(
        group="training",
        name="base_training",
        node=TrainingConfig,
    )
