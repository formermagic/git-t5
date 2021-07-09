from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from src.tokenizer_model import TokenizerConfig


@dataclass
class DefaultArgumentsConfig:
    pass


@dataclass
class DefaultConfig:
    arguments: DefaultArgumentsConfig = DefaultArgumentsConfig()
    tokenizer_model: TokenizerConfig = TokenizerConfig()


def register_base_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultConfig)
    cs.store(
        group="tokenizer_model",
        name="base_tokenizer_model",
        node=TokenizerConfig,
    )
    cs.store(
        group="arguments",
        name="base_arguments",
        node=DefaultArgumentsConfig,
    )
