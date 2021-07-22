from dataclasses import dataclass

import hydra
from git_t5.core import (
    SentencePieceTokenizer,
    SentencePieceTokenizerConfig,
    SentencePieceTrainer,
    SentencePieceTrainerConfig,
)
from hydra.core.config_store import ConfigStore

from .config import DefaultConfig, register_base_configs


@dataclass
class Config(DefaultConfig):
    tokenizer: SentencePieceTokenizerConfig = SentencePieceTokenizerConfig()
    tokenizer_trainer: SentencePieceTrainerConfig = SentencePieceTrainerConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=Config)
    cs.store(
        group="tokenizer",
        name="base_tokenizer",
        node=SentencePieceTokenizerConfig,
    )
    cs.store(
        group="tokenizer_trainer",
        name="base_tokenizer_trainer",
        node=SentencePieceTrainerConfig,
    )


@hydra.main(config_path="../../conf", config_name="config_tokenizer")
def hydra_entry(cfg: Config) -> None:
    tokenizer = SentencePieceTokenizer(cfg.tokenizer)
    trainer = SentencePieceTrainer(cfg.tokenizer_trainer)
    trainer.train(tokenizer)


def main() -> None:
    register_base_configs()
    register_configs()
    hydra_entry()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
