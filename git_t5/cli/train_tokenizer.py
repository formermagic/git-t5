import os
from dataclasses import dataclass

import hydra
from git_t5.core import (
    SentencePieceTokenizerConfig,
    SentencePieceTrainer,
    SentencePieceTrainerConfig,
)
from hydra.core.config_store import ConfigStore

from .config import DefaultConfig, register_base_configs


@dataclass
class Config(DefaultConfig):
    tokenizer_trainer: SentencePieceTrainerConfig = SentencePieceTrainerConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=Config)
    cs.store(
        group="tokenizer_trainer",
        name="base_tokenizer_trainer",
        node=SentencePieceTrainerConfig,
    )
    cs.store(
        group="tokenizer_trainer/tokenizer",
        name="base_tokenizer",
        node=SentencePieceTokenizerConfig,
    )


@hydra.main(config_path="conf", config_name="config_tokenizer")
def hydra_entry(cfg: Config) -> None:
    trainer = SentencePieceTrainer.from_config(cfg.tokenizer_trainer)
    trainer.train()


def main() -> None:
    # enable rust iterators multithreading
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    register_base_configs()
    register_configs()
    hydra_entry()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
