import os
import shutil
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import datasets
import hydra
from git_t5.tokenizer_model import (
    BaseTokenizer,
    SentencePieceTokenizer,
    SentencePieceTokenizerConfig,
)
from hydra.core.config_store import ConfigStore
from transformers import T5TokenizerFast

from .config import DefaultArgumentsConfig, DefaultConfig, register_base_configs


@dataclass
class TokenizerTrainerArgs(DefaultArgumentsConfig):
    dataset_path: Optional[str] = None
    save_directory: Optional[str] = None


@dataclass
class Config(DefaultConfig):
    arguments: TokenizerTrainerArgs = TokenizerTrainerArgs()
    tokenizer: SentencePieceTokenizerConfig = SentencePieceTokenizerConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=Config)
    cs.store(
        group="tokenizer_model",
        name="base_tokenizer_model",
        node=SentencePieceTokenizerConfig,
    )
    cs.store(
        group="arguments",
        name="base_tokenizer_args",
        node=TokenizerTrainerArgs,
    )


def batch_iterator(
    dataset: datasets.Dataset,
    column_name: str,
    batch_size: int = 1024,
) -> Iterator[Iterator[str]]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][column_name]  # type: ignore


def save(tokenizer: BaseTokenizer, output_path: Union[str, os.PathLike]) -> str:
    # remove previous dir if one exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # make a new output directory
    os.makedirs(output_path, exist_ok=True)
    # save pretrained tokenizer file
    tokenizer_file = os.path.join(output_path, "tokenizer.json")
    tokenizer.save(tokenizer_file)
    return tokenizer_file


@hydra.main(config_path="../../conf", config_name="config_tokenizer")
def main(cfg: Config) -> None:
    if cfg.arguments.dataset_path is None:
        raise ValueError("Please, specify `arguments.dataset_path` argument.")

    dataset = datasets.load_from_disk(cfg.arguments.dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["train"]

    if cfg.arguments.save_directory is None:
        raise ValueError("Please, specify `arguments.save_directory` argument.")

    tokenizer = SentencePieceTokenizer(cfg.tokenizer)
    tokenizer.train_from_iterator(batch_iterator(dataset, "text"))
    tokenizer_file = save(tokenizer, cfg.arguments.save_directory)

    tokenizer = T5TokenizerFast(
        vocab_file=None,
        merges_file=None,
        tokenizer_file=tokenizer_file,
        model_max_length=cfg.tokenizer.model_max_length,
    )

    tokenizer.save_pretrained(
        save_directory=cfg.arguments.save_directory,
        legacy_format=False,
    )


if __name__ == "__main__":
    register_base_configs()
    register_configs()
    main()  # pylint: disable=no-value-for-parameter
