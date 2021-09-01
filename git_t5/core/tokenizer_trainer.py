import os
import shutil
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import datasets
from omegaconf import MISSING
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from transformers import T5TokenizerFast

from .tokenizer_model import (
    SentencePieceTokenizer,
    SentencePieceTokenizerConfig,
    TokenizerConfig,
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


@dataclass
class TokenizerTrainerConfig:
    save_directory: str = MISSING
    tokenizer: TokenizerConfig = TokenizerConfig()


@dataclass
class SentencePieceTrainerConfig(TokenizerTrainerConfig):
    dataset_column: str = MISSING
    dataset_path: Optional[str] = None
    dataset_files: Optional[List[str]] = None
    tokenizer: SentencePieceTokenizerConfig = SentencePieceTokenizerConfig()


@dataclass
class SentencePieceTrainer:
    config: SentencePieceTrainerConfig
    tokenizer: SentencePieceTokenizer

    @classmethod
    def from_config(cls, config: SentencePieceTrainerConfig) -> "SentencePieceTrainer":
        tokenizer = SentencePieceTokenizer(config.tokenizer)
        return SentencePieceTrainer(config, tokenizer)

    def train(self) -> None:
        if self.config.dataset_path:
            dataset = datasets.load_from_disk(self.config.dataset_path)
            if isinstance(dataset, datasets.DatasetDict):
                dataset = dataset["train"]

            dataset_iter = batch_iterator(dataset, self.config.dataset_column)
            self.tokenizer.train_from_iterator(dataset_iter)
        elif self.config.dataset_files:
            self.tokenizer.train_from_files(self.config.dataset_files)
        else:
            raise ValueError("`dataset_path` or `dataset_files` must be specified.")

        # save pre-trained rust tokenizer
        tokenizer_file = save(self.tokenizer, self.config.save_directory)

        # create a fast tokenizer wrapper around rust backend
        tokenizer_fast = T5TokenizerFast(
            vocab_file=None,
            tokenizer_file=tokenizer_file,
            model_max_length=self.tokenizer.config.model_max_length,
        )

        # concatenate extra_ids and user defined additional_special_tokens
        special_tokens = self.tokenizer.config.additional_special_tokens
        additional_special_tokens = tokenizer_fast.additional_special_tokens
        additional_special_tokens += list(special_tokens)

        tokenizer_fast.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}  # type: ignore
        )

        # save t5 fast tokenizer in a compact format
        tokenizer_fast.save_pretrained(
            save_directory=self.config.save_directory,
            legacy_format=False,
        )
