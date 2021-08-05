from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import datasets
import jax
import numpy as np
from git_t5.data import (
    DataCollatorForT5MLM,
    DataLoader,
    compute_input_and_target_lengths,
    prepare_dataset,
)
from omegaconf import MISSING
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

if TYPE_CHECKING:
    from .trainer import T5Trainer
else:
    T5Trainer = Any


def tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    column: str,
) -> Callable[..., Dict[str, Union[List[List[int]], np.ndarray]]]:
    def wrap_fn(
        examples: Dict[str, List[str]]
    ) -> Dict[str, Union[List[List[int]], np.ndarray]]:
        return tokenizer(examples[column], return_attention_mask=False)  # type: ignore

    return wrap_fn


@dataclass
class DataModuleConfig:
    pass


@dataclass
class T5DataModuleConfig(DataModuleConfig):
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_column: Optional[str] = None
    model_path: Optional[str] = MISSING  # derive from model config
    tokenizer_path: Optional[str] = MISSING  # derive from model config
    use_fast_tokenizer: bool = MISSING  # derive from model config
    cache_dir: Optional[str] = MISSING  # derive from model config
    overwrite_cache: bool = False
    validation_size: float = 0.05
    max_sequence_length: Optional[int] = None
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: Optional[int] = None
    mlm_probability: float = 0.15
    mean_noise_span_length: float = 3.0
    decoder_start_token_id: int = MISSING
    seed: int = MISSING  # derive from model config


@dataclass
class T5DataModule:
    config: T5DataModuleConfig
    dataset: datasets.DatasetDict
    tokenizer: PreTrainedTokenizerBase
    data_collator: DataCollatorForT5MLM
    max_sequence_length: int
    input_length: int
    target_length: int
    trainer: Optional[T5Trainer] = None

    @classmethod
    def from_config(cls, config: T5DataModuleConfig) -> "T5DataModule":
        tokenizer = cls.load_tokenizer(config)
        max_sequence_length = config.max_sequence_length or VERY_LARGE_INTEGER
        max_sequence_length = min(max_sequence_length, tokenizer.model_max_length)

        input_length, target_length = compute_input_and_target_lengths(
            max_sequence_length,
            noise_density=config.mlm_probability,
            mean_noise_span_length=config.mean_noise_span_length,
            extra_tokens_per_span_inputs=1,
            extra_tokens_per_span_targets=1,
        )

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        sentinel_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")  # type: ignore
        if eos_token_id is None:
            raise ValueError("Tokenizer must have an existing `eos_token_id` value.")
        if pad_token_id is None:
            raise ValueError("Tokenizer must have an existing `pad_token_id` value.")
        if sentinel_token_id is None:
            raise ValueError("Tokenizer must have an existing `eos_token_id` value.")

        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=config.mlm_probability,
            mean_noise_span_length=config.mean_noise_span_length,
            input_length=max_sequence_length,
            target_length=target_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sentinel_token_id=sentinel_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
        )

        dataset = cls.load_dataset(config)
        dataset = cls.prepare_dataset(config, dataset, tokenizer, input_length)

        return T5DataModule(
            config,
            dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_sequence_length=max_sequence_length,
            input_length=input_length,
            target_length=target_length,
        )

    def train_dataloader(self) -> DataLoader:
        total_batch_size = self.config.train_batch_size * jax.device_count()
        return DataLoader(
            self.dataset["train"],
            batch_size=total_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            seed=self.config.seed,
        )

    def valid_dataloader(self) -> DataLoader:
        total_batch_size = self.config.eval_batch_size * jax.device_count()
        return DataLoader(
            self.dataset["validation"],
            batch_size=total_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            seed=None,
        )

    @classmethod
    def prepare_dataset(
        cls,
        config: T5DataModuleConfig,
        dataset: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        input_length: int,
    ) -> datasets.DatasetDict:
        if config.dataset_column is None:
            raise ValueError(
                "You must provide a `dataset_column` to specify which column of the dataset to use."
            )

        dataset = prepare_dataset(
            dataset,
            tokenize_fn(tokenizer, config.dataset_column),
            input_length=input_length,
            batch_size=128,
            load_from_cache_file=not config.overwrite_cache,
            num_workers=config.num_workers,
        )

        return dataset

    @classmethod
    def load_dataset(cls, config: T5DataModuleConfig) -> datasets.DatasetDict:
        if config.dataset_name is not None:
            dataset = datasets.load_dataset(
                config.dataset_name,
                config.dataset_config_name,
                cache_dir=config.cache_dir,
            )

            if not isinstance(dataset, datasets.DatasetDict):
                dataset = datasets.DatasetDict(train=dataset)

            if "validation" not in dataset.keys():
                valid_percentage = int(config.validation_size * 100)
                dataset["validation"] = datasets.load_dataset(
                    config.dataset_name,
                    config.dataset_config_name,
                    split=f"train[:{valid_percentage}%]",
                    cache_dir=config.cache_dir,
                )
                dataset["train"] = datasets.load_dataset(
                    config.dataset_name,
                    config.dataset_config_name,
                    split=f"train[{valid_percentage}%:]",
                    cache_dir=config.cache_dir,
                )
        elif config.dataset_path is not None:
            dataset = datasets.load_from_disk(config.dataset_path)
            if not isinstance(dataset, datasets.DatasetDict):
                dataset = datasets.DatasetDict(train=dataset)

            if "validation" not in dataset.keys():
                dataset = dataset["train"].train_test_split(
                    test_size=config.validation_size,
                    load_from_cache_file=not config.overwrite_cache,
                )
                dataset["validation"] = dataset.pop("test")
        else:
            raise ValueError("`dataset_name` or `dataset_path` must be specified.")

        return dataset

    @classmethod
    def load_tokenizer(cls, config: T5DataModuleConfig) -> PreTrainedTokenizerBase:
        if config.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path,
                cache_dir=config.cache_dir,
                use_fast=config.use_fast_tokenizer,
            )
        elif config.model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                cache_dir=config.cache_dir,
                use_fast=config.use_fast_tokenizer,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using `tokenizer_path`."
            )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return tokenizer
