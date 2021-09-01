import copy
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import datasets as hfds
import jax
import numpy as np
from git_t5.core import HFDatasetConfig, LocalDatasetConfig, MultitaskDatasetConfig
from git_t5.data import (
    DataCollatorForT5MLM,
    Dataset,
    T5Dataset,
    T5MultitaskDataset,
    compute_input_and_target_lengths,
    prepare_dataset,
)
from git_t5.utils import resolve_object, stack_mappings
from omegaconf import MISSING
from torch.utils import data
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

if TYPE_CHECKING:
    from git_t5.cli.train_model import Config
    from git_t5.core.trainer import T5Trainer
else:
    Config = Any
    T5Trainer = Any

T = typing.TypeVar("T")


def tokenize_fn(
    tokenizer: PreTrainedTokenizerBase,
    column: str,
) -> Callable[..., Dict[str, Union[List[List[int]], np.ndarray]]]:
    def wrap_fn(
        examples: Dict[str, List[str]]
    ) -> Dict[str, Union[List[List[int]], np.ndarray]]:
        return tokenizer(
            examples[column],
            truncation=False,
            return_attention_mask=False,
        )  # type: ignore

    return wrap_fn


def select_subset(
    dataset: hfds.Dataset,
    size: Union[float, int],
    seed: Optional[int] = None,
) -> hfds.Dataset:
    num_samples: int
    if isinstance(size, int) or size > 1:
        num_samples = min(int(size), len(dataset))
    else:
        num_samples = int(len(dataset) * size)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), (num_samples,))

    return dataset.select(indices)


def _collate_fn(samples: List[T]) -> T:
    assert len(samples) == 1
    return samples[0]


@dataclass
class DataModuleConfig:
    pass


@dataclass
class T5DataModuleConfig(DataModuleConfig):
    validation_size: float = 0.05
    max_sequence_length: Optional[int] = None
    train_batch_size: int = 8
    valid_batch_size: int = 8
    num_proc: Optional[int] = None
    num_workers: Optional[int] = None
    mlm_probability: float = 0.15
    mean_noise_span_length: float = 3.0
    decoder_start_token_id: int = MISSING
    limit_train_size: float = 1.0
    limit_valid_size: float = 1.0


@dataclass
class T5DataModule:
    config: Config
    datasets: Dict[str, Dataset]
    tokenizer: PreTrainedTokenizerBase
    data_collator: DataCollatorForT5MLM
    max_sequence_length: int
    input_length: int
    target_length: int
    trainer: Optional[T5Trainer] = None

    @classmethod
    def from_config(cls, config: Config) -> "T5DataModule":
        tokenizer = cls.load_tokenizer(config)
        max_sequence_length = config.data.max_sequence_length or VERY_LARGE_INTEGER
        max_sequence_length = min(max_sequence_length, tokenizer.model_max_length)

        input_length, target_length = compute_input_and_target_lengths(
            max_sequence_length,
            noise_density=config.data.mlm_probability,
            mean_noise_span_length=config.data.mean_noise_span_length,
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
            noise_density=config.data.mlm_probability,
            mean_noise_span_length=config.data.mean_noise_span_length,
            input_length=max_sequence_length,
            target_length=target_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sentinel_token_id=sentinel_token_id,
            decoder_start_token_id=config.data.decoder_start_token_id,
        )

        datasets = cls.load_datasets(config, tokenizer, input_length)
        datasets = stack_mappings(datasets)
        datasets = {
            "train": cls.get_dataset(
                datasets["train"],
                batch_size=config.data.train_batch_size,
                collate_fn=data_collator,
                shuffle=True,
                drop_last=True,
                seed=config.training.seed,
            ),
            "valid": cls.get_dataset(
                datasets["validation"],
                batch_size=config.data.valid_batch_size,
                collate_fn=data_collator,
                shuffle=False,
                drop_last=True,
                seed=None,
            ),
        }

        return T5DataModule(
            config,
            datasets=datasets,
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_sequence_length=max_sequence_length,
            input_length=input_length,
            target_length=target_length,
        )

    def train_dataloader(self) -> data.DataLoader:
        num_workers = self.config.data.num_workers or 0
        return data.DataLoader(
            self.datasets["train"],
            collate_fn=_collate_fn,
            num_workers=num_workers,
        )

    def valid_dataloader(self) -> data.DataLoader:
        num_workers = self.config.data.num_workers or 0
        return data.DataLoader(
            self.datasets["valid"],
            collate_fn=_collate_fn,
            num_workers=num_workers,
        )

    @classmethod
    def get_dataset(
        cls,
        datasets: List[hfds.Dataset],
        batch_size: int,
        collate_fn: typing.Callable[..., Dict[str, np.ndarray]],
        shuffle: bool,
        drop_last: bool,
        seed: Optional[int],
    ) -> Dataset:
        batch_size = batch_size * jax.device_count()

        if len(datasets) == 1:
            dataset = T5Dataset(
                datasets[0],
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
            )
        else:
            dataset = T5MultitaskDataset(
                datasets,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
            )

        return dataset

    @classmethod
    def prepare_dataset(
        cls,
        dataset: hfds.DatasetDict,
        config: Config,
        tokenizer: PreTrainedTokenizerBase,
        input_length: int,
    ) -> hfds.DatasetDict:
        if config.dataset.column_name is None:
            raise ValueError(
                "You must provide a `column_name` to specify which column of the dataset to use."
            )

        dataset = prepare_dataset(
            dataset,
            tokenize_fn(tokenizer, config.dataset.column_name),
            input_length=input_length,
            batch_size=128,
            load_from_cache_file=not config.training.overwrite_cache,
            num_proc=config.data.num_proc,
        )

        # limit preprocessed dataset if needed
        dataset = cls.limit_dataset(dataset, config)

        return dataset

    @classmethod
    def load_datasets(
        cls,
        config: Config,
        tokenizer: PreTrainedTokenizerBase,
        input_length: int,
    ) -> List[Dict[str, hfds.Dataset]]:
        dataset_config = resolve_object(config.dataset)
        datasets: List[Dict[str, hfds.Dataset]] = []
        if isinstance(dataset_config, MultitaskDatasetConfig):
            dataset_config.resolve()
            for _, dataset_config in dataset_config.tasks.items():
                config = copy.deepcopy(config)
                config.dataset = dataset_config
                dataset = cls.load_dataset(config)
                dataset = cls.prepare_dataset(dataset, config, tokenizer, input_length)
                datasets.append(dataset)
        else:
            dataset = cls.load_dataset(config)
            dataset = cls.prepare_dataset(dataset, config, tokenizer, input_length)
            datasets.append(dataset)

        return datasets

    @classmethod
    def load_dataset(cls, config: Config) -> hfds.DatasetDict:
        dataset_config = resolve_object(config.dataset)
        if isinstance(dataset_config, LocalDatasetConfig):
            dataset = hfds.load_from_disk(dataset_config.dataset_path)
            if not isinstance(dataset, hfds.DatasetDict):
                dataset = hfds.DatasetDict(train=dataset)

            if "validation" not in dataset.keys():
                dataset = dataset["train"].train_test_split(
                    test_size=config.data.validation_size,
                    load_from_cache_file=not config.training.overwrite_cache,
                )
                dataset["validation"] = dataset.pop("test")
        elif isinstance(dataset_config, HFDatasetConfig):
            dataset = hfds.load_dataset(
                dataset_config.dataset_name,
                dataset_config.dataset_config,
                cache_dir=config.training.cache_dir,
            )

            if not isinstance(dataset, hfds.DatasetDict):
                dataset = hfds.DatasetDict(train=dataset)

            if "validation" not in dataset.keys():
                valid_percentage = int(config.data.validation_size * 100)
                dataset["validation"] = hfds.load_dataset(
                    dataset_config.dataset_name,
                    dataset_config.dataset_config,
                    split=f"train[:{valid_percentage}%]",
                    cache_dir=config.training.cache_dir,
                )
                dataset["train"] = hfds.load_dataset(
                    dataset_config.dataset_name,
                    dataset_config.dataset_config,
                    split=f"train[{valid_percentage}%:]",
                    cache_dir=config.training.cache_dir,
                )
        else:
            raise ValueError("Unknown dataset type provided.")

        # limit loaded dataset if needed
        dataset = cls.limit_dataset(dataset, config)

        return dataset

    @classmethod
    def limit_dataset(
        cls,
        dataset: hfds.DatasetDict,
        config: Config,
    ) -> hfds.DatasetDict:
        if config.data.limit_train_size != 1:
            dataset["train"] = select_subset(
                dataset["train"],
                config.data.limit_train_size,
                config.training.seed,
            )

        if config.data.limit_valid_size != 1:
            dataset["validation"] = select_subset(
                dataset["validation"],
                config.data.limit_valid_size,
                config.training.seed,
            )

        return dataset

    @classmethod
    def load_tokenizer(cls, config: Config) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.tokenizer_path,
            use_fast=config.tokenizer.use_fast,
            cache_dir=config.training.cache_dir,
        )

        return tokenizer
