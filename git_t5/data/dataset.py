import itertools
import math
import typing
from abc import abstractmethod
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

import datasets as hfds
import jax
import more_itertools
import numpy as np
from flax.training.common_utils import shard
from torch.utils import data

T = typing.TypeVar("T")


def flatten(nested_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    result = []
    packed_values = zip_longest(*nested_dict.values())
    for packed_value in packed_values:
        flat_dict = {}
        for key, value in zip(nested_dict.keys(), packed_value):
            if value is None:
                continue
            flat_dict[key] = value
        result.append(flat_dict)
    return result


def chunked_iterator(
    iterable: Iterable[T], chunk_size: int, drop_last: bool
) -> Iterator[List[T]]:
    for chunk in more_itertools.chunked(iterable, chunk_size):
        if len(chunk) != chunk_size and drop_last:
            continue
        yield chunk


class Dataset(data.IterableDataset):
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        raise RuntimeError("`Dataset` subclasses do not support indexing.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        raise NotImplementedError


class T5Dataset(Dataset):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataset: hfds.Dataset,
        batch_size: int,
        collate_fn: Callable[..., Any],
        shuffle: bool,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self) -> int:
        instance_count = len(self.dataset)
        if self.drop_last or instance_count % self.batch_size == 0:
            return instance_count // self.batch_size
        return 1 + (instance_count // self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.shuffle:
            self.dataset = self.dataset.shuffle(self.seed)

        # get the total number of instances per epoch
        length = len(self.dataset)

        # calculate the number of batches per worker
        worker_info = data.get_worker_info()
        if worker_info is not None:
            num_workers = float(worker_info.num_workers) * self.batch_size
            per_worker = int(math.ceil(length / num_workers))
            iter_start = worker_info.id * per_worker
            iter_stop = min(iter_start + per_worker, length)
        else:
            iter_start = 0
            iter_stop = length

        # split the batched instances across the workers
        batches = chunked_iterator(self.dataset, self.batch_size, self.drop_last)
        batches = itertools.islice(batches, iter_start, iter_stop)
        for batch in batches:
            batch = self.collate_fn(batch)
            batch = jax.tree_map(shard, batch)
            yield batch


class T5MultitaskDataset(Dataset):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        datasets: Union[Dict[str, hfds.Dataset], List[hfds.Dataset]],
        batch_size: int,
        collate_fn: Callable[..., Any],
        shuffle: bool,
        drop_last: bool = True,
        seed: Optional[int] = None,
        limit_instances: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(datasets, list):
            datasets = {
                f"dataset_{idx}": dataset for idx, dataset in enumerate(datasets)
            }

        self.datasets = datasets
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.limit_instances = limit_instances

    def __len__(self) -> int:
        dataset_counts: Dict[str, int]
        if self.limit_instances is None:
            dataset_counts = {
                name: len(dataset) for name, dataset in self.datasets.items()
            }
        else:
            dataset_counts = self.get_task_num_instances(
                self.datasets,
                self.limit_instances,
            )

        return self.count_batches(dataset_counts)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.shuffle:
            self.datasets = {
                name: dataset.shuffle(self.seed)
                for name, dataset in self.datasets.items()
            }

        # get the joined instances across all datasets
        epoch_instances = self.get_epoch_instances()
        epoch_instances = more_itertools.roundrobin(*epoch_instances.values())

        # get the total number of instances per epoch
        epoch_counts = self.get_dataset_counts()
        length = sum(epoch_counts.values())

        # calculate the number of batches per worker
        worker_info = data.get_worker_info()
        if worker_info is not None:
            num_workers = float(worker_info.num_workers) * self.batch_size
            per_worker = int(math.ceil(length / num_workers))
            iter_start = worker_info.id * per_worker
            iter_stop = min(iter_start + per_worker, length)
        else:
            iter_start = 0
            iter_stop = length

        # split the batched instances across the workers
        batches = chunked_iterator(epoch_instances, self.batch_size, self.drop_last)
        batches = itertools.islice(batches, iter_start, iter_stop)
        for batch in batches:
            batch = self.collate_fn(batch)
            batch = jax.tree_map(shard, batch)
            yield batch

    def get_epoch_instances(self) -> Dict[str, Iterable[Dict[str, np.ndarray]]]:
        dataset_counts = self.get_dataset_counts()
        return {
            key: itertools.islice(self.datasets[key], count)
            for key, count in dataset_counts.items()
        }

    def get_dataset_counts(self) -> Dict[str, int]:
        dataset_counts: Dict[str, int]
        if self.limit_instances is None:
            dataset_counts = {
                name: len(dataset) for name, dataset in self.datasets.items()
            }
        else:
            dataset_counts = self.get_task_num_instances(
                self.datasets,
                self.limit_instances,
            )
        return dataset_counts

    def get_task_num_instances(
        self, datasets: Dict[str, hfds.Dataset], limit_instances: int
    ) -> Dict[str, int]:
        dataset_proportions = self.get_task_proportions(datasets)
        total_proportions = sum(dataset_proportions.values())
        return {
            key: math.floor(proportion * limit_instances / total_proportions)
            for key, proportion in dataset_proportions.items()
        }

    def get_task_proportions(
        self, datasets: Dict[str, hfds.Dataset]
    ) -> Dict[str, float]:
        return {key: 1 / len(datasets) for key in datasets}

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        instance_count = sum(dataset_counts.values())
        if self.drop_last or instance_count % self.batch_size == 0:
            return instance_count // self.batch_size
        return 1 + (instance_count // self.batch_size)
