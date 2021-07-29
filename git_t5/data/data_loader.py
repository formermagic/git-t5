from itertools import zip_longest
from typing import Any, Callable, Dict, Iterator, List, Optional

import datasets
import jax
import numpy as np
from flax.training.common_utils import shard


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


class DataLoader:
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        collate_fn: Callable[..., Any],
        shuffle: bool,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.seed = seed

        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        batch = self.dataset[index * self.batch_size : (index + 1) * self.batch_size]
        if not isinstance(batch, dict):
            raise ValueError(f"Expected a dict, got {type(batch)}.")
        batch = flatten(batch)
        batch = self.collate_fn(batch)
        batch = jax.tree_map(shard, batch)
        return batch

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
