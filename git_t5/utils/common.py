from collections import defaultdict
from typing import Dict, List, TypeVar

K = TypeVar("K")
T = TypeVar("T")


def stack_mappings(mappings: List[Dict[K, T]]) -> Dict[K, List[T]]:
    def flatten(sequence: List[List[T]]) -> List[T]:
        return [item for subsequence in sequence for item in subsequence]

    result = defaultdict(list)
    keys = set(flatten([mapping.keys() for mapping in mappings]))  # type: ignore
    for mapping in mappings:
        for key in keys:
            result[key].append(mapping.get(key, None))

    return dict(result)
