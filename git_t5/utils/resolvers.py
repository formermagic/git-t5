from typing import TypeVar

from omegaconf import DictConfig

T = TypeVar("T")


def resolve_object(obj: T) -> T:
    if isinstance(obj, DictConfig):
        obj = obj._to_object()  # pylint: disable=protected-access
    return obj
