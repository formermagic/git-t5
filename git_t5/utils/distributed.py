from functools import wraps
from typing import Callable, TypeVar

import jax

T = TypeVar("T", bound=Callable)


def rank_zero_only(fn: T) -> T:
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if jax.process_index() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn  # type: ignore
