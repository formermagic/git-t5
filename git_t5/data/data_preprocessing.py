import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import datasets
import numpy as np

Dataset = TypeVar("Dataset", bound=Union[datasets.Dataset, datasets.DatasetDict])


def column_names(dataset: Any) -> List[str]:
    if isinstance(dataset, datasets.DatasetDict):
        values = dataset["train"].column_names
    elif isinstance(dataset, datasets.Dataset):
        values = dataset.column_names
    else:
        raise ValueError("Unsupported dataset type.")
    return values


def pad_batch(
    sequences: List[Union[List[int], np.ndarray]],
    pad: int,
    pad_first: bool = False,
) -> np.ndarray:
    batch_size = len(sequences)
    max_length = max(len(sequence) for sequence in sequences)
    batch = np.full((batch_size, max_length), pad, dtype=np.int32)
    for idx, sequence in enumerate(sequences):
        sequence = sequence[:-1]
        length = len(sequence)
        if pad_first:
            batch[idx, -length:] = sequence
        else:
            batch[idx, :length] = sequence
    return batch


def select_random_chunk(
    dataset: Dataset,
    feature_key: str,
    max_sequence_length: int,
    seed: Optional[int] = None,
    uniform_random_start: bool = False,
    load_from_cache_file: bool = True,
    num_proc: Optional[int] = None,
) -> Dataset:
    if max_sequence_length is None:
        raise ValueError("Must specify `max_sequence_length`.")

    def map_fn(
        features: Dict[str, Union[List[int], np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        tokens = np.asarray(features[feature_key])
        num_tokens = np.size(tokens)
        rng = np.random.default_rng(seed)
        if uniform_random_start:
            start = rng.integers(
                low=-max_sequence_length + 1, high=num_tokens, dtype=np.int32
            )
            end = np.minimum(start + max_sequence_length, num_tokens)
            start = np.maximum(start, 0)
        else:
            num_segments = np.ceil(np.divide(num_tokens, max_sequence_length))
            num_segments = num_segments.astype(np.int32)
            start = rng.integers(low=0, high=num_segments, dtype=np.int32)
            end = np.minimum(start + max_sequence_length, num_tokens)
        chunk = {feature_key: tokens[start:end]}
        return chunk

    dataset = dataset.filter(
        lambda x: np.not_equal(np.size(x[feature_key]), 0).item(),
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    dataset = dataset.map(
        map_fn,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )

    return dataset


def split_tokens(
    dataset: Dataset,
    feature_key: str,
    batch_size: int,
    pad_token_id: int,
    min_tokens_per_segment: Optional[int],
    max_tokens_per_segment: int,
    seed: Optional[int] = None,
    drop_last: bool = True,
    load_from_cache_file: bool = True,
    num_proc: Optional[int] = None,
) -> Dataset:
    def map_fn(
        features: Dict[str, List[Union[List[int], np.ndarray]]]
    ) -> Dict[str, List[np.ndarray]]:
        # pad and roll the batch by the `feature_key`
        tokens = pad_batch(features[feature_key], pad_token_id)
        tokens = np.reshape(tokens, [-1])
        tokens = tokens[tokens != pad_token_id]
        num_tokens = np.size(tokens)

        # calculate the length of segments
        if min_tokens_per_segment is None:
            length: int = max_tokens_per_segment
        else:
            rng = np.random.default_rng(seed)
            length: int = (
                rng.uniform(
                    low=math.log(min_tokens_per_segment),
                    high=math.log(max_tokens_per_segment),
                    size=(1,),
                )
                .astype(np.int32)
                .item()
            )

        # split chunks into padded segments (batches)
        num_segments = np.ceil(np.divide(num_tokens, length))
        num_segments = num_segments.astype(np.int32)
        padding = num_segments * length - num_tokens
        padded = np.pad(tokens, [[0, padding]])

        outputs = np.reshape(padded, [-1, length])
        lengths = np.concatenate(
            [np.repeat(length, num_segments - 1), [length - padding]], axis=0
        )

        # unwrap the padded segments + drop the last rows if needed
        input_ids: List[np.ndarray] = []
        for output, length in zip(outputs, lengths):
            if drop_last and length != max_tokens_per_segment:
                continue

            input_ids.append(output[:length])

        return {feature_key: input_ids}

    dataset = dataset.map(
        map_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=column_names(dataset),
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )

    return dataset


def noise_span_to_unique_sentinel(
    tokens: np.ndarray,
    noise_mask: np.ndarray,
    sentinel_id: int,
    tokens_reversed: bool,
) -> np.ndarray:
    prev_token_is_noise = np.pad(noise_mask[:-1], [[1, 0]])
    first_noise_tokens = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)
    if tokens_reversed:
        sentinel = sentinel_id + 1 - np.cumsum(first_noise_tokens)
    else:
        sentinel = sentinel_id - 1 + np.cumsum(first_noise_tokens)
    tokens = np.where(first_noise_tokens, sentinel, tokens)
    return tokens[~subsequent_noise_tokens]


def nonnoise_span_to_unique_sentinel(
    tokens: np.ndarray,
    noise_mask: np.ndarray,
    sentinel_id: int,
    tokens_reversed: bool,
) -> np.ndarray:
    return noise_span_to_unique_sentinel(
        tokens,
        np.logical_not(noise_mask),
        sentinel_id,
        tokens_reversed,
    )


def prepare_dataset(
    dataset: Dataset,
    tokenize_fn: Callable[..., Dict[str, Union[List[List[int]], np.ndarray]]],
    input_length: int,
    batch_size: int,
    drop_last: bool = True,
    load_from_cache_file: bool = True,
    num_proc: Optional[int] = None,
) -> Dataset:
    ds = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=column_names(dataset),
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    ds = select_random_chunk(
        ds,
        feature_key="input_ids",
        max_sequence_length=65536,
        uniform_random_start=False,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    ds = split_tokens(
        ds,
        feature_key="input_ids",
        batch_size=batch_size,
        pad_token_id=0,
        min_tokens_per_segment=None,
        max_tokens_per_segment=input_length,
        drop_last=drop_last,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    return ds


def denoise(
    dataset: Dataset,
    noise_density: float,
    eos_token_id: int,
    noise_mask_fn: Callable[..., np.ndarray],
    inputs_fn: Callable[..., np.ndarray],
    targets_fn: Callable[..., np.ndarray],
    load_from_cache_file: bool = True,
    num_proc: Optional[int] = None,
) -> Dataset:
    def map_fn(
        features: Dict[str, Union[List[int], np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        tokens = features["input_ids"]
        noise_mask = noise_mask_fn(np.size(tokens), noise_density)

        input_ids = inputs_fn(tokens, noise_mask)
        input_ids = np.concatenate([input_ids, [eos_token_id]], axis=0)

        if targets_fn is not None:
            labels = targets_fn(tokens, noise_mask)
        else:
            labels = tokens
        labels = np.concatenate((labels, [eos_token_id]), axis=0)

        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(
        map_fn,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )

    return dataset


def compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float,
    mean_noise_span_length: float,
    extra_tokens_per_span_inputs: int,
    extra_tokens_per_span_targets: int,
) -> Tuple[int, int]:
    def compute_lengths(tokens_length: int) -> Tuple[int, int]:
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        return (
            num_nonnoise_tokens + num_noise_spans * extra_tokens_per_span_inputs + 1,
            num_noise_tokens + num_noise_spans * extra_tokens_per_span_targets + 1,
        )

    tokens_length = inputs_length
    while compute_lengths(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = compute_lengths(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length
