from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from git_t5.data import DataCollatorForT5MLM
from transformers import PreTrainedTokenizerBase

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


def encode_input(
    tokenizer: PreTrainedTokenizerBase,
    input_text: str,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    extra_tokens_per_span_inputs: int = 1,
    extra_tokens_per_span_targets: int = 1,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
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

    encoding = tokenizer(
        input_text,
        truncation=False,
        return_attention_mask=False,
        return_length=True,
    )

    input_length = encoding.pop("length")
    input_length = input_length[0]
    input_length, target_length = compute_lengths(input_length)

    np.random.seed(seed)

    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        input_length=input_length,
        target_length=target_length,
        eos_token_id=tokenizer.eos_token_id,  # type: ignore
        pad_token_id=tokenizer.pad_token_id,  # type: ignore
        decoder_start_token_id=tokenizer.pad_token_id,  # type: ignore
        sentinel_token_id=tokenizer.convert_tokens_to_ids("<extra_id_0>"),  # type: ignore
    )

    batch = data_collator([encoding])  # type: ignore
    batch = {key: torch.tensor(val) for key, val in batch.items()}

    return batch, target_length
