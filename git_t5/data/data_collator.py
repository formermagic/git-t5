from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from transformers import PreTrainedTokenizerBase

from .data_preprocessing import (
    noise_span_to_unique_sentinel,
    nonnoise_span_to_unique_sentinel,
)


def shift_tokens_right(
    input_ids: np.ndarray,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[..., 1:] = input_ids[..., :-1]
    shifted_input_ids[..., 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = np.where(
        shifted_input_ids == -100,
        pad_token_id,
        shifted_input_ids,
    )

    return shifted_input_ids


@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        eos_token_id: (:obj:`int`):
            The eos token id of the model.
        pad_token_id: (:obj:`int`):
            The pad token id of the model.
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model.
        sentinel_token_id: (:obj:`int):
            The first sentinel token id of the model.
        sentinel_tokens_reversed: (:obj:`bool):
            Whether sentinel tokens arranged in reverse order.
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    eos_token_id: int
    pad_token_id: int
    decoder_start_token_id: int
    sentinel_token_id: int
    sentinel_tokens_reversed: bool = False

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # add span corruption noise for each example
        examples = [self.denoise(example) for example in examples]

        # collate input examples
        batch = {
            column: np.array([examples[idx][column] for idx in range(len(examples))])
            for column in examples[0].keys()
        }

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        return batch

    def denoise(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        tokens = features["input_ids"]
        noise_mask = self.random_spans_noise_mask(np.size(tokens))

        input_ids = self.append_eos_token(
            noise_span_to_unique_sentinel(
                tokens,
                noise_mask,
                self.sentinel_token_id,
                self.sentinel_tokens_reversed,
            )
        )
        labels = self.append_eos_token(
            nonnoise_span_to_unique_sentinel(
                tokens,
                noise_mask,
                self.sentinel_token_id,
                self.sentinel_tokens_reversed,
            )
        )
        decoder_input_ids = shift_tokens_right(
            labels,
            self.pad_token_id,
            self.decoder_start_token_id,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

    def append_eos_token(self, tokens: np.ndarray) -> np.ndarray:
        return np.concatenate((tokens, [self.eos_token_id]), axis=0)

    def random_spans_noise_mask(self, length: int) -> np.ndarray:
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
