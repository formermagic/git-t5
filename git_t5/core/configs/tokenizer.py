from dataclasses import dataclass

from omegaconf import MISSING
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class PreTrainedTokenizerConfig:
    tokenizer_path: str = MISSING
    use_fast: bool = True


class PreTrainedTokenizer:
    @classmethod
    def from_config(cls, config: PreTrainedTokenizerConfig) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            use_fast=config.use_fast,
        )
