from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PreTrainedTokenizerConfig:
    tokenizer_path: str = MISSING
    use_fast: bool = True
