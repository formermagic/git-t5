import json
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

Token = Union[str, AddedToken]


@dataclass
class TokenizerConfig:
    vocab_size: int = 30_000
    special_tokens: List[str] = field(default_factory=list)
    additional_special_tokens: List[str] = field(default_factory=list)
    model_max_length: Optional[int] = None
    show_progress: bool = True


@dataclass
class SentencePieceTokenizerConfig(TokenizerConfig):
    dropout: Optional[float] = None
    add_prefix_space: bool = False
    trim_offsets: bool = False
    min_frequency: int = 2
    lowercase: bool = False
    remove_extra_spaces: bool = True
    unicode_normalizer: Optional[str] = "nfkc"
    unk_token: str = "<unk>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"


# pylint: disable=too-many-arguments
class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, config: SentencePieceTokenizerConfig) -> None:
        self.config = config

        # sentencepiece special tokens map
        self.special_tokens_map = {
            "pad": {"id": 0, "token": config.pad_token},
            "eos": {"id": 1, "token": config.eos_token},
            "unk": {"id": 2, "token": config.unk_token},
        }

        # sentencepiece special tokens list
        self.special_tokens: List[Token] = [""] * len(self.special_tokens_map)
        for token_dict in self.special_tokens_map.values():
            self.special_tokens[token_dict["id"]] = token_dict["token"]

        self.special_tokens += list(config.special_tokens)

        # sentencepiece byte-level bpe tokenizer
        tokenizer = Tokenizer(BPE(dropout=config.dropout, unk_token=config.unk_token))
        # original sentencepiece normalization
        tokenizer.normalizer = normalizers.Sequence(  # type: ignore
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
            ]
        )
        # byte-level pre-tokenization similar to GPT-2
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(  # type: ignore
            add_prefix_space=config.add_prefix_space
        )
        # byte-level decoding similar to GPT-2
        tokenizer.decoder = decoders.ByteLevel()  # type: ignore
        # original sentencepiece post processing
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single=f"$A {self.special_tokens_map['eos']['token']}",
            special_tokens=[
                (
                    self.special_tokens_map["eos"]["token"],
                    self.special_tokens_map["eos"]["id"],
                )
            ],
        )

        parameters = {
            "model": "ByteLevelBPE",
            "add_prefix_space": config.add_prefix_space,
            "dropout": config.dropout,
            "trim_offsets": config.trim_offsets,
        }

        super().__init__(tokenizer, parameters)

    def train_from_files(self, files: Union[str, List[str]]) -> None:
        """Train the model using the given files"""

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            show_progress=self.config.show_progress,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        if isinstance(files, str):
            files = [files]

        self._tokenizer.train(files, trainer=trainer)
        self._add_token_id("unk")

    def train_from_iterator(
        self, iterator: Union[Iterator[str], Iterator[Iterator[str]]]
    ) -> None:
        """Train the model using the given iterator"""

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            show_progress=self.config.show_progress,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._add_token_id("unk")

    def _add_token_id(self, token: str) -> None:
        token_id = f"{token}_id"
        tokenizer_json = json.loads(self._tokenizer.to_str())
        tokenizer_json["model"][token_id] = self.special_tokens_map[token]["id"]
        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
