import json
from typing import Any, Iterator, List, Optional, Union

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


# pylint: disable=too-many-arguments
class SentencePieceBPETokenizer(BaseTokenizer):
    def __init__(
        self,
        add_prefix_space: bool = False,
        dropout: Optional[float] = None,
        trim_offsets: bool = False,
        unk_token: Token = "<unk>",
        eos_token: Token = "</s>",
        pad_token: Token = "<pad>",
    ) -> None:
        # sentencepiece special tokens map
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
        }

        # sentencepiece special tokens list
        self.special_tokens_list: List[Token] = [""] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        # sentencepiece byte-level bpe tokenizer
        tokenizer = Tokenizer(BPE(dropout=dropout, unk_token=unk_token))
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
            add_prefix_space=add_prefix_space
        )
        # byte-level decoding similar to GPT-2
        tokenizer.decoder = decoders.ByteLevel()  # type: ignore
        # original sentencepiece post processing
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[
                (
                    self.special_tokens["eos"]["token"],
                    self.special_tokens["eos"]["id"],
                )
            ],
        )

        parameters = {
            "model": "ByteLevelBPE",
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
            "trim_offsets": trim_offsets,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(
        vocab_filename: str, merges_filename: str, **kwargs: Any
    ) -> "SentencePieceBPETokenizer":
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)  # type: ignore
        return SentencePieceBPETokenizer(vocab, merges, **kwargs)

    # pylint: disable=dangerous-default-value
    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Token] = [],
    ) -> None:
        """Train the model using the given files"""

        special_tokens = self.special_tokens_list + special_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        if isinstance(files, str):
            files = [files]

        self._tokenizer.train(files, trainer=trainer)
        self._add_token_id("unk")

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Token] = [],
    ) -> None:
        """Train the model using the given iterator"""

        special_tokens = self.special_tokens_list + special_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._add_token_id("unk")

    def _add_token_id(self, token: str) -> None:
        token_id = f"{token}_id"
        tokenizer_json = json.loads(self._tokenizer.to_str())
        tokenizer_json["model"][token_id] = self.special_tokens[token]["id"]
        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
