# Copyright (c) 2019-present, Facebook, Inc. and The FormerMagic Inc. team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import tokenize
from abc import abstractmethod
from enum import Enum
from io import BytesIO
from typing import Dict, List, Union

from sacrebleu.tokenizers import TokenizerV14International

PYTHON_CHAR2TOKEN = {
    "#": "<STOKEN0>",
    "\\n": "<STOKEN1>",
    '"""': "<STOKEN2>",
    "'''": "<STOKEN3>",
}

PYTHON_TOKEN2CHAR = {
    "<STOKEN0>": "#",
    "<STOKEN1>": "\\n",
    "<STOKEN2>": '"""',
    "<STOKEN3>": "'''",
}


class SpecialToken(Enum):
    INDENT = "<indent>"
    DEDENT = "<dedent>"
    NEWLINE = "<newline>"
    STR_NEWLINE = "<strnewline>"
    TAB_SYMBOL = "<tabsymbol>"
    SPACE = "▁"
    SPACE_TOKEN = "<spacetoken>"
    END_COM = "<endcom>"
    END_MARKER = "<endmarker>"


class Tokenizer:
    def __init__(
        self,
        special_tokens: List[str],
        char2token: Dict[str, str],
        token2char: Dict[str, str],
    ) -> None:
        self.special_tokens = special_tokens
        self.char2token = char2token
        self.token2char = token2char
        self.tokenizer_v14_international = TokenizerV14International()

    @abstractmethod
    def tokenize(
        self,
        text: str,
        keep_comments: bool,
        add_token_space: bool,
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def detokenize(
        self,
        tokens: Union[List[str], str],
        add_token_space: bool,
    ) -> str:
        raise NotImplementedError()

    def stringify(self, tokens: List[str], add_token_space: bool) -> str:
        """Joins tokens into line with or without spaces aroung special tokens."""
        line = " ".join(tokens)
        if not add_token_space:
            tokens_pattern = "|".join(self.special_tokens)
            line = re.sub(r"\s*(%s)\s*" % tokens_pattern, r"\1", line)
        return line

    def destringify(self, line: str, add_token_space: bool) -> str:
        """Adds spaces to special tokens in the tokenized line if they are missing."""
        if not add_token_space:
            tokens_pattern = "|".join(self.special_tokens)
            line = re.sub(r"\s*(%s)\s*" % tokens_pattern, r" \1 ", line)
        return line

    def process_string(
        self,
        token: str,
        is_comment: bool,
        use_bleu_tokenization: bool = False,
    ) -> str:
        if is_comment:
            token = re.sub(" +", " ", token)
            token = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", token)
            if len(re.sub(r"\W", "", token)) < 2:
                return ""

        token = token.replace(" ", f" {SpecialToken.SPACE.value} ")
        for char, special_token in self.char2token.items():
            token = token.replace(char, special_token)

        if token.startswith("<STOKEN0>"):
            if token.endswith("\n"):
                token = token[:-1]
            token += f" {SpecialToken.END_COM.value}"

        token = token.replace("\n", f" {SpecialToken.STR_NEWLINE.value} ")
        token = token.replace("\t", f" {SpecialToken.TAB_SYMBOL.value} ")
        token = re.sub(" +", " ", token)
        if use_bleu_tokenization:
            token = self.tokenizer_v14_international(token)
        token = re.sub(" +", " ", token)

        # split string prefix if one stands at the beginning
        regex = r"""([bruf]*) ((\"""|'''|"|')(?:(?!\3)(?:\\.|[^\\]))*\3)"""
        token = re.sub(regex, r"\1\2", token)

        for special_token, char in self.token2char.items():
            token = token.replace(special_token, char)

        token = token.replace("\r", "")

        return token


class PyTokenizer(Tokenizer):
    SPECIAL_TOKENS: List[str] = [token.value for token in SpecialToken]

    def __init__(self):
        super().__init__(
            special_tokens=self.SPECIAL_TOKENS,
            char2token=PYTHON_CHAR2TOKEN,
            token2char=PYTHON_TOKEN2CHAR,
        )

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def tokenize(
        self,
        text: str,
        keep_comments: bool = False,
        add_token_space: bool = True,
    ) -> str:
        # pylint: disable=too-many-nested-blocks
        try:
            assert isinstance(text, str)
            text = text.replace(r"\r", "")
            tokens = []

            try:
                text_bytes = BytesIO(text.encode("utf-8"))
                iterator = tokenize.tokenize(text_bytes.readline)
            except SyntaxError as err:
                raise err

            removed_docstr = 0
            while True:
                try:
                    token_info = next(iterator)
                except (
                    tokenize.TokenError,
                    IndentationError,
                    SyntaxError,
                    UnicodeDecodeError,
                ) as err:
                    raise Exception(
                        f'Impossible to parse tokens because icorrect source code "{text[0:30]}"...'
                    ) from err
                except StopIteration as err:
                    raise Exception("End of iterator before ENDMARKER token.") from err
                # pylint: disable=no-else-continue
                if token_info.type in [tokenize.ENCODING, tokenize.NL]:
                    continue
                elif token_info.type == tokenize.NEWLINE:
                    if removed_docstr == 1:
                        removed_docstr = 0
                        continue
                    tokens.append(SpecialToken.NEWLINE.value)
                elif token_info.type == tokenize.COMMENT:
                    if keep_comments:
                        com = self.process_string(
                            token_info.string,
                            is_comment=True,
                        )
                        if len(com) > 0:
                            tokens.append(com)
                    else:
                        continue
                elif token_info.type == tokenize.STRING:
                    if token_info.string == token_info.line.strip():  # docstring
                        if not keep_comments:
                            removed_docstr = 1
                            continue
                        else:
                            coms = self.process_string(
                                token_info.string,
                                is_comment=True,
                            )
                            if len(coms) > 0:
                                tokens.append(coms)
                            else:
                                removed_docstr = 1
                    else:
                        tokens.append(
                            self.process_string(
                                token_info.string,
                                is_comment=False,
                            )
                        )
                elif token_info.type == tokenize.INDENT:
                    tokens.append(SpecialToken.INDENT.value)
                elif token_info.type == tokenize.DEDENT:
                    # empty block
                    if tokens[-1] == SpecialToken.INDENT.value:
                        tokens = tokens[:-1]
                    else:
                        tokens.append(SpecialToken.DEDENT.value)
                elif token_info.type == tokenize.ENDMARKER:
                    tokens.append(SpecialToken.END_MARKER.value)
                    break
                else:
                    tokens.append(token_info.string)
            assert tokens[-1] == SpecialToken.END_MARKER.value, "Error, no end marker"
        except KeyboardInterrupt as err:
            raise err
        except BaseException:  # pylint: disable=broad-except
            tokens = []
        return self.stringify(tokens[:-1], add_token_space)

    def detokenize(
        self,
        tokens: Union[List[str], str],
        add_token_space: bool = True,
    ) -> str:
        try:
            assert isinstance(tokens, (str, list))
            if isinstance(tokens, list):
                tokens = " ".join(tokens)

            # restore missing spaces around special tokens
            tokens = self.destringify(tokens, add_token_space)

            # prepare tokenized string for splitting
            tokens = (
                tokens.replace(
                    SpecialToken.END_COM.value,
                    SpecialToken.NEWLINE.value,
                )
                .replace(
                    SpecialToken.SPACE.value,
                    SpecialToken.SPACE_TOKEN.value,
                )
                .replace(
                    SpecialToken.STR_NEWLINE.value,
                    SpecialToken.NEWLINE.value,
                )
            )

            # prepare tokenized lines for detokenization
            lines = tokens.split(SpecialToken.NEWLINE.value)

            tabs = ""
            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith(f"{SpecialToken.INDENT.value} "):
                    tabs += "    "
                    line = line.replace(f"{SpecialToken.INDENT.value} ", tabs)
                elif line.startswith(SpecialToken.DEDENT.value):
                    number_dedent = line.count(SpecialToken.DEDENT.value)
                    tabs = tabs[4 * number_dedent :]
                    line = line.replace(SpecialToken.DEDENT.value, "")
                    line = line.strip()
                    line = tabs + line
                elif line == SpecialToken.DEDENT.value:
                    line = ""
                else:
                    line = tabs + line

                # python tokenizer ignores comment indentation
                # this way we check if comment should get more tabs
                try:
                    next_line = lines[idx + 1].strip()
                    if line.strip().startswith("#"):
                        if next_line.startswith(f"{SpecialToken.INDENT.value} "):
                            line = "    " + line
                except IndexError:
                    pass

                lines[idx] = line

            untokenized = "\n".join(lines)

            # detokenize string and comment
            untokenized = (
                untokenized.replace(SpecialToken.STR_NEWLINE.value, "\n")
                .replace(SpecialToken.TAB_SYMBOL.value, "\t")
                .replace(f" {SpecialToken.SPACE_TOKEN.value} ", " ")
                .replace(SpecialToken.SPACE_TOKEN.value, " ")
            )

            # detokenize imports
            untokenized = (
                untokenized.replace(". ", ".")
                .replace(" .", ".")
                .replace("import.", "import .")
                .replace("from.", "from .")
            )

            untokenized = untokenized.replace("> >", ">>").replace("< <", "<<")
            return untokenized
        except KeyboardInterrupt as err:
            raise err
        except BaseException:  # pylint: disable=broad-except
            return ""
