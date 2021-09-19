import difflib
import re
from abc import abstractmethod
from typing import List, Optional

from .tokenizer import Tokenizer


class DiffGen:
    @abstractmethod
    def generate(
        self,
        source_content: str,
        target_content: str,
        source_filename: Optional[str] = None,
        target_filename: Optional[str] = None,
        context_size: int = 3,
        lineterm: str = "",
    ) -> str:
        raise NotImplementedError()


class GitDiff(DiffGen):
    def generate(
        self,
        source_content: str,
        target_content: str,
        source_filename: Optional[str] = None,
        target_filename: Optional[str] = None,
        context_size: int = 3,
        lineterm: str = "",
    ) -> str:
        if source_filename is None and target_filename is not None:
            source_filename = target_filename
        elif target_filename is None and source_filename is not None:
            target_filename = source_filename
        else:
            raise ValueError(
                "Either `source_filename` or `target_filename` must be provided."
            )

        source_lines = source_content.splitlines()
        target_lines = target_content.splitlines()

        lines = difflib.unified_diff(
            source_lines,
            target_lines,
            source_filename,
            target_filename,
            n=context_size,
            lineterm=lineterm,
        )

        lines = [line.rstrip() for line in lines]

        return "\n".join(lines)


class CodeDiff(DiffGen):
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def generate(
        self,
        source_content: str,
        target_content: str,
        source_filename: Optional[str] = None,
        target_filename: Optional[str] = None,
        context_size: int = 3,
        lineterm: str = "",
    ) -> str:
        if source_filename is None and target_filename is not None:
            source_filename = target_filename
        elif target_filename is None and source_filename is not None:
            target_filename = source_filename
        else:
            raise ValueError(
                "Either `source_filename` or `target_filename` must be provided."
            )

        source_lines = self.tokenize(source_content)
        target_lines = self.tokenize(target_content)

        lines = difflib.unified_diff(
            source_lines,
            target_lines,
            source_filename,
            target_filename,
            n=context_size,
            lineterm=lineterm,
        )

        lines = [self.prepare_line(line) for line in lines]

        return " ".join(lines)

    def prepare_line(self, line: str) -> str:
        # remove common metadata string
        line = re.sub(r"@@ .* @@", "<newline>", line)
        # add a space to the succeeding token after `-` token
        line = re.sub(r"^\-([^-]+)", r"- \g<1>", line)
        # add a space to the succeeding token after `+` token
        line = re.sub(r"^\+([^+]+)", r"+ \g<1>", line)
        return line

    def tokenize(self, content: str) -> List[str]:
        content = self.tokenizer.tokenize(
            content,
            keep_comments=False,
            add_token_space=True,
        )
        content = content.replace("<newline>", "<newline>\n")
        content_lines = [line.strip() for line in content.split("\n")]
        return content_lines
