import difflib
import re
from typing import List

from .code_tokenizer import Tokenizer


class UnidiffBuilder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def build(
        self,
        source_content: str,
        target_content: str,
        source_filename: str,
        target_filename: str,
        context_size: int = 3,
        lineterm: str = "",
    ) -> str:
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
