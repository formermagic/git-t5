from html.parser import HTMLParser
from io import StringIO


class HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def reset(self) -> None:
        super().reset()
        self.text = StringIO()

    def strip_tags(self, text: str) -> str:
        self.reset()
        self.feed(text)
        return self.get_data()

    def handle_data(self, data: str) -> None:
        self.text.write(data)

    def error(self, message: str) -> None:
        raise RuntimeError(message)

    def get_data(self) -> str:
        return self.text.getvalue()
