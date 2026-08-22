from collections.abc import Iterable
from typing import Protocol


class SentenceSplitter(Protocol):
    def split(self, text: str) -> Iterable[str]: ...


class LanguageIdentifier(Protocol):
    def is_english(self, text: str) -> bool: ...
