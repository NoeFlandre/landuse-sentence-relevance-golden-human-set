from collections.abc import Iterable
from typing import Protocol, runtime_checkable


class SentenceSplitter(Protocol):
    def split(self, text: str) -> Iterable[str]: ...


@runtime_checkable
class BatchSentenceSplitter(Protocol):
    def split_many(self, texts: Iterable[str]) -> Iterable[Iterable[str]]: ...


def split_sentences_many(
    splitter: SentenceSplitter,
    texts: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    text_items = tuple(texts)
    unique_texts = tuple(dict.fromkeys(text_items))
    if isinstance(splitter, BatchSentenceSplitter):
        unique_groups = tuple(tuple(sentences) for sentences in splitter.split_many(unique_texts))
    else:
        unique_groups = tuple(tuple(splitter.split(text)) for text in unique_texts)
    groups_by_text = dict(zip(unique_texts, unique_groups, strict=True))
    return tuple(groups_by_text[text] for text in text_items)


class LanguageIdentifier(Protocol):
    def is_english(self, text: str) -> bool: ...


@runtime_checkable
class BatchLanguageIdentifier(Protocol):
    def is_english_many(self, texts: Iterable[str]) -> Iterable[bool]: ...
