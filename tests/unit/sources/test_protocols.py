from collections.abc import Iterable

from landuse_sentence_relevance.sources.protocols import split_sentences_many


class RecordingBatchSplitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def split(self, text: str) -> tuple[str, ...]:
        return (f"{text}.",)

    def split_many(self, texts: Iterable[str]) -> tuple[tuple[str, ...], ...]:
        text_items = tuple(texts)
        self.calls.append(text_items)
        return tuple((f"{text}.",) for text in text_items)


def test_split_sentences_many_deduplicates_texts_without_changing_order() -> None:
    splitter = RecordingBatchSplitter()

    result = split_sentences_many(splitter, ("same", "other", "same"))

    assert result == (("same.",), ("other.",), ("same.",))
    assert splitter.calls == [("same", "other")]
