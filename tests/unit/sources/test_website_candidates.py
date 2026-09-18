from __future__ import annotations

from collections.abc import Iterable

import pytest

import landuse_sentence_relevance.sources.website_candidates as website_candidates
from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.sentence_selection import SentencePart
from landuse_sentence_relevance.sources.website_candidates import WebsiteCandidateBuilder


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


class FakeLanguageIdentifier:
    def is_english(self, text: str) -> bool:
        return not text.startswith("NONEN")


class BatchSplitter(FakeSplitter):
    def __init__(self) -> None:
        self.scalar_calls: list[str] = []
        self.batch_calls: list[tuple[str, ...]] = []

    def split(self, text: str) -> Iterable[str]:
        self.scalar_calls.append(text)
        return super().split(text)

    def split_many(self, texts: Iterable[str]) -> Iterable[Iterable[str]]:
        text_items = tuple(texts)
        self.batch_calls.append(text_items)
        return tuple(tuple(text.split("|")) for text in text_items)


class BatchLanguageIdentifier(FakeLanguageIdentifier):
    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, ...]] = []

    def is_english_many(self, texts: Iterable[str]) -> Iterable[bool]:
        text_items = tuple(texts)
        self.batch_calls.append(text_items)
        return tuple(self.is_english(text) for text in text_items)


def test_builder_returns_the_first_english_sentence_from_a_website_field() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=CellQuota(1),
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = tuple(
        builder.candidates_for_row(
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "website_text": "NONEN heading.|The place contains a wetland.",
            },
            "cell-1",
        )
    )

    assert [candidate.sentence for candidate in candidates] == [
        "The place contains a wetland.",
    ]


def test_builder_preserves_metadata_for_each_website_field() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = tuple(
        builder.candidates_for_row(
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "name": "Place",
                "region": "Region",
                "website": "https://example.test",
                "contact_website": "https://contact.example.test",
                "website_text": "Title.|The place contains a wetland.",
                "contact_website_text": "Contact title.|A forest surrounds the place.",
            },
            "cell-1",
        )
    )

    assert [
        (candidate.source_field, candidate.sentence, candidate.source_url) for candidate in candidates
    ] == [
        ("website_text", "The place contains a wetland.", "https://example.test"),
        ("contact_website_text", "A forest surrounds the place.", "https://contact.example.test"),
    ]
    assert [(candidate.place_name, candidate.region) for candidate in candidates] == [
        ("Place", "Region"),
        ("Place", "Region"),
    ]
    assert [candidate.candidate_id for candidate in candidates] == [
        "website:p1:website_text:1",
        "website:p1:contact_website_text:1",
    ]


def test_builder_seed_deterministically_orders_contextual_sentences() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_text": "Title.|First.|Second.|Third.",
    }

    first = tuple(
        WebsiteCandidateBuilder(
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            seed="seed-a",
            candidate_quota=CellQuota(3),
            max_text_characters=None,
            require_paragraph=False,
        ).candidates_for_row(row, "cell-1")
    )
    second = tuple(
        WebsiteCandidateBuilder(
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            seed="seed-b",
            candidate_quota=CellQuota(3),
            max_text_characters=None,
            require_paragraph=False,
        ).candidates_for_row(row, "cell-1")
    )

    assert [candidate.sentence for candidate in first] == ["Second.", "First.", "Third."]
    assert [candidate.sentence for candidate in second] == ["Third.", "First.", "Second."]


def test_builder_rejects_title_only_text_after_the_character_limit() -> None:
    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website_text": "Title. Detailed sentence.",
    }
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=6,
        require_paragraph=True,
    )

    assert builder.has_eligible_text(row) is False
    assert tuple(builder.candidates_for_row(row, "cell-1")) == ()


def test_builder_keeps_a_later_field_when_an_earlier_field_is_missing() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = tuple(
        builder.candidates_for_row(
            {
                "polygon_id": "p1",
                "lat": 45.0,
                "lon": 2.0,
                "contact_website_text": "Contact title.|The place has a forest.",
            },
            "cell-1",
        )
    )

    assert [candidate.source_field for candidate in candidates] == ["contact_website_text"]


def test_builder_batches_splitting_and_scalar_language_selection() -> None:
    splitter = BatchSplitter()
    builder = WebsiteCandidateBuilder(
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        seed="seed-a",
        candidate_quota=CellQuota(1),
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "website": "https://example.test",
                    "website_text": "Title.|First.|Second.|Third.",
                },
                "cell-1",
            ),
        )
    )

    assert [[candidate.sentence for candidate in row] for row in candidates] == [["Second."]]
    assert splitter.batch_calls == [("Title.|First.|Second.|Third.",)]
    assert splitter.scalar_calls == []
    assert candidates[0][0].candidate_id == "website:p1:website_text:2"
    assert candidates[0][0].source_url == "https://example.test"


def test_builder_reuses_batch_sentence_groups_for_all_field_candidates() -> None:
    class BatchOnlySplitter(BatchSplitter):
        def split(self, text: str) -> Iterable[str]:
            raise AssertionError("all-field batch candidates must reuse split_many")

    splitter = BatchOnlySplitter()
    builder = WebsiteCandidateBuilder(
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "website_text": "Title.|The place contains a wetland.",
                },
                "cell-1",
            ),
        )
    )

    assert [candidate.sentence for candidate in candidates[0]] == ["The place contains a wetland."]
    assert splitter.batch_calls == [("Title.|The place contains a wetland.",)]


def test_builder_keeps_later_website_field_when_the_first_is_missing_in_batch() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=BatchSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "contact_website_text": "Contact title.|The place has a forest.",
                },
                "cell-1",
            ),
        )
    )

    assert [candidate.source_field for candidate in candidates[0]] == ["contact_website_text"]


def test_builder_keeps_later_contextual_field_when_the_first_is_title_only() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=BatchSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=True,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "website_text": "Title.",
                    "contact_website_text": "First. Second.",
                },
                "cell-1",
            ),
        )
    )

    assert [candidate.source_field for candidate in candidates[0]] == ["contact_website_text"]


def test_builder_applies_character_limit_before_batch_context_check() -> None:
    splitter = BatchSplitter()
    builder = WebsiteCandidateBuilder(
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=6,
        require_paragraph=True,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "website_text": "Title. Detailed sentence.",
                },
                "cell-1",
            ),
        )
    )

    assert candidates == ((),)
    assert splitter.batch_calls == [()]


def test_builder_batches_language_selection_and_preserves_sentence_index() -> None:
    splitter = BatchSplitter()
    language_identifier = BatchLanguageIdentifier()
    builder = WebsiteCandidateBuilder(
        splitter=splitter,
        language_identifier=language_identifier,
        seed="seed-a",
        candidate_quota=CellQuota(1),
        max_text_characters=None,
        require_paragraph=False,
    )

    candidates = builder.candidates_for_rows(
        (
            (
                {
                    "polygon_id": "p1",
                    "lat": 45.0,
                    "lon": 2.0,
                    "website": "https://example.test",
                    "website_text": "Title.|First.|Second.|Third.",
                },
                "cell-1",
            ),
        )
    )

    assert [candidate.sentence for candidate in candidates[0]] == ["Second."]
    assert candidates[0][0].candidate_id == "website:p1:website_text:2"
    assert candidates[0][0].source_url == "https://example.test"
    assert language_identifier.batch_calls


def test_builder_preserves_prepared_batch_work_and_url_metadata() -> None:
    class SplitterThatMustNotBeCalled(FakeSplitter):
        def split(self, text: str) -> Iterable[str]:
            raise AssertionError("prepared batch sentences must be reused")

    row = {
        "polygon_id": "p1",
        "lat": 45.0,
        "lon": 2.0,
        "website": "https://example.test",
        "website_text": "Original.",
    }
    builder = WebsiteCandidateBuilder(
        splitter=SplitterThatMustNotBeCalled(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=CellQuota(2),
        max_text_characters=None,
        require_paragraph=False,
    )
    work = builder._field_work_for_rows(((row, "cell-1"),))
    row["website_text"] = None

    candidates = tuple(builder._field_candidates_for_work(work[0], ("Prepared.",)))

    assert [candidate.sentence for candidate in candidates] == ["Prepared."]
    assert candidates[0].source_url == "https://example.test"


def test_builder_preserves_pending_row_indexes_in_batch_work() -> None:
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=CellQuota(1),
        max_text_characters=None,
        require_paragraph=False,
    )
    rows = (
        ({"polygon_id": "p0", "lat": 45.0, "lon": 2.0}, "cell-0"),
        ({"polygon_id": "p1", "lat": 46.0, "lon": 3.0, "website_text": "Sentence."}, "cell-1"),
    )

    work = builder._field_batch_work(rows, (1,), ("website_text", "website"))

    assert [field.row_index for field in work] == [1]


def test_builder_rejects_mismatched_batch_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "Sentence."}
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=CellQuota(1),
        max_text_characters=None,
        require_paragraph=False,
    )
    rows = ((row, "cell-1"),)
    work = builder._field_work_for_rows(rows)

    with pytest.raises(ValueError):
        builder._field_work_for_rows(rows, row_indexes=())
    with pytest.raises(ValueError):
        builder._append_scalar_field_candidates([[]], work, ())
    with pytest.raises(ValueError):
        builder._group_row_candidates(rows, work, ())

    monkeypatch.setattr(website_candidates, "select_batch_parts", lambda parts, identifier: ())
    with pytest.raises(ValueError):
        builder._append_batch_language_candidates(
            [[]],
            work,
            (("Sentence.",),),
            BatchLanguageIdentifier(),
        )


def test_builder_skips_empty_selected_parts_and_keeps_later_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = WebsiteCandidateBuilder(
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        seed="test",
        candidate_quota=None,
        max_text_characters=None,
        require_paragraph=False,
    )
    monkeypatch.setattr(
        website_candidates,
        "prioritize_sentences",
        lambda sentences, seed, accept: (SentencePart(0, ""), SentencePart(1, "Valid.")),
    )

    candidates = tuple(
        builder._field_candidate_parts(
            ("ignored",),
            "test",
            "p1",
            45.0,
            2.0,
            "cell-1",
            "website_text",
            None,
            {"name": "Place", "region": "Region"},
        )
    )

    assert [candidate.sentence for candidate in candidates] == ["Valid."]
