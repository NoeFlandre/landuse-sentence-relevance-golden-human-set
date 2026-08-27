import logging
from collections.abc import Iterable

import pytest

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.website import WebsiteCandidateSource


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


class RecordingSplitter(FakeSplitter):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def split(self, text: str) -> Iterable[str]:
        self.texts.append(text)
        return super().split(text)


class FakeLanguageIdentifier:
    def is_english(self, text: str) -> bool:
        return not text.startswith("NONEN")


def test_website_source_declares_text_and_url_fields_as_pairs() -> None:
    assert WebsiteCandidateSource._FIELD_SPECS == (
        ("website_text", "website"),
        ("contact_website_text", "contact_website"),
    )


def test_website_source_uses_both_website_text_fields(caplog) -> None:
    rows = [
        {
            "polygon_id": "p1",
            "has_any_website": True,
            "lat": 45.0,
            "lon": 2.0,
            "name": "Place",
            "region": "Region",
            "website": "https://example.test",
            "contact_website": "https://contact.example.test",
            "website_text": "Website sentence.",
            "contact_website_text": "Contact sentence.",
        },
    ]

    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
    )

    caplog.set_level(logging.INFO)
    candidates = list(source.iter_candidates())

    assert [candidate.source_field for candidate in candidates] == ["website_text", "contact_website_text"]
    assert [candidate.source_url for candidate in candidates] == [
        "https://example.test",
        "https://contact.example.test",
    ]
    assert all(candidate.source is Source.WEBSITE for candidate in candidates)
    assert "Website source: streamed 1 rows; yielded 2 English candidates" in caplog.text


def test_website_source_rejects_non_english_sentences() -> None:
    rows = [
        {
            "polygon_id": "p1",
            "has_any_website": True,
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "NONEN sentence.|English sentence.",
            "contact_website_text": None,
        },
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1",
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["English sentence."]


def test_website_source_can_reuse_the_wikipedia_candidate_cells() -> None:
    rows = [
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "Keep this."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Skip this."},
    ]
    source = WebsiteCandidateSource(
        row_loader=lambda: rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        allowed_cells={"cell-1"},
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Keep this."]


def test_website_source_discovers_cells_before_processing_selected_cells() -> None:
    rows = [
        {"website_text": "Missing location."},
        {"polygon_id": "p0", "lat": 44.0, "lon": 1.0},
        {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."},
        {"polygon_id": "p2", "lat": 46.0, "lon": 3.0, "website_text": "Second."},
        {"polygon_id": "p3", "lat": 47.0, "lon": 4.0, "website_text": "Third."},
    ]
    calls = 0

    def row_loader() -> Iterable[dict[str, object]]:
        nonlocal calls
        calls += 1
        yield from rows

    cells = {44.0: "cell-a", 45.0: "cell-a", 46.0: "cell-b", 47.0: "cell-c"}
    source = WebsiteCandidateSource(
        row_loader=row_loader,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: cells[latitude],
        candidate_cell_count=2,
        center_of_cell=lambda cell: {"cell-a": (0.0, 0.0), "cell-b": (0.0, 30.0), "cell-c": (0.0, 60.0)}[
            cell
        ],
        seed="test",
        max_candidates_per_cell=1,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert calls == 2
    assert len(candidates) == 2
    assert len({candidate.h3_cell for candidate in candidates}) == 2


def test_website_source_stops_after_enough_full_candidate_cells() -> None:
    def rows() -> Iterable[dict[str, object]]:
        yield {
            "polygon_id": "p1",
            "lat": 45.0,
            "lon": 2.0,
            "website_text": "Cell one.|Cell one extra.",
        }
        yield {
            "polygon_id": "p2",
            "lat": 46.0,
            "lon": 3.0,
            "website_text": "Cell two.|Cell two extra.",
        }
        raise AssertionError("the website stream should stop after the cell budget is filled")

    source = WebsiteCandidateSource(
        row_loader=rows,
        splitter=FakeSplitter(),
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        max_candidates_per_cell=8,
        minimum_candidate_cells=2,
        minimum_candidates_per_cell=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "Cell one.",
        "Cell one extra.",
        "Cell two.",
        "Cell two extra.",
    ]


def test_website_source_bounds_splitter_work_per_cell() -> None:
    def rows() -> Iterable[dict[str, object]]:
        yield {"polygon_id": "p1", "lat": 45.0, "lon": 2.0, "website_text": "First."}
        yield {"polygon_id": "p2", "lat": 45.0, "lon": 2.0, "website_text": "Second."}
        yield {"polygon_id": "p3", "lat": 45.0, "lon": 2.0, "website_text": "Skip this."}
        yield {"polygon_id": "p4", "lat": 46.0, "lon": 3.0, "website_text": "Third."}
        yield {"polygon_id": "p5", "lat": 46.0, "lon": 3.0, "website_text": "Fourth."}
        raise AssertionError("the bounded website stream should stop before the next row")

    splitter = RecordingSplitter()
    source = WebsiteCandidateSource(
        row_loader=rows,
        splitter=splitter,
        language_identifier=FakeLanguageIdentifier(),
        cell_for_location=lambda latitude, longitude: "cell-1" if latitude == 45.0 else "cell-2",
        minimum_candidate_cells=2,
        max_rows_per_cell=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "First.",
        "Second.",
        "Third.",
        "Fourth.",
    ]
    assert splitter.texts == ["First.", "Second.", "Third.", "Fourth."]


def test_website_source_rejects_invalid_candidate_budget() -> None:
    with pytest.raises(ValueError, match="max_candidates_per_cell must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_candidates_per_cell=0,
        )

    with pytest.raises(ValueError, match="minimum_candidate_cells must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            minimum_candidate_cells=0,
        )

    with pytest.raises(ValueError, match="center_of_cell is required"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            candidate_cell_count=1,
        )

    with pytest.raises(ValueError, match="max_rows_per_cell must be positive"):
        WebsiteCandidateSource(
            row_loader=lambda: [],
            splitter=FakeSplitter(),
            language_identifier=FakeLanguageIdentifier(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_rows_per_cell=0,
        )
