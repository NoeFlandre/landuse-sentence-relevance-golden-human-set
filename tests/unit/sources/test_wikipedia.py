import logging
from collections.abc import Iterable

import pytest

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource, _polygon_metadata


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


class RecordingSplitter(FakeSplitter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def split(self, text: str) -> Iterable[str]:
        self.calls.append(text)
        return super().split(text)


def test_polygon_discovery_keeps_only_candidate_metadata() -> None:
    row = {
        "polygon_id": "p1",
        "has_english_wikipedia": True,
        "lat": 45.0,
        "lon": 2.0,
        "name": "Place",
        "region": "Region",
        "geometry": {"coordinates": [[[[2.0, 45.0]]]]},
    }

    assert _polygon_metadata(row) == {
        "polygon_id": "p1",
        "has_english_wikipedia": True,
        "lat": 45.0,
        "lon": 2.0,
        "name": "Place",
        "region": "Region",
    }


def test_wikipedia_source_joins_only_english_wikipedia_rows(caplog) -> None:
    rows = {
        "polygons": [
            {
                "polygon_id": "p1",
                "has_english_wikipedia": True,
                "lat": 45.0,
                "lon": 2.0,
                "name": "Place",
                "region": "Region",
            },
            {
                "polygon_id": "p2",
                "has_english_wikipedia": True,
                "lat": 46.0,
                "lon": 3.0,
                "name": "Other",
                "region": "Region",
            },
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p1", "document_id": "d2", "project": "wikivoyage", "language": "en"},
            {"polygon_id": "p2", "document_id": "d3", "project": "wikipedia", "language": "fr"},
        ],
        "wikipedia_sections": [
            {
                "document_id": "d1",
                "section_id": "s1",
                "language": "en",
                "text": "First sentence.|Second sentence.",
                "page_id": 123,
            },
            {
                "document_id": "d2",
                "section_id": "s2",
                "language": "en",
                "text": "Should not be used.",
                "page_id": 456,
            },
        ],
    }

    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=10,
    )

    caplog.set_level(logging.INFO)
    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["First sentence.", "Second sentence."]
    assert all(candidate.source is Source.WIKIPEDIA for candidate in candidates)
    assert all(candidate.source_url == "https://en.wikipedia.org/?curid=123" for candidate in candidates)
    assert "Wikipedia source: yielded 2 candidates from 2 sections" in caplog.text


def test_wikipedia_source_does_not_emit_empty_sentences() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": " | useful "},
        ],
    }
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=10,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["useful"]


def test_wikipedia_source_bounds_polygons_per_cell() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "one"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "two"},
        ],
    }
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=1,
    )

    assert len(list(source.iter_candidates())) == 1


def test_wikipedia_source_does_not_split_sections_after_cell_capacity() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "one|two"},
            {"document_id": "d1", "section_id": "s2", "language": "en", "text": "three"},
        ],
    }
    splitter = RecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=10,
        max_candidates_per_cell=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["one", "two"]
    assert splitter.calls == ["one|two"]


def test_wikipedia_source_stops_after_enough_full_candidate_cells() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 46.0, "lon": 3.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "one|two"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "three|four"},
        ],
    }
    splitter = RecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: "cell-a" if latitude == 45.0 else "cell-b",
        max_polygons_per_cell=10,
        max_candidates_per_cell=2,
        minimum_candidate_cells=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["one", "two"]
    assert splitter.calls == ["one|two"]


def test_wikipedia_source_preselects_a_deterministic_cell_budget() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 46.0, "lon": 3.0},
            {"polygon_id": "p3", "has_english_wikipedia": True, "lat": 47.0, "lon": 4.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p3", "document_id": "d3", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "one"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "two"},
            {"document_id": "d3", "section_id": "s3", "language": "en", "text": "three"},
        ],
    }
    cells = {45.0: "cell-a", 46.0: "cell-b", 47.0: "cell-c"}
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: cells[latitude],
        max_polygons_per_cell=10,
        candidate_cell_count=2,
        center_of_cell=lambda cell: {"cell-a": (45.0, 2.0), "cell-b": (46.0, 3.0), "cell-c": (47.0, 4.0)}[
            cell
        ],
    )

    candidates = list(source.iter_candidates())

    assert len(source.candidate_cells) == 2
    assert {candidate.h3_cell for candidate in candidates} == set(source.candidate_cells)


def test_wikipedia_source_exposes_only_cells_with_candidates() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 46.0, "lon": 3.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "useful"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": ""},
        ],
    }
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-a" if latitude == 45.0 else "cell-b",
        max_polygons_per_cell=10,
        candidate_cell_count=2,
        center_of_cell=lambda cell: {"cell-a": (45.0, 2.0), "cell-b": (46.0, 3.0)}[cell],
    )

    list(source.iter_candidates())

    assert source.candidate_cells == {"cell-a"}


def test_wikipedia_source_rejects_an_invalid_candidate_cell_budget() -> None:
    with pytest.raises(ValueError, match="candidate_cell_count must be positive"):
        WikipediaCandidateSource(
            row_loader=lambda config: [],
            splitter=FakeSplitter(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            candidate_cell_count=0,
        )


def test_wikipedia_source_rejects_an_invalid_minimum_candidate_cell_count() -> None:
    with pytest.raises(ValueError, match="minimum_candidate_cells must be positive"):
        WikipediaCandidateSource(
            row_loader=lambda config: [],
            splitter=FakeSplitter(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            minimum_candidate_cells=0,
        )
