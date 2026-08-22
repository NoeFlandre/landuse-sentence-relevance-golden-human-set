from collections.abc import Iterable

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.sources.wikipedia import WikipediaCandidateSource


class FakeSplitter:
    def split(self, text: str) -> Iterable[str]:
        return text.split("|")


def test_wikipedia_source_joins_only_english_wikipedia_rows() -> None:
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

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["First sentence.", "Second sentence."]
    assert all(candidate.source is Source.WIKIPEDIA for candidate in candidates)
    assert all(candidate.source_url == "https://en.wikipedia.org/?curid=123" for candidate in candidates)


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
