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


class BatchRecordingSplitter(RecordingSplitter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[tuple[str, ...]] = []

    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        self.batch_calls.append(texts)
        return tuple(tuple(text.split("|")) for text in texts)


class CountingRows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.seen = 0

    def __iter__(self):
        for row in self._rows:
            self.seen += 1
            yield row


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

    assert [candidate.sentence for candidate in candidates] == ["Second sentence."]
    assert all(candidate.source is Source.WIKIPEDIA for candidate in candidates)
    assert all(candidate.source_url == "https://en.wikipedia.org/?curid=123" for candidate in candidates)
    assert "Wikipedia source: yielded 1 candidates from 2 sections" in caplog.text


def test_wikipedia_source_prefers_a_sentence_inside_the_section() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {
                "document_id": "d1",
                "section_id": "s1",
                "language": "en",
                "text": "Heading-like title|The place contains a broad wetland.",
            },
        ],
    }
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=10,
        max_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["The place contains a broad wetland."]


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
        max_candidates_per_cell=1,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["two"]
    assert splitter.calls == ["one|two"]


def test_wikipedia_source_bounds_section_text_before_splitting() -> None:
    text = "Title|Sentence"
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": text},
        ],
    }
    splitter = RecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_polygons_per_cell=10,
        max_text_characters=8,
    )

    list(source.iter_candidates())

    assert splitter.calls == [text[:8]]


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

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["two", "four"]
    assert splitter.calls == ["one|two", "three|four"]


def test_wikipedia_source_splits_a_shared_section_once() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 46.0, "lon": 3.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "Heading|Sentence"},
        ],
    }
    splitter = RecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_polygons_per_cell=10,
        max_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert len(candidates) == 2
    assert splitter.calls == ["Heading|Sentence"]


def test_wikipedia_source_batches_sections_when_splitter_supports_it() -> None:
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
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "One title|One sentence"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "Two title|Two sentence"},
        ],
    }
    splitter = BatchRecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_polygons_per_cell=10,
        max_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert [candidate.sentence for candidate in candidates] == ["One sentence", "Two sentence"]
    assert splitter.batch_calls == [("One title|One sentence", "Two title|Two sentence")]
    assert splitter.calls == []


def test_wikipedia_source_splits_at_most_one_section_per_candidate_cell() -> None:
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
            {
                "document_id": "d1",
                "section_id": "s1",
                "language": "en",
                "text": "Cell one title|Cell one sentence",
            },
            {
                "document_id": "d1",
                "section_id": "s2",
                "language": "en",
                "text": "Unused title|Unused sentence",
            },
            {
                "document_id": "d2",
                "section_id": "s3",
                "language": "en",
                "text": "Cell two title|Cell two sentence",
            },
        ],
    }
    splitter = BatchRecordingSplitter()
    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_polygons_per_cell=10,
        max_candidates_per_cell=1,
        candidate_cell_count=2,
        center_of_cell=lambda cell: (float(cell.removeprefix("cell-")), 0.0),
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == [
        "Cell one sentence",
        "Cell two sentence",
    ]
    assert splitter.batch_calls == [("Cell one title|Cell one sentence", "Cell two title|Cell two sentence")]


def test_wikipedia_source_merges_contiguous_polygon_and_link_shards() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": -20.0, "lon": 130.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "One title|One sentence"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "Two title|Two sentence"},
        ],
    }

    def row_shards_loader(config: str):
        return (rows[config][:1], rows[config][1:])

    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        row_shards_loader=row_shards_loader,
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_polygons_per_cell=2,
        max_candidates_per_cell=1,
    )

    candidates = list(source.iter_candidates())

    assert {candidate.sentence for candidate in candidates} == {"One sentence", "Two sentence"}
    assert {candidate.h3_cell for candidate in candidates} == {"cell-45.0", "cell--20.0"}


def test_wikipedia_source_bounds_rows_per_remote_polygon_shard() -> None:
    polygon_rows = CountingRows(
        [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ]
    )
    rows = {
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "Title|Sentence"},
        ],
    }

    def row_shards_loader(config: str):
        return (polygon_rows,) if config == "polygons" else (rows[config],)

    source = WikipediaCandidateSource(
        row_loader=lambda config: rows.get(config, ()),
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        row_shards_loader=row_shards_loader,
        max_polygon_rows_per_shard=1,
        max_polygons_per_cell=10,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Sentence"]
    assert polygon_rows.seen == 1


def test_wikipedia_source_scans_section_shards_in_stable_order() -> None:
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
            {"polygon_id": "p2", "has_english_wikipedia": True, "lat": -20.0, "lon": 130.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
            {"polygon_id": "p2", "document_id": "d2", "project": "wikipedia", "language": "en"},
        ],
        "wikipedia_sections": [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "One title|One sentence"},
            {"document_id": "d2", "section_id": "s2", "language": "en", "text": "Two title|Two sentence"},
        ],
    }
    splitter = RecordingSplitter()

    def row_shards_loader(config: str):
        return (rows[config][:1], rows[config][1:])

    source = WikipediaCandidateSource(
        row_loader=lambda config: rows[config],
        row_shards_loader=row_shards_loader,
        splitter=splitter,
        cell_for_location=lambda latitude, longitude: f"cell-{latitude}",
        max_polygons_per_cell=2,
        max_candidates_per_cell=1,
        max_section_rows_per_shard=1,
        max_stream_workers=2,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["One sentence", "Two sentence"]
    assert splitter.calls == ["One title|One sentence", "Two title|Two sentence"]


def test_wikipedia_source_bounds_rows_per_remote_section_shard() -> None:
    section_rows = CountingRows(
        [
            {"document_id": "d1", "section_id": "s1", "language": "en", "text": "Title|Sentence"},
            {"document_id": "d1", "section_id": "s2", "language": "en", "text": "Unused"},
        ]
    )
    rows = {
        "polygons": [
            {"polygon_id": "p1", "has_english_wikipedia": True, "lat": 45.0, "lon": 2.0},
        ],
        "polygon_document_links": [
            {"polygon_id": "p1", "document_id": "d1", "project": "wikipedia", "language": "en"},
        ],
    }

    def row_shards_loader(config: str):
        if config == "wikipedia_sections":
            return (section_rows,)
        return (rows[config],)

    source = WikipediaCandidateSource(
        row_loader=lambda config: rows.get(config, ()),
        row_shards_loader=row_shards_loader,
        splitter=FakeSplitter(),
        cell_for_location=lambda latitude, longitude: "cell-1",
        max_section_rows_per_shard=1,
        max_polygons_per_cell=10,
    )

    assert [candidate.sentence for candidate in source.iter_candidates()] == ["Sentence"]
    assert section_rows.seen == 1


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


def test_wikipedia_source_excludes_cells_already_used_by_a_previous_pass() -> None:
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
        excluded_cells={"cell-a"},
        center_of_cell=lambda cell: {"cell-a": (45.0, 2.0), "cell-b": (46.0, 3.0), "cell-c": (47.0, 4.0)}[
            cell
        ],
    )

    candidates = list(source.iter_candidates())

    assert {candidate.h3_cell for candidate in candidates} == {"cell-b", "cell-c"}


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

    with pytest.raises(ValueError, match="max_section_rows_per_shard must be positive"):
        WikipediaCandidateSource(
            row_loader=lambda config: [],
            splitter=FakeSplitter(),
            cell_for_location=lambda latitude, longitude: "cell-1",
            max_section_rows_per_shard=0,
        )
