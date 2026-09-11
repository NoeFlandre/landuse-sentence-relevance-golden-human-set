from __future__ import annotations

import logging

import pytest

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sentence_selection import SentencePart
from landuse_sentence_relevance.sources import website_discovery
from landuse_sentence_relevance.sources.website_discovery import (
    _boundary_count,
    _compact_row,
    _counts_by_cell,
    _discovery_row,
    _discovery_row_rounds,
    _DiscoveryScan,
    _interleave_discovery_rows,
    _merge_candidate_scan_results,
    _scan_candidate_rows,
    _scan_candidate_shards,
    _select_discovery_rows,
)
from landuse_sentence_relevance.sources.website_text import (
    _apply_language_selection_round,
    _candidate_part_groups,
    _FieldWork,
    _first_accepted_part,
    _language_selection_groups,
    _location_from_row,
)


def _candidate(candidate_id: str, cell: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sentence=candidate_id,
        source=Source.WEBSITE,
        source_record_id=candidate_id,
        source_field="website_text",
        h3_cell=cell,
        h3_resolution=3,
        latitude=1.0,
        longitude=2.0,
    )


def _row(polygon_id: str, text: str) -> dict[str, object]:
    return {
        "polygon_id": polygon_id,
        "osm_id": f"osm-{polygon_id}",
        "lat": 1.0,
        "lon": 2.0,
        "name": f"Name {polygon_id}",
        "region": "Region",
        "website": f"https://{polygon_id}.example",
        "contact_website": f"https://contact.{polygon_id}.example",
        "website_text": text,
        "contact_website_text": "Contact paragraph.",
    }


def test_location_uses_osm_fallback_and_requires_all_coordinates() -> None:
    assert _location_from_row({"osm_id": 42, "lat": "1.5", "lon": "-2.5"}) == ("42", 1.5, -2.5)
    assert _location_from_row({"polygon_id": "p1", "lat": None, "lon": 2.5}) is None
    assert _location_from_row({"polygon_id": "p1", "lat": 1.5}) is None
    assert _location_from_row({"polygon_id": 42, "lat": 1.5, "lon": -2.5}) == ("42", 1.5, -2.5)


def test_candidate_part_groups_use_field_specific_deterministic_seeds() -> None:
    work = (
        _FieldWork(0, {}, "a", 1.0, 2.0, "cell-1", "website_text", "website", None, "text"),
        _FieldWork(1, {}, "b", 1.0, 2.0, "cell-1", "website_text", "website", None, "text"),
    )
    sentences = (("Title.", "A.", "B.", "C.", "D.", "E."),) * 2

    groups = _candidate_part_groups(work, sentences, "seed")

    assert groups[0] != groups[1]
    with pytest.raises(ValueError):
        _candidate_part_groups(work, sentences[:1], "seed")


def test_language_selection_groups_skip_empty_pending_groups() -> None:
    parts = (SentencePart(0, "sentence"),)

    assert _language_selection_groups([(), parts], [None, None]) == ((1, parts),)


def test_language_selection_round_preserves_offset_across_groups() -> None:
    first = (SentencePart(0, "first"),)
    second = (SentencePart(0, "second"),)
    pending: list[tuple[SentencePart, ...]] = [first, second]
    selected: list[SentencePart | None] = [None, None]

    _apply_language_selection_round(
        pending,
        selected,
        ((0, first), (1, second)),
        (True, False),
    )

    assert selected == [first[0], None]
    assert pending == [(), ()]


def test_first_accepted_part_requires_a_prediction_for_each_part() -> None:
    with pytest.raises(ValueError):
        _first_accepted_part((SentencePart(0, "first"),), ())


def test_counts_by_cell_counts_duplicates_without_crossing_cells() -> None:
    assert _counts_by_cell((_candidate("one", "a"), _candidate("two", "a"), _candidate("three", "b"))) == {
        "a": 2,
        "b": 1,
    }


def test_compact_row_keeps_metadata_and_bounds_text_fields() -> None:
    compact = _compact_row(_row("p1", "  A long paragraph.  "), max_text_characters=7)

    assert compact == {
        "polygon_id": "p1",
        "osm_id": "osm-p1",
        "lat": 1.0,
        "lon": 2.0,
        "name": "Name p1",
        "region": "Region",
        "website": "https://p1.example",
        "contact_website": "https://contact.p1.example",
        "website_text": "A long ",
        "contact_website_text": "Contact",
    }


def test_discovery_row_enforces_a_per_cell_limit_and_tracks_each_cell() -> None:
    counts: dict[str, int] = {}

    assert _discovery_row(_row("a1", "A."), "cell-a", counts, 1, None) is not None
    assert _discovery_row(_row("a2", "A second."), "cell-a", counts, 1, None) is None
    assert _discovery_row(_row("b1", "B."), "cell-b", counts, 1, None) is not None
    assert counts == {"cell-a": 1, "cell-b": 1}
    unbounded: dict[str, int] = {}
    _discovery_row(_row("c1", "C."), "cell-c", unbounded, None, None)
    _discovery_row(_row("c2", "C second."), "cell-c", unbounded, None, None)
    assert unbounded == {"cell-c": 2}


def test_scan_candidate_rows_reports_limits_cells_and_compact_rows(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    rows = (
        {**_row("p1", "First paragraph."), "cell": "cell-a"},
        {**_row("p1b", "Fallback paragraph."), "cell": "cell-a"},
        {**_row("p2", "Second paragraph."), "cell": "cell-b"},
    )

    result = _scan_candidate_rows(
        rows,
        lambda row: str(row["cell"]),
        rows_per_cell=1,
        max_rows=3,
        max_text_characters=6,
        progress_label="test scan",
    )

    assert "test scan: scanned 1 row" in caplog.text
    assert result.rows_seen == 3
    assert result.complete is False
    assert result.cells == frozenset({"cell-a", "cell-b"})
    assert len(result.rows) == 2
    assert result.rows[0][1]["website_text"] == "First "


def test_scan_candidate_shards_handles_empty_and_parallel_streams() -> None:
    empty = _scan_candidate_shards((), lambda row: "unused", 1, None, 4, 2)
    assert empty == _DiscoveryScan(frozenset(), 0, (), True)

    result = _scan_candidate_shards(
        ((_row("a", "Alpha paragraph."),), (_row("b", "Beta paragraph."),)),
        lambda row: str(row["polygon_id"]),
        rows_per_cell=1,
        max_rows=None,
        max_text_characters=5,
        max_workers=2,
    )

    assert result.rows_seen == 2
    assert result.complete is True
    assert result.cells == frozenset({"a", "b"})
    assert [row["website_text"] for _, row in result.rows] == ["Alpha", "Beta "]


def test_single_discovery_shard_preserves_limits_and_progress_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    rows = (
        {**_row("single-1", "Alpha paragraph."), "cell": "cell-a"},
        {**_row("single-2", "Fallback paragraph."), "cell": "cell-a"},
    )

    result = _scan_candidate_shards(
        (rows,),
        lambda row: str(row["cell"]),
        rows_per_cell=1,
        max_rows=None,
        max_text_characters=5,
        max_workers=2,
    )

    assert "Website cell discovery shard 1: scanned 1 row" in caplog.text
    assert result.rows_seen == 2
    assert result.complete is True
    assert len(result.rows) == 1
    assert result.rows[0][1]["website_text"] == "Alpha"


def test_multiple_discovery_shards_honor_worker_and_cell_limits(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)
    observed: dict[str, object] = {}

    class RecordingExecutor:
        def __init__(self, *, max_workers: int | None) -> None:
            observed["max_workers"] = max_workers

        def __enter__(self) -> RecordingExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def map(self, function, *iterables):
            return tuple(function(*arguments) for arguments in zip(*iterables, strict=False))

    monkeypatch.setattr(website_discovery, "ThreadPoolExecutor", RecordingExecutor)
    streams = (
        (
            {**_row("a1", "Alpha paragraph."), "cell": "cell-a"},
            {**_row("a2", "Alpha second."), "cell": "cell-a"},
        ),
        (
            {**_row("b1", "Beta paragraph."), "cell": "cell-b"},
            {**_row("b2", "Beta second."), "cell": "cell-b"},
        ),
    )

    result = _scan_candidate_shards(
        streams,
        lambda row: str(row["cell"]),
        rows_per_cell=1,
        max_rows=2,
        max_text_characters=5,
        max_workers=3,
    )

    assert observed["max_workers"] == 2
    assert result.rows_seen == 4
    assert result.complete is False
    assert len(result.rows) == 2
    assert "Website cell discovery shard 1: scanned 1 row" in caplog.text
    assert "Website cell discovery shard 2: scanned 1 row" in caplog.text


def test_merge_candidate_scan_results_unions_rows_counts_and_completion() -> None:
    merged = _merge_candidate_scan_results(
        (
            _DiscoveryScan(frozenset({"a"}), 2, (("a", {"id": "a"}),), True),
            _DiscoveryScan(frozenset({"b"}), 3, (("b", {"id": "b"}),), False),
        )
    )

    assert merged == _DiscoveryScan(
        frozenset({"a", "b"}),
        5,
        (("a", {"id": "a"}), ("b", {"id": "b"})),
        False,
    )


def test_select_discovery_rows_ranks_and_limits_each_cell() -> None:
    rows = (
        ("a", {"website_text": "Short."}),
        ("a", {"website_text": "Long. More context."}),
        ("b", {"website_text": "Other."}),
    )

    assert _select_discovery_rows(rows, max_rows_per_cell=1) == (
        {"website_text": "Long. More context."},
        {"website_text": "Other."},
    )


def test_discovery_row_rounds_handle_empty_and_multiple_rows_per_round() -> None:
    ranked = (({"id": "a1"}, {"id": "a2"}, {"id": "a3"}), ({"id": "b1"},))

    assert tuple(_discovery_row_rounds((), 2)) == ()
    assert tuple(_discovery_row_rounds(ranked, 2)) == (
        ({"id": "a1"}, {"id": "b1"}, {"id": "a2"}),
        ({"id": "a3"},),
    )


def test_interleave_discovery_rows_handles_empty_and_unequal_cells() -> None:
    assert _interleave_discovery_rows(()) == ()
    assert _interleave_discovery_rows((({"id": "a1"}, {"id": "a2"}), ({"id": "b1"},))) == (
        {"id": "a1"},
        {"id": "b1"},
        {"id": "a2"},
    )


def test_boundary_count_counts_only_sentence_punctuation() -> None:
    assert _boundary_count(("X marks one!", "No X here.")) == 2
