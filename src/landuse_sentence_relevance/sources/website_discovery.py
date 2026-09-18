"""Discovery, ranking, and bounded streaming helpers for website rows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat, zip_longest
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.website_text import (
    WEBSITE_FIELD_SPECS,
    bounded_text,
    field_text,
)

DiscoveryRow = tuple[str, Mapping[str, Any]]
RowShardsLoader = Callable[[], Iterable[Iterable[Mapping[str, Any]]]]
logger = logging.getLogger("landuse_sentence_relevance.sources.website")


@dataclass(frozen=True, slots=True)
class DiscoveryScan:
    cells: frozenset[str]
    rows_seen: int
    rows: tuple[DiscoveryRow, ...]
    complete: bool


def counts_by_cell(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.h3_cell] = counts.get(candidate.h3_cell, 0) + 1
    return counts


def candidate_cells(candidates: Iterable[Candidate]) -> set[str]:
    return {candidate.h3_cell for candidate in candidates}


def _compact_row(row: Mapping[str, Any], max_text_characters: int | None) -> Mapping[str, Any]:
    fields = {
        field: row.get(field)
        for field in (
            "polygon_id",
            "osm_id",
            "lat",
            "lon",
            "name",
            "region",
            "website",
            "contact_website",
        )
    }
    for text_field, _ in WEBSITE_FIELD_SPECS:
        text = field_text(row, text_field)
        if text is not None:
            fields[text_field] = bounded_text(text, max_text_characters)
    return fields


def scan_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
    candidate_cell: Callable[[Mapping[str, Any]], str | None],
    rows_per_cell: int | None,
    max_rows: int | None,
    max_text_characters: int | None,
    progress_label: str,
) -> DiscoveryScan:
    cells: set[str] = set()
    discovery_rows: list[DiscoveryRow] = []
    row_counts: dict[str, int] = {}
    rows_seen = 0
    for row in rows:
        rows_seen += 1
        log_stream_progress(logger, progress_label, rows_seen)
        cell = candidate_cell(row)
        if cell is not None:
            cells.add(cell)
            discovery = _discovery_row(row, cell, row_counts, rows_per_cell, max_text_characters)
            if discovery is not None:
                discovery_rows.append(discovery)
        if _scan_limit_reached(rows_seen, max_rows):
            break
    return DiscoveryScan(
        cells=frozenset(cells),
        rows_seen=rows_seen,
        rows=tuple(discovery_rows),
        complete=not _scan_limit_reached(rows_seen, max_rows),
    )


def _discovery_row(
    row: Mapping[str, Any],
    cell: str,
    row_counts: dict[str, int],
    rows_per_cell: int | None,
    max_text_characters: int | None,
) -> DiscoveryRow | None:
    if rows_per_cell is not None and row_counts.get(cell, 0) >= rows_per_cell:
        return None
    row_counts[cell] = row_counts.get(cell, 0) + 1
    return cell, _compact_row(row, max_text_characters)


def _scan_limit_reached(rows_seen: int, max_rows: int | None) -> bool:
    return max_rows is not None and rows_seen >= max_rows


def scan_candidate_shards(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    candidate_cell: Callable[[Mapping[str, Any]], str | None],
    rows_per_cell: int | None,
    max_rows: int | None,
    max_text_characters: int | None,
    max_workers: int,
) -> DiscoveryScan:
    if not streams:
        return DiscoveryScan(cells=frozenset(), rows_seen=0, rows=(), complete=True)
    if len(streams) == 1:
        results = (
            scan_candidate_rows(
                streams[0],
                candidate_cell,
                rows_per_cell,
                max_rows,
                max_text_characters,
                "Website cell discovery shard 1",
            ),
        )
    else:
        arguments = (
            streams,
            repeat(candidate_cell),
            repeat(rows_per_cell),
            repeat(max_rows),
            repeat(max_text_characters),
            (f"Website cell discovery shard {index + 1}" for index in range(len(streams))),
        )
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(streams)),
        ) as executor:
            results = executor.map(scan_candidate_rows, *arguments)
    return _merge_candidate_scan_results(results)


def _merge_candidate_scan_results(
    results: Iterable[DiscoveryScan],
) -> DiscoveryScan:
    cells: set[str] = set()
    discovery_rows: list[DiscoveryRow] = []
    rows_seen = 0
    complete = True
    for result in results:
        cells.update(result.cells)
        rows_seen += result.rows_seen
        discovery_rows.extend(result.rows)
        complete = complete and result.complete
    return DiscoveryScan(
        cells=frozenset(cells),
        rows_seen=rows_seen,
        rows=tuple(discovery_rows),
        complete=complete,
    )


def select_discovery_rows(
    rows: Iterable[DiscoveryRow],
    max_rows_per_cell: int,
) -> tuple[Mapping[str, Any], ...]:
    return _interleave_discovery_rows(ranked_discovery_rows(rows, max_rows_per_cell))


def ranked_discovery_rows(
    rows: Iterable[DiscoveryRow],
    max_rows_per_cell: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    rows_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for cell, row in rows:
        rows_by_cell.setdefault(cell, []).append(row)
    return tuple(
        tuple(sorted(cell_rows, key=_discovery_score, reverse=True)[:max_rows_per_cell])
        for cell_rows in rows_by_cell.values()
    )


def discovery_row_rounds(
    ranked_rows: tuple[tuple[Mapping[str, Any], ...], ...],
    rows_per_round: int,
) -> Iterable[tuple[Mapping[str, Any], ...]]:
    max_rows = max((len(cell_rows) for cell_rows in ranked_rows), default=0)
    for offset in range(0, max_rows, rows_per_round):
        yield _interleave_discovery_rows(
            tuple(cell_rows[offset : offset + rows_per_round] for cell_rows in ranked_rows)
        )


def _interleave_discovery_rows(
    ranked_rows: tuple[tuple[Mapping[str, Any], ...], ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row_group in zip_longest(*ranked_rows) for row in row_group if row is not None)


def _discovery_score(row: Mapping[str, Any]) -> tuple[int, int]:
    texts = _discovery_texts(row)
    return _boundary_count(texts), sum(len(text) for text in texts)


def _discovery_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        text for text_field, _ in WEBSITE_FIELD_SPECS if (text := field_text(row, text_field)) is not None
    )


def _boundary_count(texts: Iterable[str]) -> int:
    return sum(sum(text.count(marker) for marker in ".!?") for text in texts)


def budgeted_rows(
    row_loader: Callable[[], Iterable[Mapping[str, Any]]],
    candidate_counts: Mapping[str, int],
    candidate_quota: CellQuota | None,
    minimum_candidate_cells: int | None,
    row_counts: Mapping[str, int],
    row_quota: CellQuota | None,
) -> Iterable[Mapping[str, Any]]:
    rows = iter(row_loader())
    while not source_budget_filled(
        candidate_counts,
        candidate_quota,
        minimum_candidate_cells,
        row_counts,
        row_quota,
    ):
        try:
            yield next(rows)
        except StopIteration:
            return


def source_budget_filled(
    candidate_counts: Mapping[str, int],
    candidate_quota: CellQuota | None,
    minimum_candidate_cells: int | None,
    row_counts: Mapping[str, int],
    row_quota: CellQuota | None,
) -> bool:
    if candidate_quota is not None:
        return _budget_reached(candidate_quota, candidate_counts, minimum_candidate_cells)
    return _budget_reached(
        row_quota,
        row_counts,
        minimum_candidate_cells,
    )


def _budget_reached(
    quota: CellQuota | None,
    counts: Mapping[str, int],
    target_cells: int | None,
) -> bool:
    return (
        quota is not None
        and target_cells is not None
        and quota.is_reached(counts, counts, target_cells=target_cells)
    )


def bounded_candidates(
    candidates: Iterable[Candidate],
    candidate_counts: dict[str, int],
    candidate_quota: CellQuota | None,
    cell: str,
) -> Iterable[Candidate]:
    iterator = iter(candidates)
    while candidate_quota is None or not candidate_quota.is_full(cell, candidate_counts):
        try:
            candidate = next(iterator)
        except StopIteration:
            return
        candidate_counts[candidate.h3_cell] = candidate_counts.get(candidate.h3_cell, 0) + 1
        yield candidate


logger = logging.getLogger(__name__)

#: What this module offers the website source, and nothing else.
#:
#: These names carried a leading underscore while `website.py` imported all eleven of them --
#: the split moved the code out but never promoted its contract, so the marker said "internal"
#: while another module depended on every one. They are listed here rather than left implicit
#: because eleven names for one consumer is a surface worth seeing in one place.
__all__ = [
    "DiscoveryRow",
    "DiscoveryScan",
    "RowShardsLoader",
    "bounded_candidates",
    "budgeted_rows",
    "candidate_cells",
    "counts_by_cell",
    "discovery_row_rounds",
    "ranked_discovery_rows",
    "scan_candidate_rows",
    "scan_candidate_shards",
    "select_discovery_rows",
    "source_budget_filled",
]
