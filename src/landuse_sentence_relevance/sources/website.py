from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from itertools import batched
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.domain.stratification import select_spread_cells
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.protocols import (
    BatchSentenceSplitter,
    LanguageIdentifier,
    SentenceSplitter,
)
from landuse_sentence_relevance.sources.validation import (
    validate_candidate_cell_settings as _validate_candidate_cell_settings,
)
from landuse_sentence_relevance.sources.validation import validate_optional_limit as _validate_optional_limit
from landuse_sentence_relevance.sources.website_candidates import (
    EligibleRow,
    WebsiteCandidateBuilder,
)
from landuse_sentence_relevance.sources.website_discovery import (
    DiscoveryRow,
    RowShardsLoader,
    _bounded_candidates,
    _budgeted_rows,
    _candidate_cells,
    _counts_by_cell,
    _discovery_row_rounds,
    _DiscoveryScan,
    _ranked_discovery_rows,
    _scan_candidate_rows,
    _scan_candidate_shards,
    _select_discovery_rows,
    _source_budget_filled,
)
from landuse_sentence_relevance.sources.website_text import (
    WEBSITE_FIELD_SPECS,
    _location_from_row,
)

logger = logging.getLogger(__name__)
_WEBSITE_BATCH_SIZE = 32
_DISCOVERY_PREFILTER_OVERSAMPLE_FACTOR = 2
_DISCOVERY_PREFILTER_ROWS_PER_ROUND = 1


class WebsiteCandidateSource:
    """Stream both OSM website text fields and retain only English sentences."""

    _FIELD_SPECS = WEBSITE_FIELD_SPECS

    def __init__(
        self,
        row_loader: Callable[[], Iterable[Mapping[str, Any]]],
        splitter: SentenceSplitter,
        language_identifier: LanguageIdentifier,
        cell_for_location: Callable[[float, float], str],
        candidate_cell_count: int | None = None,
        center_of_cell: Callable[[str], tuple[float, float]] | None = None,
        seed: str = "website",
        allowed_cells: Iterable[str] | None = None,
        max_candidates_per_cell: int | None = None,
        minimum_candidate_cells: int | None = None,
        minimum_candidates_per_cell: int | None = None,
        max_rows_per_cell: int | None = None,
        max_text_characters: int | None = None,
        excluded_cells: Iterable[str] | None = None,
        row_shards_loader: RowShardsLoader | None = None,
        max_discovery_rows_per_shard: int | None = None,
        max_stream_workers: int = 4,
        require_paragraph: bool = False,
    ) -> None:
        _validate_candidate_cell_settings(candidate_cell_count, center_of_cell)
        _validate_optional_limit(max_candidates_per_cell, "max_candidates_per_cell")
        _validate_optional_limit(minimum_candidate_cells, "minimum_candidate_cells")
        _validate_optional_limit(minimum_candidates_per_cell, "minimum_candidates_per_cell")
        _validate_optional_limit(max_rows_per_cell, "max_rows_per_cell")
        _validate_optional_limit(max_text_characters, "max_text_characters")
        _validate_optional_limit(max_discovery_rows_per_shard, "max_discovery_rows_per_shard")
        _validate_optional_limit(max_stream_workers, "max_stream_workers")
        self._row_loader = row_loader
        self._splitter = splitter
        self._language_identifier = language_identifier
        self._cell_for_location = cell_for_location
        self._candidate_cell_count = candidate_cell_count
        self._center_of_cell = center_of_cell
        self._seed = seed
        self._candidate_cells: frozenset[str] | None = None
        self._discovery_rows: tuple[Mapping[str, Any], ...] = ()
        self._discovery_candidates: tuple[Candidate, ...] = ()
        self._discovery_candidate_counts: dict[str, int] = {}
        self._discovery_row_counts: dict[str, int] = {}
        self._discovery_rows_preprocessed = False
        self._discovery_scan_complete = False
        self._allowed_cells = None if allowed_cells is None else frozenset(allowed_cells)
        self._excluded_cells = frozenset(excluded_cells or ())
        self._max_text_characters = max_text_characters
        self._row_shards_loader = row_shards_loader
        self._max_discovery_rows_per_shard = max_discovery_rows_per_shard
        self._max_stream_workers = max_stream_workers
        self._require_paragraph = require_paragraph
        self._minimum_candidate_cells = minimum_candidate_cells
        self._candidate_quota = (
            None
            if max_candidates_per_cell is None
            else CellQuota(max_candidates_per_cell, minimum_candidates_per_cell)
        )
        self._candidate_builder = WebsiteCandidateBuilder(
            splitter=splitter,
            language_identifier=language_identifier,
            seed=seed,
            candidate_quota=self._candidate_quota,
            max_text_characters=max_text_characters,
            require_paragraph=require_paragraph,
        )
        self._row_quota = None if max_rows_per_cell is None else CellQuota(max_rows_per_cell)

    def iter_candidates(self) -> Iterable[Candidate]:
        self._discover_candidate_cells()
        logger.info("Website source: streaming polygon rows")
        rows_seen = sum(self._discovery_row_counts.values())
        candidates_seen = 0
        candidate_counts = dict(self._discovery_candidate_counts)
        row_counts = dict(self._discovery_row_counts)
        for candidate in self._discovery_candidates:
            candidates_seen += 1
            yield candidate
        rows = self._candidate_rows(candidate_counts, row_counts)
        for rows_seen, row_candidates in self._candidate_groups(rows, row_counts, candidate_counts):
            log_stream_progress(logger, "Website rows", rows_seen)
            for candidate in row_candidates:
                candidates_seen += 1
                yield candidate
        if _source_budget_filled(
            candidate_counts,
            self._candidate_quota,
            self._minimum_candidate_cells,
            row_counts,
            self._row_quota,
        ):
            logger.info("Website source: candidate cell budget filled; stopping row scan")
        logger.info(
            "Website source: streamed %d rows; yielded %d English candidates",
            rows_seen,
            candidates_seen,
        )

    def _candidate_groups(
        self,
        rows: Iterable[Mapping[str, Any]],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[tuple[int, Iterable[Candidate]]]:
        if not isinstance(self._splitter, BatchSentenceSplitter):
            yield from self._single_candidate_groups(rows, row_counts, candidate_counts)
            return
        yield from self._batch_candidate_groups(row_counts, candidate_counts)

    def _single_candidate_groups(
        self,
        rows: Iterable[Mapping[str, Any]],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[tuple[int, Iterable[Candidate]]]:
        for rows_seen, row in enumerate(rows, start=1):
            yield rows_seen, self._bounded_row_candidates(row, row_counts, candidate_counts)

    def _batch_candidate_groups(
        self,
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[tuple[int, Iterable[Candidate]]]:
        rows_seen = 0
        row_sources = (
            self._reusable_discovery_rows(),
            _budgeted_rows(
                self._row_loader,
                candidate_counts,
                self._candidate_quota,
                self._minimum_candidate_cells,
                row_counts,
                self._row_quota,
            ),
        )
        for source_index, source_rows in enumerate(row_sources):
            if source_index > 0 and self._candidate_scan_complete(candidate_counts, row_counts):
                return
            for row_batch in batched(source_rows, _WEBSITE_BATCH_SIZE):
                rows_seen += len(row_batch)
                yield (
                    rows_seen,
                    self._bounded_row_candidates_batch(
                        row_batch,
                        row_counts,
                        candidate_counts,
                    ),
                )

    def _discover_candidate_cells(self) -> frozenset[str] | None:
        if self._candidate_cell_count is None:
            return None
        return self._discover_configured_cells()

    def _discover_configured_cells(self) -> frozenset[str]:
        assert self._candidate_cell_count is not None
        assert self._center_of_cell is not None
        logger.info("Website source: discovering globally spread candidate H3 cells")
        scan = self._scan_candidate_cells()
        selected = self._selected_discovery_cells(scan)
        self._candidate_cells = frozenset(selected)
        logger.info(
            "Website source: selected %d globally spread candidate H3 cells from %d cells across %d rows",
            len(selected),
            len(scan.cells),
            scan.rows_seen,
        )
        self._prepare_discovery_rows(selected, scan.rows)
        self._discovery_scan_complete = scan.complete
        return self._candidate_cells

    def _selected_discovery_cells(self, scan: _DiscoveryScan) -> tuple[str, ...]:
        if not self._should_prefilter_discovery():
            return self._selected_candidate_cells(scan.cells)
        validation_cells = self._prefilter_discovery_cells(scan.cells)
        candidates, row_counts = self._prefilter_discovery_rows(scan.rows, validation_cells)
        selected = self._selected_candidate_cells(_candidate_cells(candidates))
        self._cache_prefiltered_discovery(candidates, row_counts, selected)
        return selected

    def _prefilter_discovery_cells(self, cells: Iterable[str]) -> frozenset[str]:
        assert self._candidate_cell_count is not None
        assert self._center_of_cell is not None
        target = self._minimum_candidate_cells or self._candidate_cell_count
        oversample_count = max(4, target * _DISCOVERY_PREFILTER_OVERSAMPLE_FACTOR)
        selected = select_spread_cells(
            cells,
            target_count=oversample_count,
            center_of_cell=self._center_of_cell,
            seed=f"{self._seed}:discovery-prefilter",
        )
        logger.info(
            "Website source: pre-filtering %d geographically spread cells out of %d discovered cells",
            len(selected),
            len(set(cells)),
        )
        return frozenset(selected)

    def _cache_prefiltered_discovery(
        self,
        candidates: tuple[Candidate, ...],
        row_counts: Mapping[str, int],
        selected: Iterable[str],
    ) -> None:
        selected_cells = frozenset(selected)
        self._discovery_candidates = tuple(
            candidate for candidate in candidates if candidate.h3_cell in selected_cells
        )
        self._discovery_candidate_counts = _counts_by_cell(self._discovery_candidates)
        self._discovery_row_counts = {
            cell: count for cell, count in row_counts.items() if cell in selected_cells
        }
        self._discovery_rows_preprocessed = True

    def _prepare_discovery_rows(
        self,
        selected: Iterable[str],
        discovery_rows: Iterable[DiscoveryRow],
    ) -> None:
        if self._discovery_rows_preprocessed:
            logger.info(
                "Website source: retained %d English candidates across %d cells during bounded discovery",
                len(self._discovery_candidates),
                len({candidate.h3_cell for candidate in self._discovery_candidates}),
            )
            return
        self._discovery_rows = self._selected_discovery_rows(selected, discovery_rows)

    def _should_prefilter_discovery(self) -> bool:
        return self._require_paragraph and self._candidate_quota is not None

    def _prefilter_discovery_rows(
        self,
        discovery_rows: Iterable[DiscoveryRow],
        cells: Iterable[str],
    ) -> tuple[tuple[Candidate, ...], dict[str, int]]:
        selected_cells = frozenset(cells)
        rows = tuple((cell, row) for cell, row in discovery_rows if cell in selected_cells)
        ranked_rows = _ranked_discovery_rows(rows, max_rows_per_cell=self._discovery_row_limit())
        row_counts: dict[str, int] = {}
        candidate_counts: dict[str, int] = {}
        if isinstance(self._splitter, BatchSentenceSplitter):
            candidates = self._prefilter_batch_rows(ranked_rows, row_counts, candidate_counts)
        else:
            candidates = self._prefilter_scalar_rows(ranked_rows, row_counts, candidate_counts)
        return tuple(candidates), row_counts

    def _prefilter_batch_rows(
        self,
        ranked_rows: tuple[tuple[Mapping[str, Any], ...], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> list[Candidate]:
        return self._collect_prefilter_rounds(
            _discovery_row_rounds(ranked_rows, _DISCOVERY_PREFILTER_ROWS_PER_ROUND),
            lambda row_round: self._prefilter_batch_round(
                row_round,
                row_counts,
                candidate_counts,
            ),
            row_counts,
            candidate_counts,
        )

    def _prefilter_batch_round(
        self,
        rows: tuple[Mapping[str, Any], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        return (
            candidate
            for row_batch in batched(rows, _WEBSITE_BATCH_SIZE)
            for candidate in self._bounded_row_candidates_batch(
                row_batch,
                row_counts,
                candidate_counts,
            )
        )

    def _prefilter_scalar_rows(
        self,
        ranked_rows: tuple[tuple[Mapping[str, Any], ...], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> list[Candidate]:
        return self._collect_prefilter_rounds(
            _discovery_row_rounds(ranked_rows, _DISCOVERY_PREFILTER_ROWS_PER_ROUND),
            lambda row_round: self._prefilter_scalar_round(row_round, row_counts, candidate_counts),
            row_counts,
            candidate_counts,
        )

    def _prefilter_scalar_round(
        self,
        rows: tuple[Mapping[str, Any], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        return (
            candidate
            for row in rows
            for candidate in self._bounded_row_candidates(row, row_counts, candidate_counts)
        )

    def _collect_prefilter_rounds(
        self,
        row_rounds: Iterable[tuple[Mapping[str, Any], ...]],
        process_round: Callable[[tuple[Mapping[str, Any], ...]], Iterable[Candidate]],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for row_round in row_rounds:
            candidates.extend(process_round(row_round))
            if self._prefilter_budget_reached(candidate_counts, row_counts):
                break
        return candidates

    def _prefilter_budget_reached(
        self,
        candidate_counts: Mapping[str, int],
        row_counts: Mapping[str, int],
    ) -> bool:
        return _source_budget_filled(
            candidate_counts,
            self._candidate_quota,
            self._minimum_candidate_cells,
            row_counts,
            self._row_quota,
        )

    def _selected_candidate_cells(self, cells: Iterable[str]) -> tuple[str, ...]:
        assert self._candidate_cell_count is not None
        assert self._center_of_cell is not None
        return select_spread_cells(
            cells,
            target_count=self._candidate_cell_count,
            center_of_cell=self._center_of_cell,
            seed=self._seed,
        )

    def _selected_discovery_rows(
        self,
        selected: Iterable[str],
        discovery_rows: Iterable[DiscoveryRow],
    ) -> tuple[Mapping[str, Any], ...]:
        selected_cells = frozenset(selected)
        return _select_discovery_rows(
            ((cell, row) for cell, row in discovery_rows if cell in selected_cells),
            max_rows_per_cell=self._discovery_row_limit(),
        )

    def _discovery_row_limit(self) -> int:
        return 1 if self._row_quota is None else self._row_quota.capacity

    def _scan_candidate_cells(self) -> _DiscoveryScan:
        rows_per_cell = None if self._row_quota is None else self._row_quota.capacity
        if self._row_shards_loader is None:
            return _scan_candidate_rows(
                self._row_loader(),
                self._candidate_cell,
                rows_per_cell,
                self._max_discovery_rows_per_shard,
                self._max_text_characters,
                "Website cell discovery",
            )
        return _scan_candidate_shards(
            tuple(self._row_shards_loader()),
            self._candidate_cell,
            rows_per_cell,
            self._max_discovery_rows_per_shard,
            self._max_text_characters,
            self._max_stream_workers,
        )

    def _reusable_discovery_rows(self) -> tuple[Mapping[str, Any], ...]:
        if self._discovery_rows_preprocessed or not self._has_reusable_discovery_rows():
            return ()
        return self._discovery_rows

    def _discovery_rows_exhausted(self, row_counts: Mapping[str, int]) -> bool:
        if not self._has_reusable_discovery_rows():
            return False
        return self._discovery_scan_complete or self._all_discovered_cell_rows_full(row_counts)

    def _has_reusable_discovery_rows(self) -> bool:
        return (
            self._candidate_quota is not None
            and self._row_quota is not None
            and self._candidate_cells is not None
        )

    def _all_discovered_cell_rows_full(self, row_counts: Mapping[str, int]) -> bool:
        if self._candidate_cells is None:
            return False
        assert self._row_quota is not None
        return all(self._row_quota.is_full(cell, row_counts) for cell in self._candidate_cells)

    def _candidate_scan_complete(
        self,
        candidate_counts: Mapping[str, int],
        row_counts: Mapping[str, int],
    ) -> bool:
        return (
            _source_budget_filled(
                candidate_counts,
                self._candidate_quota,
                self._minimum_candidate_cells,
                row_counts,
                self._row_quota,
            )
            or self._all_candidate_cells_full(candidate_counts)
            or self._discovery_rows_exhausted(row_counts)
        )

    def _all_candidate_cells_full(self, candidate_counts: Mapping[str, int]) -> bool:
        if self._candidate_cells is None or self._candidate_quota is None:
            return False
        return all(self._candidate_quota.is_full(cell, candidate_counts) for cell in self._candidate_cells)

    def _candidate_rows(
        self,
        candidate_counts: dict[str, int],
        row_counts: dict[str, int],
    ) -> Iterable[Mapping[str, Any]]:
        yield from self._reusable_discovery_rows()
        if self._candidate_scan_complete(candidate_counts, row_counts):
            return
        yield from _budgeted_rows(
            self._row_loader,
            candidate_counts,
            self._candidate_quota,
            self._minimum_candidate_cells,
            row_counts,
            self._row_quota,
        )

    def _candidate_cell(self, row: Mapping[str, Any]) -> str | None:
        if not self._candidate_builder.has_eligible_text(row):
            return None
        location = _location_from_row(row)
        if location is None:
            return None
        _, latitude, longitude = location
        cell = self._cell_for_location(latitude, longitude)
        return cell if self._is_allowed_cell(cell) else None

    def _bounded_row_candidates(
        self,
        row: Mapping[str, Any],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        eligible = self._eligible_row(row, row_counts)
        if eligible is None:
            return
        _, cell = eligible
        yield from _bounded_candidates(
            self._row_candidates(row, cell),
            candidate_counts,
            self._candidate_quota,
            cell,
        )

    def _bounded_row_candidates_batch(
        self,
        rows: tuple[Mapping[str, Any], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        eligible_rows = self._eligible_batch_rows(rows, row_counts, candidate_counts)
        if self._uses_single_candidate_batch():
            yield from self._one_candidate_per_cell_batch(eligible_rows, candidate_counts)
            return

        yield from self._bounded_batch_candidates(eligible_rows, candidate_counts)

    def _eligible_batch_rows(
        self,
        rows: tuple[Mapping[str, Any], ...],
        row_counts: dict[str, int],
        candidate_counts: dict[str, int],
    ) -> tuple[EligibleRow, ...]:
        return tuple(
            eligible
            for row in rows
            if (eligible := self._eligible_row(row, row_counts)) is not None
            and not self._candidate_cell_is_full(eligible[1], candidate_counts)
        )

    def _uses_single_candidate_batch(self) -> bool:
        return self._candidate_quota is not None and self._candidate_quota.capacity == 1

    def _bounded_batch_candidates(
        self,
        rows: tuple[EligibleRow, ...],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        candidate_groups = self._row_candidates_batch(rows)
        for (_, cell), candidates in zip(rows, candidate_groups, strict=True):
            yield from _bounded_candidates(candidates, candidate_counts, self._candidate_quota, cell)

    def _candidate_cell_is_full(self, cell: str, candidate_counts: Mapping[str, int]) -> bool:
        return self._candidate_quota is not None and self._candidate_quota.is_full(cell, candidate_counts)

    def _one_candidate_per_cell_batch(
        self,
        rows: tuple[EligibleRow, ...],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        candidate_groups = self._row_candidates_batch(rows)
        assert self._candidate_quota is not None
        for (_, cell), candidates in zip(rows, candidate_groups, strict=True):
            if self._candidate_cell_is_full(cell, candidate_counts):
                continue
            selected = tuple(_bounded_candidates(candidates, candidate_counts, self._candidate_quota, cell))
            if selected:
                yield from selected

    def _eligible_row(
        self,
        row: Mapping[str, Any],
        row_counts: dict[str, int],
    ) -> EligibleRow | None:
        cell = self._eligible_cell(row)
        if cell is None or not self._candidate_builder.has_eligible_text(row):
            return None
        if self._row_quota is not None and self._row_quota.is_full(cell, row_counts):
            return None
        row_counts[cell] = row_counts.get(cell, 0) + 1
        return row, cell

    def _eligible_cell(self, row: Mapping[str, Any]) -> str | None:
        location = _location_from_row(row)
        if location is None:
            return None
        _, latitude, longitude = location
        cell = self._cell_for_location(latitude, longitude)
        return cell if self._is_allowed_cell(cell) else None

    def _is_allowed_cell(self, cell: str) -> bool:
        return (
            cell not in self._excluded_cells
            and (self._allowed_cells is None or cell in self._allowed_cells)
            and (self._candidate_cells is None or cell in self._candidate_cells)
        )

    def _row_candidates(self, row: Mapping[str, Any], cell: str | None = None) -> Iterable[Candidate]:
        cell = self._eligible_cell(row) if cell is None else cell
        if cell is None:
            return
        yield from self._candidate_builder.candidates_for_row(row, cell)

    def _row_candidates_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        return self._candidate_builder.candidates_for_rows(rows)
