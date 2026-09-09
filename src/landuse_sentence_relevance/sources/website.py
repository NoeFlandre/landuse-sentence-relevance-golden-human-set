from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched, repeat
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sentence_selection import (
    SentencePart,
    candidate_sentence_parts,
    first_prioritized_sentence,
    prioritize_sentences,
)
from landuse_sentence_relevance.domain.stratification import select_spread_cells
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.protocols import (
    BatchLanguageIdentifier,
    BatchSentenceSplitter,
    LanguageIdentifier,
    SentenceSplitter,
    split_sentences_many,
)
from landuse_sentence_relevance.sources.validation import (
    validate_candidate_cell_settings as _validate_candidate_cell_settings,
)
from landuse_sentence_relevance.sources.validation import validate_optional_limit as _validate_optional_limit

Location = tuple[str, float, float]
FieldSpec = tuple[str, str]
DiscoveryRow = tuple[str, Mapping[str, Any]]
EligibleRow = tuple[Mapping[str, Any], str]
RowShardsLoader = Callable[[], Iterable[Iterable[Mapping[str, Any]]]]
logger = logging.getLogger(__name__)
_WEBSITE_BATCH_SIZE = 32
_LANGUAGE_SELECTION_BATCH_SIZE = 8
_CONTEXTUAL_BOUNDARY_PATTERN = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")
_MIN_CONTEXTUAL_BOUNDARIES = 2
_DISCOVERY_PREFILTER_OVERSAMPLE_FACTOR = 2
_DISCOVERY_PREFILTER_ROWS_PER_ROUND = 1


@dataclass(frozen=True, slots=True)
class _FieldWork:
    row_index: int
    row: Mapping[str, Any]
    polygon_id: str
    latitude: float
    longitude: float
    cell: str
    text_field: str
    url_field: str
    website_url: str | None
    text: str


@dataclass(frozen=True, slots=True)
class _DiscoveryScan:
    cells: frozenset[str]
    rows_seen: int
    rows: tuple[DiscoveryRow, ...]
    complete: bool


class WebsiteCandidateSource:
    """Stream both OSM website text fields and retain only English sentences."""

    _FIELD_SPECS: tuple[FieldSpec, ...] = (
        ("website_text", "website"),
        ("contact_website_text", "contact_website"),
    )

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
        if not self._has_eligible_website_text(row):
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
        if cell is None or not self._has_eligible_website_text(row):
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
        location = _location_from_row(row)
        if location is None:
            return
        polygon_id, latitude, longitude = location
        cell = self._eligible_cell(row) if cell is None else cell
        if cell is None:
            return
        for text_field, url_field in self._FIELD_SPECS:
            yield from self._field_candidates(
                row,
                polygon_id,
                latitude,
                longitude,
                cell,
                text_field,
                url_field,
            )

    def _row_candidates_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        if self._candidate_quota is not None and self._candidate_quota.capacity == 1:
            return self._single_candidate_field_batch(rows)
        return self._all_field_candidates_batch(rows)

    def _all_field_candidates_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        work = self._field_work_for_rows(rows)
        sentence_groups = split_sentences_many(self._splitter, (field.text for field in work))
        return self._group_row_candidates(rows, work, sentence_groups)

    def _single_candidate_field_batch(
        self,
        rows: tuple[EligibleRow, ...],
    ) -> tuple[tuple[Candidate, ...], ...]:
        candidates: list[list[Candidate]] = [[] for _ in rows]
        pending_indexes = tuple(range(len(rows)))
        for field_spec in self._FIELD_SPECS:
            if not pending_indexes:
                break
            pending_indexes = self._field_batch_round(rows, pending_indexes, field_spec, candidates)
        return tuple(tuple(row_candidates) for row_candidates in candidates)

    def _field_batch_round(
        self,
        rows: tuple[EligibleRow, ...],
        pending_indexes: tuple[int, ...],
        field_spec: FieldSpec,
        candidates: list[list[Candidate]],
    ) -> tuple[int, ...]:
        work = self._field_batch_work(rows, pending_indexes, field_spec)
        if not work:
            return pending_indexes
        sentence_groups = split_sentences_many(self._splitter, (field.text for field in work))
        self._append_batch_field_candidates(candidates, work, sentence_groups)
        return _pending_field_indexes(pending_indexes, candidates)

    def _field_batch_work(
        self,
        rows: tuple[EligibleRow, ...],
        pending_indexes: tuple[int, ...],
        field_spec: FieldSpec,
    ) -> tuple[_FieldWork, ...]:
        pending_rows = tuple(rows[index] for index in pending_indexes)
        return self._field_work_for_rows(pending_rows, (field_spec,), pending_indexes)

    def _append_batch_field_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> None:
        language_identifier = self._language_identifier
        if isinstance(language_identifier, BatchLanguageIdentifier):
            self._append_batch_language_candidates(
                candidates,
                work,
                sentence_groups,
                language_identifier,
            )
            return
        self._append_scalar_field_candidates(candidates, work, sentence_groups)

    def _append_scalar_field_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> None:
        for field, sentences in zip(work, sentence_groups, strict=True):
            candidate = self._first_field_candidate(field, sentences)
            if candidate is not None:
                candidates[field.row_index].append(candidate)

    def _append_batch_language_candidates(
        self,
        candidates: list[list[Candidate]],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
        language_identifier: BatchLanguageIdentifier,
    ) -> None:
        part_groups = _candidate_part_groups(work, sentence_groups, self._seed)
        selected_parts = _select_batch_parts(part_groups, language_identifier)
        for field, part in zip(work, selected_parts, strict=True):
            if part is not None:
                candidate = self._candidate_for_sentence(
                    part.text,
                    part.index,
                    field.polygon_id,
                    field.latitude,
                    field.longitude,
                    field.cell,
                    field.text_field,
                    field.website_url,
                    field.row,
                )
                if candidate is not None:
                    candidates[field.row_index].append(candidate)

    def _first_field_candidate(
        self,
        field: _FieldWork,
        sentences: Iterable[str],
    ) -> Candidate | None:
        part = first_prioritized_sentence(
            sentences,
            seed=f"{self._seed}:{field.polygon_id}:{field.text_field}",
            accept=self._language_identifier.is_english,
        )
        if part is None:
            return None
        return self._candidate_for_sentence(
            part.text,
            part.index,
            field.polygon_id,
            field.latitude,
            field.longitude,
            field.cell,
            field.text_field,
            field.website_url,
            field.row,
        )

    def _field_work_for_rows(
        self,
        rows: tuple[EligibleRow, ...],
        field_specs: tuple[FieldSpec, ...] | None = None,
        row_indexes: tuple[int, ...] | None = None,
    ) -> tuple[_FieldWork, ...]:
        work: list[_FieldWork] = []
        indexes = tuple(range(len(rows))) if row_indexes is None else row_indexes
        specs = self._FIELD_SPECS if field_specs is None else field_specs
        for row_index, (row, cell) in zip(indexes, rows, strict=True):
            work.extend(self._field_work(row_index, row, cell, specs))
        return tuple(work)

    def _group_row_candidates(
        self,
        rows: tuple[EligibleRow, ...],
        work: tuple[_FieldWork, ...],
        sentence_groups: Iterable[Iterable[str]],
    ) -> tuple[tuple[Candidate, ...], ...]:
        candidates: list[list[Candidate]] = [[] for _ in rows]
        for field, sentences in zip(work, sentence_groups, strict=True):
            candidates[field.row_index].extend(self._field_candidates_for_work(field, sentences))
        return tuple(tuple(row_candidates) for row_candidates in candidates)

    def _field_candidates_for_work(
        self,
        field: _FieldWork,
        sentences: Iterable[str],
    ) -> Iterable[Candidate]:
        return self._field_candidates(
            field.row,
            field.polygon_id,
            field.latitude,
            field.longitude,
            field.cell,
            field.text_field,
            field.url_field,
            text=field.text,
            sentences=sentences,
        )

    def _field_work(
        self,
        row_index: int,
        row: Mapping[str, Any],
        cell: str,
        field_specs: tuple[FieldSpec, ...] | None = None,
    ) -> tuple[_FieldWork, ...]:
        location = _location_from_row(row)
        if location is None:
            return ()
        specs = self._FIELD_SPECS if field_specs is None else field_specs
        return self._field_work_for_specs(row_index, row, cell, location, specs)

    def _field_work_for_specs(
        self,
        row_index: int,
        row: Mapping[str, Any],
        cell: str,
        location: Location,
        specs: tuple[FieldSpec, ...],
    ) -> tuple[_FieldWork, ...]:
        polygon_id, latitude, longitude = location
        work: list[_FieldWork] = []
        for text_field, url_field in specs:
            text = _field_text(row, text_field)
            if text is None:
                continue
            bounded = _bounded_text(text, self._max_text_characters)
            if not self._is_eligible_text(bounded):
                continue
            work.append(
                _FieldWork(
                    row_index=row_index,
                    row=row,
                    polygon_id=polygon_id,
                    latitude=latitude,
                    longitude=longitude,
                    cell=cell,
                    text_field=text_field,
                    url_field=url_field,
                    website_url=_field_url(row, url_field),
                    text=bounded,
                )
            )
        return tuple(work)

    def _field_candidates(
        self,
        row: Mapping[str, Any],
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        text_field: str,
        url_field: str,
        text: str | None = None,
        sentences: Iterable[str] | None = None,
    ) -> Iterable[Candidate]:
        text = _field_text(row, text_field) if text is None else text
        if text is None or not self._is_eligible_text(_bounded_text(text, self._max_text_characters)):
            return
        text = _bounded_text(text, self._max_text_characters)
        website_url = _field_url(row, url_field)
        yield from self._field_candidate_parts(
            self._sentences_for_text(text, sentences),
            seed=f"{self._seed}:{polygon_id}:{text_field}",
            polygon_id=polygon_id,
            latitude=latitude,
            longitude=longitude,
            cell=cell,
            text_field=text_field,
            website_url=website_url,
            row=row,
        )

    def _sentences_for_text(
        self,
        text: str,
        sentences: Iterable[str] | None,
    ) -> Iterable[str]:
        if sentences is not None:
            return sentences
        return self._splitter.split(text)

    def _has_eligible_website_text(self, row: Mapping[str, Any]) -> bool:
        return any(
            text is not None and self._is_eligible_text(_bounded_text(text, self._max_text_characters))
            for text_field, _ in self._FIELD_SPECS
            if (text := _field_text(row, text_field)) is not None
        )

    def _is_eligible_text(self, text: str) -> bool:
        return not self._require_paragraph or _is_contextual_text(text)

    def _field_candidate_parts(
        self,
        sentences: Iterable[str],
        seed: str,
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        text_field: str,
        website_url: str | None,
        row: Mapping[str, Any],
    ) -> Iterable[Candidate]:
        for part in prioritize_sentences(
            sentences,
            seed=seed,
            accept=self._language_identifier.is_english,
        ):
            candidate = self._candidate_for_sentence(
                part.text,
                part.index,
                polygon_id,
                latitude,
                longitude,
                cell,
                text_field,
                website_url,
                row,
            )
            if candidate is None:
                continue
            yield candidate

    def _candidate_for_sentence(
        self,
        sentence: str,
        sentence_index: int,
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        field: str,
        website_url: str | None,
        row: Mapping[str, Any],
    ) -> Candidate | None:
        cleaned = sentence.strip()
        if not cleaned:
            return None
        return Candidate(
            candidate_id=f"website:{polygon_id}:{field}:{sentence_index}",
            sentence=cleaned,
            source=Source.WEBSITE,
            source_record_id=polygon_id,
            source_field=field,
            h3_cell=cell,
            h3_resolution=3,
            latitude=latitude,
            longitude=longitude,
            place_name=row.get("name"),
            region=row.get("region"),
            source_url=website_url,
        )


def _location_from_row(row: Mapping[str, Any]) -> Location | None:
    polygon_id = row.get("polygon_id") or row.get("osm_id")
    latitude = row.get("lat")
    longitude = row.get("lon")
    if polygon_id is None or latitude is None or longitude is None:
        return None
    return str(polygon_id), float(latitude), float(longitude)


def _field_text(row: Mapping[str, Any], field: str) -> str | None:
    text = row.get(field)
    return text.strip() if isinstance(text, str) and text.strip() else None


def _field_url(row: Mapping[str, Any], field: str) -> str | None:
    url = row.get(field)
    return url if isinstance(url, str) else None


def _candidate_part_groups(
    work: tuple[_FieldWork, ...],
    sentence_groups: Iterable[Iterable[str]],
    seed: str,
) -> tuple[tuple[SentencePart, ...], ...]:
    return tuple(
        candidate_sentence_parts(
            sentences,
            seed=f"{seed}:{field.polygon_id}:{field.text_field}",
        )
        for field, sentences in zip(work, sentence_groups, strict=True)
    )


def _select_batch_parts(
    part_groups: tuple[tuple[SentencePart, ...], ...],
    language_identifier: BatchLanguageIdentifier,
) -> tuple[SentencePart | None, ...]:
    pending = list(part_groups)
    selected: list[SentencePart | None] = [None] * len(part_groups)
    while active_groups := _language_selection_groups(pending, selected):
        texts = tuple(part.text for _, parts in active_groups for part in parts)
        accepted = tuple(language_identifier.is_english_many(texts))
        _apply_language_selection_round(pending, selected, active_groups, accepted)
    return tuple(selected)


def _language_selection_groups(
    pending: list[tuple[SentencePart, ...]],
    selected: list[SentencePart | None],
) -> tuple[tuple[int, tuple[SentencePart, ...]], ...]:
    groups: list[tuple[int, tuple[SentencePart, ...]]] = []
    for index, parts in enumerate(pending):
        if selected[index] is not None:
            continue
        first_batch = next(batched(parts, _LANGUAGE_SELECTION_BATCH_SIZE), ())
        if first_batch:
            groups.append((index, first_batch))
    return tuple(groups)


def _apply_language_selection_round(
    pending: list[tuple[SentencePart, ...]],
    selected: list[SentencePart | None],
    active_groups: tuple[tuple[int, tuple[SentencePart, ...]], ...],
    accepted: tuple[bool, ...],
) -> None:
    offset = 0
    for index, parts in active_groups:
        offset = _apply_language_group(pending, selected, index, parts, accepted, offset)


def _apply_language_group(
    pending: list[tuple[SentencePart, ...]],
    selected: list[SentencePart | None],
    index: int,
    parts: tuple[SentencePart, ...],
    accepted: tuple[bool, ...],
    offset: int,
) -> int:
    flags = accepted[offset : offset + len(parts)]
    selected_part = _first_accepted_part(parts, flags)
    selected[index] = selected_part
    pending[index] = () if selected_part is not None else pending[index][len(parts) :]
    return offset + len(parts)


def _selected_parts(
    part_groups: tuple[tuple[SentencePart, ...], ...],
    accepted: tuple[bool, ...],
) -> tuple[SentencePart | None, ...]:
    selected: list[SentencePart | None] = []
    offset = 0
    for parts in part_groups:
        flags = accepted[offset : offset + len(parts)]
        selected.append(_first_accepted_part(parts, flags))
        offset += len(parts)
    return tuple(selected)


def _first_accepted_part(
    parts: tuple[SentencePart, ...],
    accepted: tuple[bool, ...],
) -> SentencePart | None:
    return next(
        (part for part, is_accepted in zip(parts, accepted, strict=True) if is_accepted),
        None,
    )


def _bounded_text(text: str, max_characters: int | None) -> str:
    return text if max_characters is None else text[:max_characters]


def _pending_field_indexes(
    pending_indexes: tuple[int, ...],
    candidates: list[list[Candidate]],
) -> tuple[int, ...]:
    return tuple(index for index in pending_indexes if not candidates[index])


def _counts_by_cell(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.h3_cell] = counts.get(candidate.h3_cell, 0) + 1
    return counts


def _candidate_cells(candidates: Iterable[Candidate]) -> set[str]:
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
    for text_field, _ in WebsiteCandidateSource._FIELD_SPECS:
        text = _field_text(row, text_field)
        if text is not None:
            fields[text_field] = _bounded_text(text, max_text_characters)
    return fields


def _has_website_text(row: Mapping[str, Any]) -> bool:
    return any(_field_text(row, field) is not None for field, _ in WebsiteCandidateSource._FIELD_SPECS)


def _is_contextual_text(text: str) -> bool:
    return len(_CONTEXTUAL_BOUNDARY_PATTERN.findall(text)) >= _MIN_CONTEXTUAL_BOUNDARIES


def _scan_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
    candidate_cell: Callable[[Mapping[str, Any]], str | None],
    rows_per_cell: int | None,
    max_rows: int | None,
    max_text_characters: int | None,
    progress_label: str,
) -> _DiscoveryScan:
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
    return _DiscoveryScan(
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


def _scan_candidate_shards(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    candidate_cell: Callable[[Mapping[str, Any]], str | None],
    rows_per_cell: int | None,
    max_rows: int | None,
    max_text_characters: int | None,
    max_workers: int,
) -> _DiscoveryScan:
    if not streams:
        return _DiscoveryScan(cells=frozenset(), rows_seen=0, rows=(), complete=True)
    if len(streams) == 1:
        results = (
            _scan_candidate_rows(
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
            thread_name_prefix="website-discovery",
        ) as executor:
            results = executor.map(_scan_candidate_rows, *arguments)
    return _merge_candidate_scan_results(results)


def _merge_candidate_scan_results(
    results: Iterable[_DiscoveryScan],
) -> _DiscoveryScan:
    cells: set[str] = set()
    discovery_rows: list[DiscoveryRow] = []
    rows_seen = 0
    complete = True
    for result in results:
        cells.update(result.cells)
        rows_seen += result.rows_seen
        discovery_rows.extend(result.rows)
        complete = complete and result.complete
    return _DiscoveryScan(
        cells=frozenset(cells),
        rows_seen=rows_seen,
        rows=tuple(discovery_rows),
        complete=complete,
    )


def _select_discovery_rows(
    rows: Iterable[DiscoveryRow],
    max_rows_per_cell: int,
) -> tuple[Mapping[str, Any], ...]:
    return _interleave_discovery_rows(_ranked_discovery_rows(rows, max_rows_per_cell))


def _ranked_discovery_rows(
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


def _discovery_row_rounds(
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
    selected: list[Mapping[str, Any]] = []
    for row_index in range(max((len(cell_rows) for cell_rows in ranked_rows), default=0)):
        for cell_rows in ranked_rows:
            if row_index < len(cell_rows):
                selected.append(cell_rows[row_index])
    return tuple(selected)


def _discovery_score(row: Mapping[str, Any]) -> tuple[int, int]:
    texts = _discovery_texts(row)
    return _boundary_count(texts), sum(len(text) for text in texts)


def _discovery_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        text
        for text_field, _ in WebsiteCandidateSource._FIELD_SPECS
        if (text := _field_text(row, text_field)) is not None
    )


def _boundary_count(texts: Iterable[str]) -> int:
    return sum(sum(text.count(marker) for marker in ".!?") for text in texts)


def _budgeted_rows(
    row_loader: Callable[[], Iterable[Mapping[str, Any]]],
    candidate_counts: Mapping[str, int],
    candidate_quota: CellQuota | None,
    minimum_candidate_cells: int | None,
    row_counts: Mapping[str, int],
    row_quota: CellQuota | None,
) -> Iterable[Mapping[str, Any]]:
    rows = iter(row_loader())
    while not _source_budget_filled(
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


def _source_budget_filled(
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


def _bounded_candidates(
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
