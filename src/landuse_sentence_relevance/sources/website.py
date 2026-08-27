from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.stratification import select_spread_cells
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.protocols import LanguageIdentifier, SentenceSplitter

Location = tuple[str, float, float]
FieldSpec = tuple[str, str]
logger = logging.getLogger(__name__)


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
    ) -> None:
        _validate_candidate_cell_settings(candidate_cell_count, center_of_cell)
        _validate_optional_limit(max_candidates_per_cell, "max_candidates_per_cell")
        _validate_optional_limit(minimum_candidate_cells, "minimum_candidate_cells")
        _validate_optional_limit(minimum_candidates_per_cell, "minimum_candidates_per_cell")
        _validate_optional_limit(max_rows_per_cell, "max_rows_per_cell")
        self._row_loader = row_loader
        self._splitter = splitter
        self._language_identifier = language_identifier
        self._cell_for_location = cell_for_location
        self._candidate_cell_count = candidate_cell_count
        self._center_of_cell = center_of_cell
        self._seed = seed
        self._candidate_cells: frozenset[str] | None = None
        self._allowed_cells = None if allowed_cells is None else frozenset(allowed_cells)
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
        rows_seen = 0
        candidates_seen = 0
        candidate_counts: dict[str, int] = {}
        row_counts: dict[str, int] = {}
        for rows_seen, row in enumerate(
            _budgeted_rows(
                self._row_loader,
                candidate_counts,
                self._candidate_quota,
                self._minimum_candidate_cells,
                row_counts,
                self._row_quota,
            ),
            start=1,
        ):
            log_stream_progress(logger, "Website rows", rows_seen)
            for candidate in self._bounded_row_candidates(row, row_counts, candidate_counts):
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

    def _discover_candidate_cells(self) -> frozenset[str] | None:
        if self._candidate_cell_count is None:
            return None
        assert self._center_of_cell is not None
        logger.info("Website source: discovering globally spread candidate H3 cells")
        cells, rows_seen = self._scan_candidate_cells()
        selected = select_spread_cells(
            cells,
            target_count=self._candidate_cell_count,
            center_of_cell=self._center_of_cell,
            seed=self._seed,
        )
        self._candidate_cells = frozenset(selected)
        logger.info(
            "Website source: selected %d globally spread candidate H3 cells from %d cells across %d rows",
            len(selected),
            len(cells),
            rows_seen,
        )
        return self._candidate_cells

    def _scan_candidate_cells(self) -> tuple[set[str], int]:
        cells: set[str] = set()
        rows_seen = 0
        for rows_seen, row in enumerate(self._row_loader(), start=1):
            log_stream_progress(logger, "Website cell discovery", rows_seen)
            cell = self._candidate_cell(row)
            if cell is not None:
                cells.add(cell)
        return cells, rows_seen

    def _candidate_cell(self, row: Mapping[str, Any]) -> str | None:
        if not _has_website_text(row):
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
        cell = self._eligible_cell(row)
        if cell is None or not _has_website_text(row):
            return
        if self._row_quota is not None and self._row_quota.is_full(cell, row_counts):
            return
        row_counts[cell] = row_counts.get(cell, 0) + 1
        yield from _bounded_candidates(
            self._row_candidates(row, cell),
            candidate_counts,
            self._candidate_quota,
        )

    def _eligible_cell(self, row: Mapping[str, Any]) -> str | None:
        location = _location_from_row(row)
        if location is None:
            return None
        _, latitude, longitude = location
        cell = self._cell_for_location(latitude, longitude)
        return cell if self._is_allowed_cell(cell) else None

    def _is_allowed_cell(self, cell: str) -> bool:
        return (self._allowed_cells is None or cell in self._allowed_cells) and (
            self._candidate_cells is None or cell in self._candidate_cells
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

    def _field_candidates(
        self,
        row: Mapping[str, Any],
        polygon_id: str,
        latitude: float,
        longitude: float,
        cell: str,
        text_field: str,
        url_field: str,
    ) -> Iterable[Candidate]:
        text = _field_text(row, text_field)
        if text is None:
            return
        website_url = _field_url(row, url_field)
        for sentence_index, sentence in enumerate(self._splitter.split(text)):
            candidate = self._candidate_for_sentence(
                sentence,
                sentence_index,
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
        if not cleaned or not self._language_identifier.is_english(cleaned):
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


def _has_website_text(row: Mapping[str, Any]) -> bool:
    return any(_field_text(row, field) is not None for field, _ in WebsiteCandidateSource._FIELD_SPECS)


def _validate_optional_limit(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_candidate_cell_settings(
    count: int | None,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
) -> None:
    if count is None:
        return
    _validate_optional_limit(count, "candidate_cell_count")
    if center_of_cell is None:
        raise ValueError("center_of_cell is required when candidate_cell_count is set")


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
    return _budget_reached(
        candidate_quota,
        candidate_counts,
        minimum_candidate_cells,
    ) or _budget_reached(row_quota, row_counts, minimum_candidate_cells)


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
) -> Iterable[Candidate]:
    for candidate in candidates:
        if candidate_quota is not None and candidate_quota.is_full(candidate.h3_cell, candidate_counts):
            continue
        candidate_counts[candidate.h3_cell] = candidate_counts.get(candidate.h3_cell, 0) + 1
        yield candidate
