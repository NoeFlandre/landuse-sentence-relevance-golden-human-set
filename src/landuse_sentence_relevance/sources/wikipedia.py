from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.stratification import select_spread_cells
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.protocols import SentenceSplitter

PolygonLocation = tuple[str, float, float, str]
logger = logging.getLogger(__name__)


def _rank(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def _validate_positive_limit(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_optional_limit(value: int | None, name: str) -> None:
    if value is not None:
        _validate_positive_limit(value, name)


def _validate_candidate_cell_settings(
    count: int | None,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
) -> None:
    if count is None:
        return
    _validate_positive_limit(count, "candidate_cell_count")
    if center_of_cell is None:
        raise ValueError("center_of_cell is required when candidate_cell_count is set")


class WikipediaCandidateSource:
    """Join English Wikipedia sections to a bounded set of geolocated polygons."""

    def __init__(
        self,
        row_loader: Callable[[str], Iterable[Mapping[str, Any]]],
        splitter: SentenceSplitter,
        cell_for_location: Callable[[float, float], str],
        max_polygons_per_cell: int = 8,
        max_candidates_per_cell: int = 8,
        candidate_cell_count: int | None = None,
        center_of_cell: Callable[[str], tuple[float, float]] | None = None,
        minimum_candidate_cells: int | None = None,
        seed: str = "wikipedia",
    ) -> None:
        _validate_positive_limit(max_polygons_per_cell, "max_polygons_per_cell")
        _validate_positive_limit(max_candidates_per_cell, "max_candidates_per_cell")
        _validate_candidate_cell_settings(candidate_cell_count, center_of_cell)
        _validate_optional_limit(minimum_candidate_cells, "minimum_candidate_cells")
        self._row_loader = row_loader
        self._splitter = splitter
        self._cell_for_location = cell_for_location
        self._max_polygons_per_cell = max_polygons_per_cell
        self._max_candidates_per_cell = max_candidates_per_cell
        self._candidate_cell_count = candidate_cell_count
        self._center_of_cell = center_of_cell
        self._minimum_candidate_cells = minimum_candidate_cells
        self._seed = seed
        self._selected_cells: frozenset[str] = frozenset()
        self._candidate_cells: frozenset[str] = frozenset()

    @property
    def candidate_cells(self) -> frozenset[str]:
        return self._candidate_cells

    def _selected_polygons(self) -> dict[str, Mapping[str, Any]]:
        logger.info("Wikipedia source: selecting geolocated polygons")
        buckets: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        rows_seen = 0
        for row in self._row_loader("polygons"):
            rows_seen += 1
            log_stream_progress(logger, "Wikipedia polygons", rows_seen)
            location = _polygon_location(row, self._cell_for_location)
            if location is None:
                continue
            polygon_id, _, _, cell = location
            _insert_polygon(
                buckets[cell],
                polygon_id,
                _polygon_metadata(row),
                self._max_polygons_per_cell,
                self._seed,
            )
        candidate_cells = _select_candidate_cells(
            buckets,
            self._candidate_cell_count,
            self._center_of_cell,
            self._seed,
        )
        self._selected_cells = frozenset(candidate_cells)
        self._candidate_cells = frozenset()
        selected = {polygon_id: row for cell in candidate_cells for polygon_id, row in buckets[cell].items()}
        logger.info(
            "Wikipedia source: selected %d polygons across %d candidate H3 cells from %d rows",
            len(selected),
            len(candidate_cells),
            rows_seen,
        )
        return selected

    def iter_candidates(self) -> Iterable[Candidate]:
        polygons = self._selected_polygons()
        document_to_polygons = self._document_links(polygons)
        logger.info(
            "Wikipedia source: splitting English article sections (max %d candidates per H3 cell)",
            self._max_candidates_per_cell,
        )
        sections_seen = 0
        candidates_seen = 0
        candidate_counts: dict[str, int] = {}
        candidate_cells: set[str] = set()
        for section in self._row_loader("wikipedia_sections"):
            if _candidate_budget_filled(
                self._selected_cells,
                candidate_counts,
                self._max_candidates_per_cell,
                self._minimum_candidate_cells,
            ):
                logger.info("Wikipedia source: candidate cell budget filled; stopping section scan")
                break
            sections_seen += 1
            log_stream_progress(logger, "Wikipedia sections", sections_seen)
            for candidate in self._section_candidates(
                section,
                polygons,
                document_to_polygons,
                candidate_counts,
            ):
                candidates_seen += 1
                candidate_cells.add(candidate.h3_cell)
                self._candidate_cells = frozenset(candidate_cells)
                yield candidate
        logger.info(
            "Wikipedia source: yielded %d candidates from %d sections",
            candidates_seen,
            sections_seen,
        )

    def _document_links(self, polygons: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
        logger.info("Wikipedia source: matching English Wikipedia article links")
        document_to_polygons: dict[str, set[str]] = defaultdict(set)
        links_seen = 0
        for link in self._row_loader("polygon_document_links"):
            links_seen += 1
            log_stream_progress(logger, "Wikipedia article links", links_seen)
            if not _is_english_wikipedia_link(link):
                continue
            link_ids = _link_ids(link, polygons)
            if link_ids is None:
                continue
            document_id, polygon_id = link_ids
            document_to_polygons[document_id].add(polygon_id)
        logger.info(
            "Wikipedia source: matched %d article IDs from %d links",
            len(document_to_polygons),
            links_seen,
        )
        return document_to_polygons

    def _section_candidates(
        self,
        section: Mapping[str, Any],
        polygons: Mapping[str, Mapping[str, Any]],
        document_to_polygons: Mapping[str, set[str]],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        metadata = _section_metadata(section, document_to_polygons)
        if metadata is None:
            return
        text, polygon_ids, section_id, source_url = metadata
        for polygon_id in sorted(polygon_ids):
            yield from self._available_polygon_candidates(
                polygons[polygon_id],
                polygon_id,
                text,
                section_id,
                source_url,
                candidate_counts,
            )

    def _available_polygon_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        text: str,
        section_id: str,
        source_url: str | None,
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        location = _polygon_location(polygon, self._cell_for_location)
        if location is None:
            return
        cell = location[3]
        if _cell_is_full(cell, candidate_counts, self._max_candidates_per_cell):
            return
        yield from self._bounded_polygon_candidates(
            polygon,
            polygon_id,
            text,
            section_id,
            source_url,
            cell,
            candidate_counts,
        )

    def _bounded_polygon_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        text: str,
        section_id: str,
        source_url: str | None,
        cell: str,
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        for candidate in self._polygon_sentence_candidates(
            polygon,
            polygon_id,
            text,
            section_id,
            source_url,
        ):
            if _cell_is_full(cell, candidate_counts, self._max_candidates_per_cell):
                break
            candidate_counts[cell] = candidate_counts.get(cell, 0) + 1
            yield candidate

    def _polygon_sentence_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        text: str,
        section_id: str,
        source_url: str | None,
    ) -> Iterable[Candidate]:
        location = _polygon_location(polygon, self._cell_for_location)
        if location is None:
            return
        _, latitude, longitude, cell = location
        yield from _sentence_candidates(
            self._splitter,
            text,
            polygon,
            polygon_id,
            section_id,
            source_url,
            latitude,
            longitude,
            cell,
        )


def _polygon_location(
    row: Mapping[str, Any],
    cell_for_location: Callable[[float, float], str],
) -> PolygonLocation | None:
    if not row.get("has_english_wikipedia"):
        return None
    coordinates = _coordinates(row)
    if coordinates is None:
        return None
    polygon_id, latitude, longitude = coordinates
    return str(polygon_id), latitude, longitude, cell_for_location(latitude, longitude)


def _coordinates(row: Mapping[str, Any]) -> tuple[Any, float, float] | None:
    polygon_id = row.get("polygon_id")
    latitude = row.get("lat")
    longitude = row.get("lon")
    if polygon_id is None or latitude is None or longitude is None:
        return None
    return polygon_id, float(latitude), float(longitude)


def _polygon_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "polygon_id": row.get("polygon_id"),
        "has_english_wikipedia": row.get("has_english_wikipedia"),
        "lat": row.get("lat"),
        "lon": row.get("lon"),
        "name": row.get("name"),
        "region": row.get("region"),
    }


def _is_english_wikipedia_link(link: Mapping[str, Any]) -> bool:
    return link.get("project") == "wikipedia" and link.get("language") == "en"


def _link_ids(link: Mapping[str, Any], polygons: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    polygon_id = link.get("polygon_id")
    document_id = link.get("document_id")
    if polygon_id is None or document_id is None or str(polygon_id) not in polygons:
        return None
    return str(document_id), str(polygon_id)


def _section_metadata(
    section: Mapping[str, Any],
    document_to_polygons: Mapping[str, set[str]],
) -> tuple[str, set[str], str, str | None] | None:
    if section.get("language") != "en":
        return None
    document_id = _section_document_id(section, document_to_polygons)
    if document_id is None:
        return None
    text = _section_text(section)
    if text is None:
        return None
    section_id = str(section.get("section_id") or f"{document_id}:{section.get('section_index', 0)}")
    source_url = _section_url(section)
    return text, document_to_polygons[str(document_id)], section_id, source_url


def _section_document_id(
    section: Mapping[str, Any],
    document_to_polygons: Mapping[str, set[str]],
) -> str | None:
    document_id = section.get("document_id")
    return str(document_id) if document_id is not None and str(document_id) in document_to_polygons else None


def _section_text(section: Mapping[str, Any]) -> str | None:
    text = section.get("text")
    return text if isinstance(text, str) else None


def _section_url(section: Mapping[str, Any]) -> str | None:
    page_id = section.get("page_id")
    return f"https://en.wikipedia.org/?curid={page_id}" if page_id is not None else None


def _cell_is_full(cell: str, counts: Mapping[str, int], capacity: int) -> bool:
    return counts.get(cell, 0) >= capacity


def _all_cells_full(cells: Iterable[str], counts: Mapping[str, int], capacity: int) -> bool:
    return all(_cell_is_full(cell, counts, capacity) for cell in cells)


def _candidate_budget_filled(
    cells: Iterable[str],
    counts: Mapping[str, int],
    capacity: int,
    minimum_cells: int | None,
) -> bool:
    if minimum_cells is None:
        return _all_cells_full(cells, counts, capacity)
    return _filled_cell_count(cells, counts, capacity) >= minimum_cells


def _filled_cell_count(cells: Iterable[str], counts: Mapping[str, int], capacity: int) -> int:
    return sum(_cell_is_full(cell, counts, capacity) for cell in cells)


def _select_candidate_cells(
    buckets: Mapping[str, Mapping[str, Any]],
    target_count: int | None,
    center_of_cell: Callable[[str], tuple[float, float]] | None,
    seed: str,
) -> tuple[str, ...]:
    cells = tuple(sorted(buckets))
    if target_count is None or len(cells) <= target_count:
        return cells
    if center_of_cell is None:
        raise ValueError("center_of_cell is required when candidate_cell_count is set")
    return select_spread_cells(
        cells,
        target_count=target_count,
        center_of_cell=center_of_cell,
        seed=seed,
    )


def _sentence_candidates(
    splitter: SentenceSplitter,
    text: str,
    polygon: Mapping[str, Any],
    polygon_id: str,
    section_id: str,
    source_url: str | None,
    latitude: float,
    longitude: float,
    cell: str,
) -> Iterable[Candidate]:
    for sentence_index, sentence in enumerate(splitter.split(text)):
        cleaned = sentence.strip()
        if cleaned:
            yield Candidate(
                candidate_id=f"wikipedia:{polygon_id}:{section_id}:{sentence_index}",
                sentence=cleaned,
                source=Source.WIKIPEDIA,
                source_record_id=section_id,
                source_field="wikipedia_section",
                h3_cell=cell,
                h3_resolution=3,
                latitude=latitude,
                longitude=longitude,
                place_name=polygon.get("name"),
                region=polygon.get("region"),
                source_url=source_url,
            )


def _insert_polygon(
    bucket: dict[str, Mapping[str, Any]],
    polygon_id: str,
    row: Mapping[str, Any],
    capacity: int,
    seed: str,
) -> None:
    bucket[polygon_id] = row
    if len(bucket) > capacity:
        worst_id = max(bucket, key=lambda key: (_rank(seed, key), key))
        del bucket[worst_id]
