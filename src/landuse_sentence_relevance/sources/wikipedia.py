from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import batched, repeat
from typing import Any

from landuse_sentence_relevance.domain.cell_quota import CellQuota
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.sentence_selection import prioritize_sentences
from landuse_sentence_relevance.domain.stratification import select_spread_cells
from landuse_sentence_relevance.observability import log_stream_progress
from landuse_sentence_relevance.sources.protocols import (
    BatchSentenceSplitter,
    SentenceSplitter,
    split_sentences_many,
)
from landuse_sentence_relevance.sources.validation import (
    validate_candidate_cell_settings as _validate_candidate_cell_settings,
)
from landuse_sentence_relevance.sources.validation import validate_optional_limit as _validate_optional_limit
from landuse_sentence_relevance.sources.validation import validate_positive_limit as _validate_positive_limit

PolygonLocation = tuple[str, float, float, str]
SectionMetadata = tuple[str, set[str], str, str | None]
RowShardsLoader = Callable[[str], Iterable[Iterable[Mapping[str, Any]]]]
PolygonBuckets = dict[str, dict[str, Mapping[str, Any]]]
_SECTION_BATCH_SIZE = 32
_MAX_STREAM_WORKERS = 4
logger = logging.getLogger(__name__)


def _rank(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


class WikipediaCandidateSource:
    """Join English Wikipedia sections to a bounded set of geolocated polygons."""

    def __init__(
        self,
        row_loader: Callable[[str], Iterable[Mapping[str, Any]]],
        splitter: SentenceSplitter,
        cell_for_location: Callable[[float, float], str],
        row_shards_loader: RowShardsLoader | None = None,
        max_polygon_rows_per_shard: int | None = None,
        max_section_rows_per_shard: int | None = None,
        max_text_characters: int | None = None,
        max_stream_workers: int = 4,
        max_polygons_per_cell: int = 8,
        max_candidates_per_cell: int = 8,
        candidate_cell_count: int | None = None,
        center_of_cell: Callable[[str], tuple[float, float]] | None = None,
        minimum_candidate_cells: int | None = None,
        excluded_cells: Iterable[str] = (),
        seed: str = "wikipedia",
    ) -> None:
        _validate_positive_limit(max_polygons_per_cell, "max_polygons_per_cell")
        _validate_positive_limit(max_candidates_per_cell, "max_candidates_per_cell")
        _validate_candidate_cell_settings(candidate_cell_count, center_of_cell)
        _validate_optional_limit(minimum_candidate_cells, "minimum_candidate_cells")
        _validate_optional_limit(max_polygon_rows_per_shard, "max_polygon_rows_per_shard")
        _validate_optional_limit(max_section_rows_per_shard, "max_section_rows_per_shard")
        _validate_optional_limit(max_text_characters, "max_text_characters")
        _validate_positive_limit(max_stream_workers, "max_stream_workers")
        self._row_loader = row_loader
        self._row_shards_loader = row_shards_loader
        self._max_polygon_rows_per_shard = max_polygon_rows_per_shard
        self._max_section_rows_per_shard = max_section_rows_per_shard
        self._max_text_characters = max_text_characters
        self._max_stream_workers = max_stream_workers
        self._splitter = splitter
        self._cell_for_location = cell_for_location
        self._max_polygons_per_cell = max_polygons_per_cell
        self._max_candidates_per_cell = max_candidates_per_cell
        self._candidate_cell_count = candidate_cell_count
        self._center_of_cell = center_of_cell
        self._minimum_candidate_cells = minimum_candidate_cells
        self._excluded_cells = frozenset(excluded_cells)
        self._seed = seed
        self._candidate_quota = CellQuota(max_candidates_per_cell)
        self._selected_cells: frozenset[str] = frozenset()
        self._candidate_cells: frozenset[str] = frozenset()

    @property
    def candidate_cells(self) -> frozenset[str]:
        return self._candidate_cells

    def _selected_polygons(self) -> dict[str, Mapping[str, Any]]:
        logger.info("Wikipedia source: selecting geolocated polygons")
        buckets, rows_seen = self._polygon_buckets()
        available_buckets = {
            cell: polygons for cell, polygons in buckets.items() if cell not in self._excluded_cells
        }
        candidate_cells = _select_candidate_cells(
            available_buckets,
            self._candidate_cell_count,
            self._center_of_cell,
            self._seed,
        )
        self._selected_cells = frozenset(candidate_cells)
        self._candidate_cells = frozenset()
        selected = {
            polygon_id: row for cell in candidate_cells for polygon_id, row in available_buckets[cell].items()
        }
        logger.info(
            "Wikipedia source: selected %d polygons across %d candidate H3 cells from %d rows",
            len(selected),
            len(candidate_cells),
            rows_seen,
        )
        return selected

    def _polygon_buckets(self) -> tuple[PolygonBuckets, int]:
        if self._row_shards_loader is None:
            return _scan_polygon_rows(
                self._row_loader("polygons"),
                self._cell_for_location,
                self._max_polygons_per_cell,
                self._seed,
                "Wikipedia polygons",
                self._max_polygon_rows_per_shard,
            )
        streams = tuple(self._row_shards_loader("polygons"))
        return _scan_polygon_shards(
            streams,
            self._cell_for_location,
            self._max_polygons_per_cell,
            self._seed,
            self._max_polygon_rows_per_shard,
            self._max_stream_workers,
        )

    def iter_candidates(self) -> Iterable[Candidate]:
        polygons = self._selected_polygons()
        document_to_polygons = self._document_links(polygons)
        logger.info(
            "Wikipedia source: splitting English article sections (max %d candidates per H3 cell)",
            self._max_candidates_per_cell,
        )
        candidates_seen = 0
        candidate_counts: dict[str, int] = {}
        candidate_cells: set[str] = set()
        scan_stats = {"sections_seen": 0}
        supports_batching = isinstance(self._splitter, BatchSentenceSplitter)
        target_cells = (
            len(self._selected_cells)
            if self._minimum_candidate_cells is None
            else self._minimum_candidate_cells
        )
        section_candidates = self._section_candidates_stream(
            polygons,
            document_to_polygons,
            candidate_counts,
            target_cells,
            scan_stats,
            supports_batching,
        )
        for candidate in section_candidates:
            candidates_seen += 1
            candidate_cells.add(candidate.h3_cell)
            self._candidate_cells = frozenset(candidate_cells)
            yield candidate
        logger.info(
            "Wikipedia source: yielded %d candidates from %d sections",
            candidates_seen,
            scan_stats["sections_seen"],
        )

    def _section_candidates_stream(
        self,
        polygons: Mapping[str, Mapping[str, Any]],
        document_to_polygons: Mapping[str, set[str]],
        candidate_counts: dict[str, int],
        target_cells: int,
        scan_stats: dict[str, int],
        supports_batching: bool,
    ) -> Iterable[Candidate]:
        if self._row_shards_loader is None:
            metadata_stream = _section_metadata_stream(
                self._row_loader("wikipedia_sections"),
                document_to_polygons,
                self._candidate_quota,
                self._selected_cells,
                candidate_counts,
                target_cells,
                scan_stats,
                self._max_text_characters,
            )
        else:
            metadata_stream = _parallel_section_metadata(
                tuple(self._row_shards_loader("wikipedia_sections")),
                document_to_polygons,
                self._max_section_rows_per_shard,
                self._max_stream_workers,
                scan_stats,
                self._max_text_characters,
            )
        metadata_stream = _one_section_per_cell(
            metadata_stream,
            polygons,
            self._selected_cells,
            self._cell_for_location,
        )
        if supports_batching:
            yield from self._batched_section_candidates(
                metadata_stream,
                polygons,
                candidate_counts,
                target_cells,
            )
            return
        yield from self._single_section_candidates(metadata_stream, polygons, candidate_counts)

    def _single_section_candidates(
        self,
        metadata_stream: Iterable[SectionMetadata],
        polygons: Mapping[str, Mapping[str, Any]],
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        for metadata in metadata_stream:
            yield from self._section_candidates_from_metadata(metadata, polygons, candidate_counts)

    def _batched_section_candidates(
        self,
        metadata_stream: Iterable[SectionMetadata],
        polygons: Mapping[str, Mapping[str, Any]],
        candidate_counts: dict[str, int],
        target_cells: int,
    ) -> Iterable[Candidate]:
        for section_batch in batched(metadata_stream, _SECTION_BATCH_SIZE):
            yield from self._section_candidate_batch(
                section_batch,
                polygons,
                candidate_counts,
                target_cells,
            )
            if self._candidate_quota.is_reached(
                self._selected_cells,
                candidate_counts,
                target_cells=target_cells,
            ):
                return

    def _document_links(self, polygons: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
        logger.info("Wikipedia source: matching English Wikipedia article links")
        if self._row_shards_loader is None:
            document_to_polygons, links_seen = _scan_link_rows(
                self._row_loader("polygon_document_links"),
                polygons,
                "Wikipedia article links",
            )
        else:
            document_to_polygons, links_seen = _scan_link_shards(
                tuple(self._row_shards_loader("polygon_document_links")),
                polygons,
                self._max_stream_workers,
            )
        logger.info(
            "Wikipedia source: matched %d article IDs from %d links",
            len(document_to_polygons),
            links_seen,
        )
        return document_to_polygons

    def _section_candidate_batch(
        self,
        sections: Iterable[SectionMetadata],
        polygons: Mapping[str, Mapping[str, Any]],
        candidate_counts: dict[str, int],
        target_cells: int,
    ) -> Iterable[Candidate]:
        section_items = tuple(sections)
        sentence_groups = split_sentences_many(
            self._splitter,
            (metadata[0] for metadata in section_items),
        )
        for metadata, sentences in zip(section_items, sentence_groups, strict=True):
            if self._candidate_quota.is_reached(
                self._selected_cells,
                candidate_counts,
                target_cells=target_cells,
            ):
                return
            yield from self._section_candidates_from_metadata(
                metadata,
                polygons,
                candidate_counts,
                sentences=sentences,
            )

    def _section_candidates_from_metadata(
        self,
        metadata: SectionMetadata,
        polygons: Mapping[str, Mapping[str, Any]],
        candidate_counts: dict[str, int],
        sentences: tuple[str, ...] | None = None,
    ) -> Iterable[Candidate]:
        text, polygon_ids, section_id, source_url = metadata
        selected_sentences = tuple(self._splitter.split(text)) if sentences is None else sentences
        for polygon_id in sorted(polygon_ids):
            yield from self._available_polygon_candidates(
                polygons[polygon_id],
                polygon_id,
                selected_sentences,
                section_id,
                source_url,
                candidate_counts,
            )

    def _available_polygon_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        sentences: tuple[str, ...],
        section_id: str,
        source_url: str | None,
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        location = _polygon_location(polygon, self._cell_for_location)
        if location is None:
            return
        cell = location[3]
        if self._candidate_quota.is_full(cell, candidate_counts):
            return
        yield from self._bounded_polygon_candidates(
            polygon,
            polygon_id,
            sentences,
            section_id,
            source_url,
            cell,
            candidate_counts,
        )

    def _bounded_polygon_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        sentences: tuple[str, ...],
        section_id: str,
        source_url: str | None,
        cell: str,
        candidate_counts: dict[str, int],
    ) -> Iterable[Candidate]:
        for candidate in self._polygon_sentence_candidates(
            polygon,
            polygon_id,
            sentences,
            section_id,
            source_url,
        ):
            if self._candidate_quota.is_full(cell, candidate_counts):
                break
            candidate_counts[cell] = candidate_counts.get(cell, 0) + 1
            yield candidate

    def _polygon_sentence_candidates(
        self,
        polygon: Mapping[str, Any],
        polygon_id: str,
        sentences: tuple[str, ...],
        section_id: str,
        source_url: str | None,
    ) -> Iterable[Candidate]:
        location = _polygon_location(polygon, self._cell_for_location)
        if location is None:
            return
        _, latitude, longitude, cell = location
        yield from _sentence_candidates(
            sentences,
            polygon,
            polygon_id,
            section_id,
            source_url,
            latitude,
            longitude,
            cell,
            seed=f"{self._seed}:{polygon_id}:{section_id}",
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


def _scan_polygon_rows(
    rows: Iterable[Mapping[str, Any]],
    cell_for_location: Callable[[float, float], str],
    max_polygons_per_cell: int,
    seed: str,
    progress_label: str,
    max_rows: int | None = None,
) -> tuple[PolygonBuckets, int]:
    buckets: PolygonBuckets = defaultdict(dict)
    rows_seen = 0
    for row in rows:
        rows_seen += 1
        log_stream_progress(logger, progress_label, rows_seen)
        location = _polygon_location(row, cell_for_location)
        if location is None:
            continue
        polygon_id, _, _, cell = location
        _insert_polygon(
            buckets[cell],
            polygon_id,
            _polygon_metadata(row),
            max_polygons_per_cell,
            seed,
        )
        if max_rows is not None and rows_seen >= max_rows:
            break
    return buckets, rows_seen


def _scan_polygon_shards(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    cell_for_location: Callable[[float, float], str],
    max_polygons_per_cell: int,
    seed: str,
    max_rows_per_shard: int | None,
    max_workers: int,
) -> tuple[PolygonBuckets, int]:
    if len(streams) == 0:
        return defaultdict(dict), 0
    if len(streams) == 1:
        return _scan_polygon_rows(
            streams[0],
            cell_for_location,
            max_polygons_per_cell,
            seed,
            "Wikipedia polygons",
            max_rows_per_shard,
        )
    with ThreadPoolExecutor(
        max_workers=min(len(streams), max_workers),
        thread_name_prefix="wikipedia-stream",
    ) as executor:
        results = executor.map(
            _scan_polygon_rows,
            streams,
            repeat(cell_for_location),
            repeat(max_polygons_per_cell),
            repeat(seed),
            (f"Wikipedia polygons shard {index + 1}" for index in range(len(streams))),
            repeat(max_rows_per_shard),
        )
        return _merge_polygon_results(results, max_polygons_per_cell, seed)


def _merge_polygon_results(
    results: Iterable[tuple[PolygonBuckets, int]],
    max_polygons_per_cell: int,
    seed: str,
) -> tuple[PolygonBuckets, int]:
    buckets: PolygonBuckets = defaultdict(dict)
    rows_seen = 0
    for shard_buckets, shard_rows in results:
        rows_seen += shard_rows
        for cell, polygons in shard_buckets.items():
            for polygon_id, metadata in polygons.items():
                _insert_polygon(
                    buckets[cell],
                    polygon_id,
                    metadata,
                    max_polygons_per_cell,
                    seed,
                )
    return buckets, rows_seen


def _scan_link_rows(
    rows: Iterable[Mapping[str, Any]],
    polygons: Mapping[str, Mapping[str, Any]],
    progress_label: str,
) -> tuple[dict[str, set[str]], int]:
    document_to_polygons: dict[str, set[str]] = defaultdict(set)
    links_seen = 0
    for link in rows:
        links_seen += 1
        log_stream_progress(logger, progress_label, links_seen)
        if not _is_english_wikipedia_link(link):
            continue
        link_ids = _link_ids(link, polygons)
        if link_ids is None:
            continue
        document_id, polygon_id = link_ids
        document_to_polygons[document_id].add(polygon_id)
    return document_to_polygons, links_seen


def _scan_link_shards(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    polygons: Mapping[str, Mapping[str, Any]],
    max_workers: int,
) -> tuple[dict[str, set[str]], int]:
    if len(streams) == 0:
        return defaultdict(set), 0
    if len(streams) == 1:
        return _scan_link_rows(streams[0], polygons, "Wikipedia article links")
    return _scan_link_streams_in_parallel(streams, polygons, max_workers)


def _scan_link_streams_in_parallel(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    polygons: Mapping[str, Mapping[str, Any]],
    max_workers: int,
) -> tuple[dict[str, set[str]], int]:
    with ThreadPoolExecutor(
        max_workers=min(len(streams), max_workers),
        thread_name_prefix="wikipedia-links",
    ) as executor:
        results = executor.map(
            _scan_link_rows,
            streams,
            repeat(polygons),
            (f"Wikipedia article links shard {index + 1}" for index in range(len(streams))),
        )
        return _merge_link_results(results)


def _merge_link_results(
    results: Iterable[tuple[dict[str, set[str]], int]],
) -> tuple[dict[str, set[str]], int]:
    document_to_polygons: dict[str, set[str]] = defaultdict(set)
    links_seen = 0
    for shard_links, shard_rows in results:
        links_seen += shard_rows
        for document_id, polygon_ids in shard_links.items():
            document_to_polygons[document_id].update(polygon_ids)
    return document_to_polygons, links_seen


def _parallel_section_metadata(
    streams: tuple[Iterable[Mapping[str, Any]], ...],
    document_to_polygons: Mapping[str, set[str]],
    max_rows_per_shard: int | None,
    max_workers: int,
    scan_stats: dict[str, int],
    max_text_characters: int | None,
) -> tuple[SectionMetadata, ...]:
    if not streams:
        return ()
    if len(streams) == 1:
        results = (
            _scan_section_rows(
                streams[0],
                document_to_polygons,
                max_rows_per_shard,
                "Wikipedia sections shard 1",
                max_text_characters,
            ),
        )
    else:
        with ThreadPoolExecutor(
            max_workers=min(len(streams), max_workers),
            thread_name_prefix="wikipedia-sections",
        ) as executor:
            results = executor.map(
                _scan_section_rows,
                streams,
                repeat(document_to_polygons),
                repeat(max_rows_per_shard),
                (f"Wikipedia sections shard {index + 1}" for index in range(len(streams))),
                repeat(max_text_characters),
            )
    metadata, rows_seen = _merge_section_results(results)
    scan_stats["sections_seen"] += rows_seen
    return metadata


def _scan_section_rows(
    rows: Iterable[Mapping[str, Any]],
    document_to_polygons: Mapping[str, set[str]],
    max_rows: int | None,
    progress_label: str,
    max_text_characters: int | None,
) -> tuple[tuple[SectionMetadata, ...], int]:
    metadata: list[SectionMetadata] = []
    rows_seen = 0
    for section in rows:
        rows_seen += 1
        log_stream_progress(logger, progress_label, rows_seen)
        section_metadata = _section_metadata(section, document_to_polygons, max_text_characters)
        if section_metadata is not None:
            metadata.append(section_metadata)
        if max_rows is not None and rows_seen >= max_rows:
            break
    return tuple(metadata), rows_seen


def _merge_section_results(
    results: Iterable[tuple[tuple[SectionMetadata, ...], int]],
) -> tuple[tuple[SectionMetadata, ...], int]:
    metadata: list[SectionMetadata] = []
    rows_seen = 0
    for shard_metadata, shard_rows in results:
        metadata.extend(shard_metadata)
        rows_seen += shard_rows
    return tuple(metadata), rows_seen


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
    max_text_characters: int | None = None,
) -> SectionMetadata | None:
    if section.get("language") != "en":
        return None
    document_id = _section_document_id(section, document_to_polygons)
    if document_id is None:
        return None
    text = _section_text(section, max_text_characters)
    if text is None:
        return None
    section_id = str(section.get("section_id") or f"{document_id}:{section.get('section_index', 0)}")
    source_url = _section_url(section)
    return text, document_to_polygons[str(document_id)], section_id, source_url


def _section_metadata_stream(
    sections: Iterable[Mapping[str, Any]],
    document_to_polygons: Mapping[str, set[str]],
    candidate_quota: CellQuota,
    selected_cells: frozenset[str],
    candidate_counts: Mapping[str, int],
    target_cells: int,
    scan_stats: dict[str, int],
    max_text_characters: int | None,
) -> Iterable[SectionMetadata]:
    for section in sections:
        if candidate_quota.is_reached(selected_cells, candidate_counts, target_cells=target_cells):
            logger.info("Wikipedia source: candidate cell budget filled; stopping section scan")
            return
        scan_stats["sections_seen"] += 1
        log_stream_progress(logger, "Wikipedia sections", scan_stats["sections_seen"])
        metadata = _section_metadata(section, document_to_polygons, max_text_characters)
        if metadata is not None:
            yield metadata


def _section_document_id(
    section: Mapping[str, Any],
    document_to_polygons: Mapping[str, set[str]],
) -> str | None:
    document_id = section.get("document_id")
    return str(document_id) if document_id is not None and str(document_id) in document_to_polygons else None


def _section_text(section: Mapping[str, Any], max_text_characters: int | None = None) -> str | None:
    text = section.get("text")
    if not isinstance(text, str):
        return None
    return text if max_text_characters is None else text[:max_text_characters]


def _section_url(section: Mapping[str, Any]) -> str | None:
    page_id = section.get("page_id")
    return f"https://en.wikipedia.org/?curid={page_id}" if page_id is not None else None


def _one_section_per_cell(
    metadata_stream: Iterable[SectionMetadata],
    polygons: Mapping[str, Mapping[str, Any]],
    selected_cells: frozenset[str],
    cell_for_location: Callable[[float, float], str],
) -> Iterable[SectionMetadata]:
    seen_cells: set[str] = set()
    for metadata in metadata_stream:
        section_cells = _section_cells(metadata, polygons, selected_cells, cell_for_location)
        new_cells = section_cells - seen_cells
        if not new_cells:
            continue
        seen_cells.update(new_cells)
        yield metadata


def _section_cells(
    metadata: SectionMetadata,
    polygons: Mapping[str, Mapping[str, Any]],
    selected_cells: frozenset[str],
    cell_for_location: Callable[[float, float], str],
) -> set[str]:
    cells: set[str] = set()
    for polygon_id in metadata[1]:
        polygon = polygons.get(polygon_id)
        if polygon is None:
            continue
        location = _polygon_location(polygon, cell_for_location)
        if location is not None and location[3] in selected_cells:
            cells.add(location[3])
    return cells


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
    sentences: tuple[str, ...],
    polygon: Mapping[str, Any],
    polygon_id: str,
    section_id: str,
    source_url: str | None,
    latitude: float,
    longitude: float,
    cell: str,
    seed: str,
) -> Iterable[Candidate]:
    for part in prioritize_sentences(sentences, seed=seed):
        yield Candidate(
            candidate_id=f"wikipedia:{polygon_id}:{section_id}:{part.index}",
            sentence=part.text,
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
