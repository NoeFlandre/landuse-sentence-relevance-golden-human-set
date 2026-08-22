from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.sources.protocols import SentenceSplitter

PolygonLocation = tuple[str, float, float, str]


def _rank(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


class WikipediaCandidateSource:
    """Join English Wikipedia sections to a bounded set of geolocated polygons."""

    def __init__(
        self,
        row_loader: Callable[[str], Iterable[Mapping[str, Any]]],
        splitter: SentenceSplitter,
        cell_for_location: Callable[[float, float], str],
        max_polygons_per_cell: int = 8,
        seed: str = "wikipedia",
    ) -> None:
        if max_polygons_per_cell < 1:
            raise ValueError("max_polygons_per_cell must be positive")
        self._row_loader = row_loader
        self._splitter = splitter
        self._cell_for_location = cell_for_location
        self._max_polygons_per_cell = max_polygons_per_cell
        self._seed = seed

    def _selected_polygons(self) -> dict[str, Mapping[str, Any]]:
        buckets: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in self._row_loader("polygons"):
            location = _polygon_location(row, self._cell_for_location)
            if location is None:
                continue
            polygon_id, _, _, cell = location
            _insert_polygon(buckets[cell], polygon_id, row, self._max_polygons_per_cell, self._seed)
        return {polygon_id: row for bucket in buckets.values() for polygon_id, row in bucket.items()}

    def iter_candidates(self) -> Iterable[Candidate]:
        polygons = self._selected_polygons()
        document_to_polygons = self._document_links(polygons)
        for section in self._row_loader("wikipedia_sections"):
            yield from self._section_candidates(section, polygons, document_to_polygons)

    def _document_links(self, polygons: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
        document_to_polygons: dict[str, set[str]] = defaultdict(set)
        for link in self._row_loader("polygon_document_links"):
            if not _is_english_wikipedia_link(link):
                continue
            link_ids = _link_ids(link, polygons)
            if link_ids is None:
                continue
            document_id, polygon_id = link_ids
            document_to_polygons[document_id].add(polygon_id)
        return document_to_polygons

    def _section_candidates(
        self,
        section: Mapping[str, Any],
        polygons: Mapping[str, Mapping[str, Any]],
        document_to_polygons: Mapping[str, set[str]],
    ) -> Iterable[Candidate]:
        metadata = _section_metadata(section, document_to_polygons)
        if metadata is None:
            return
        text, polygon_ids, section_id, source_url = metadata
        for polygon_id in sorted(polygon_ids):
            polygon = polygons[polygon_id]
            yield from self._polygon_sentence_candidates(
                polygon,
                polygon_id,
                text,
                section_id,
                source_url,
            )

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
