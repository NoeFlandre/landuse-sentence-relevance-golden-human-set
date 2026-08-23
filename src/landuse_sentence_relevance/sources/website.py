from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.sources.protocols import LanguageIdentifier, SentenceSplitter

Location = tuple[str, float, float]
FieldSpec = tuple[str, str]


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
    ) -> None:
        self._row_loader = row_loader
        self._splitter = splitter
        self._language_identifier = language_identifier
        self._cell_for_location = cell_for_location

    def iter_candidates(self) -> Iterable[Candidate]:
        for row in self._row_loader():
            location = _location_from_row(row)
            if location is None:
                continue
            polygon_id, latitude, longitude = location
            cell = self._cell_for_location(latitude, longitude)
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
