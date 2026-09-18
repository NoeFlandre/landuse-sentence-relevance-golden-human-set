"""Text and sentence-selection helpers for the website source."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import batched
from typing import Any

from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.domain.sentence_selection import (
    SentencePart,
    candidate_sentence_parts,
)
from landuse_sentence_relevance.sources.protocols import BatchLanguageIdentifier

#: The text-shaping vocabulary the website sources share.
#:
#: `website.py` and `website_candidates.py` both read rows through these, so they are the
#: module's contract whatever they were called: an underscore on a name two other modules import
#: tells a reader it is safe to change when it is not.
__all__ = [
    "WEBSITE_FIELD_SPECS",
    "FieldSpec",
    "FieldWork",
    "Location",
    "bounded_text",
    "candidate_part_groups",
    "field_text",
    "field_url",
    "is_contextual_text",
    "location_from_row",
    "pending_field_indexes",
    "select_batch_parts",
]

Location = tuple[str, float, float]
FieldSpec = tuple[str, str]
WEBSITE_FIELD_SPECS: tuple[FieldSpec, ...] = (
    ("website_text", "website"),
    ("contact_website_text", "contact_website"),
)
_LANGUAGE_SELECTION_BATCH_SIZE = 8
_CONTEXTUAL_BOUNDARY_PATTERN = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")
_MIN_CONTEXTUAL_BOUNDARIES = 2


@dataclass(frozen=True, slots=True)
class FieldWork:
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


def location_from_row(row: Mapping[str, Any]) -> Location | None:
    polygon_id = row.get("polygon_id") or row.get("osm_id")
    latitude = row.get("lat")
    longitude = row.get("lon")
    if polygon_id is None or latitude is None or longitude is None:
        return None
    return str(polygon_id), float(latitude), float(longitude)


def field_text(row: Mapping[str, Any], field: str) -> str | None:
    text = row.get(field)
    return text.strip() if isinstance(text, str) and text.strip() else None


def field_url(row: Mapping[str, Any], field: str) -> str | None:
    url = row.get(field)
    return url if isinstance(url, str) else None


def candidate_part_groups(
    work: tuple[FieldWork, ...],
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


def select_batch_parts(
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
        first_batch = tuple(next(batched(parts, _LANGUAGE_SELECTION_BATCH_SIZE), ()))
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


def _first_accepted_part(
    parts: tuple[SentencePart, ...],
    accepted: tuple[bool, ...],
) -> SentencePart | None:
    return next(
        (part for part, is_accepted in zip(parts, accepted, strict=True) if is_accepted),
        None,
    )


def bounded_text(text: str, max_characters: int | None) -> str:
    return text if max_characters is None else text[:max_characters]


def pending_field_indexes(
    pending_indexes: tuple[int, ...],
    candidates: list[list[Candidate]],
) -> tuple[int, ...]:
    return tuple(index for index in pending_indexes if not candidates[index])


def is_contextual_text(text: str) -> bool:
    return len(_CONTEXTUAL_BOUNDARY_PATTERN.findall(text)) >= _MIN_CONTEXTUAL_BOUNDARIES
