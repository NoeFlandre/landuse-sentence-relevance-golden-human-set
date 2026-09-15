"""Compact V3 candidate checkpoints and the immutable V2 benchmark reader."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.storage.atomic import TextWriter, atomic_write

_BENCHMARK_COLUMNS = (
    "sentence",
    "label",
    "polygon_name",
    "h3_cell",
    "latitude",
    "longitude",
    "source",
    "region",
    "source_url",
)


@dataclass(frozen=True, slots=True)
class V2Benchmark:
    """The compact V2 annotations and cells that a V3 run must reserve."""

    annotations: tuple[Annotation, ...]
    reserved_cells: frozenset[str]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class V3CandidateProgress:
    """A resumable reservoir snapshot and the source passes it completed."""

    candidates: tuple[Candidate, ...]
    completed_sources: frozenset[Source]


class V3CandidateProgressStore:
    """Persist bounded V3 progress atomically without retaining upstream rows."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def load(self, expected_metadata: Mapping[str, Any]) -> V3CandidateProgress | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        saved_metadata = payload.get("metadata")
        if saved_metadata != dict(expected_metadata):
            raise ValueError("V3 candidate progress metadata does not match the current configuration")
        candidates = tuple(Candidate.from_dict(dict(row)) for row in payload["candidates"])
        completed_sources = frozenset(_source(value) for value in payload.get("completed_sources", ()))
        return V3CandidateProgress(candidates=candidates, completed_sources=completed_sources)

    def save(
        self,
        candidates: Iterable[Candidate],
        completed_sources: Iterable[Source],
        metadata: Mapping[str, Any],
    ) -> None:
        sources = frozenset(completed_sources)
        unknown = sources - set(Source)
        if unknown:
            names = ", ".join(sorted(source.value for source in unknown))
            raise ValueError(f"completed sources are outside the V3 profile: {names}")
        payload = {
            "metadata": dict(metadata),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "completed_sources": sorted(source.value for source in sources),
        }

        def write_payload(handle: TextWriter) -> None:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

        atomic_write(self._path, write_payload)


def load_v2_benchmark(path: Path) -> V2Benchmark:
    """Read only the committed V2 annotations needed to reserve their cells."""

    path = path.expanduser()
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    annotations: list[Annotation] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _BENCHMARK_COLUMNS:
            raise ValueError("V2 benchmark columns do not match the frozen benchmark contract")
        for index, row in enumerate(reader):
            annotations.append(_benchmark_annotation(row, index))
    return V2Benchmark(
        annotations=tuple(annotations),
        reserved_cells=frozenset(annotation.candidate.h3_cell for annotation in annotations),
        fingerprint=fingerprint,
    )


def _benchmark_annotation(row: Mapping[str, str | None], index: int) -> Annotation:
    sentence = _required_text(row, "sentence")
    source = _source(_required_text(row, "source"))
    label = _label(_required_text(row, "label"))
    cell = _required_text(row, "h3_cell")
    latitude = _required_float(row, "latitude")
    longitude = _required_float(row, "longitude")
    candidate = Candidate(
        candidate_id=f"v2:{index}:{source.value}:{cell}",
        sentence=sentence,
        source=source,
        source_record_id=f"v2:{index}",
        source_field="v2_benchmark",
        h3_cell=cell,
        h3_resolution=3,
        latitude=latitude,
        longitude=longitude,
        place_name=_optional_text(row.get("polygon_name")),
        region=_optional_text(row.get("region")),
        source_url=_optional_text(row.get("source_url")),
    )
    return Annotation(candidate=candidate, label=label)


def _required_text(row: Mapping[str, str | None], field: str) -> str:
    value = _optional_text(row.get(field))
    if value is None:
        raise ValueError(f"V2 benchmark field {field} must be non-empty")
    return value


def _required_float(row: Mapping[str, str | None], field: str) -> float:
    value = _required_text(row, field)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"V2 benchmark field {field} must be numeric") from error


def _source(value: str) -> Source:
    try:
        return Source(value)
    except ValueError as error:
        raise ValueError(f"V2 benchmark source is invalid: {value}") from error


def _label(value: str) -> Label:
    try:
        return Label(value)
    except ValueError as error:
        raise ValueError(f"V2 benchmark label is invalid: {value}") from error


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
