"""Compact V2 seed loading for the isolated V3 candidate reservoir."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from landuse_sentence_relevance.domain.constraints import V2_SOURCES
from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, SourceLabelQuotas
from landuse_sentence_relevance.domain.seeding import SeedPlan, plan_seed

_REQUIRED_COLUMNS = frozenset(
    {
        "sentence",
        "label",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    }
)


def load_v2_seed_plan(
    path: Path,
    *,
    quotas: SourceLabelQuotas = V3_QUOTAS,
    seed: str,
) -> SeedPlan:
    """Read the compact V2 benchmark and produce a deterministic V3 seed plan."""

    annotations = tuple(_read_annotations(path.expanduser()))
    return plan_seed(annotations, quotas, seed)


def _read_annotations(path: Path) -> list[Annotation]:
    annotations = _read_annotation_rows(path)
    if not annotations:
        raise ValueError(f"{path} has no rows")
    _require_unique_cells(path, annotations)
    return annotations


def _read_annotation_rows(path: Path) -> list[Annotation]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames or ())
        return [_annotation_from_row(path, index, row) for index, row in enumerate(reader)]


def _require_unique_cells(path: Path, annotations: Sequence[Annotation]) -> None:
    cells = [annotation.candidate.h3_cell for annotation in annotations]
    if len(set(cells)) != len(cells):
        raise ValueError(f"{path} contains a duplicate H3 cell")


def _require_columns(path: Path, fieldnames: Sequence[str]) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing benchmark columns: {missing}")


def _annotation_from_row(path: Path, index: int, row: Mapping[str, str | None]) -> Annotation:
    source = _required_enum(path, row, "source", Source)
    if source not in V2_SOURCES:
        raise ValueError(f"{path} row {index + 2} contains a non-V2 source: {source.value}")
    label = _required_enum(path, row, "label", Label)
    candidate = Candidate(
        candidate_id=f"v2:{index:06d}",
        sentence=_required_text(path, row, "sentence"),
        source=source,
        source_record_id=f"v2:{index:06d}",
        source_field="v2_benchmark",
        h3_cell=_required_text(path, row, "h3_cell"),
        h3_resolution=3,
        latitude=_required_float(path, row, "latitude"),
        longitude=_required_float(path, row, "longitude"),
        place_name=_optional_text(row.get("polygon_name")),
        region=_optional_text(row.get("region")),
        source_url=_optional_text(row.get("source_url")),
    )
    return Annotation(candidate=candidate, label=label)


def _required_enum(path: Path, row: Mapping[str, str | None], name: str, enum_type):
    value = row.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{path} has an invalid {name} value: {value!r}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{path} has an invalid {name} value: {value!r}") from error


def _required_text(path: Path, row: Mapping[str, str | None], name: str) -> str:
    value = _optional_text(row.get(name))
    if value is None:
        raise ValueError(f"{path} has a row missing {name}")
    return value


def _required_float(path: Path, row: Mapping[str, str | None], name: str) -> float:
    value = _required_text(path, row, name)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{path} has an invalid {name} value: {value!r}") from error


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
