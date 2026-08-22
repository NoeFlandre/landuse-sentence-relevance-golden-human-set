from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from landuse_sentence_relevance.domain.models import Annotation, Label, Source


class FinalDatasetNotReadyError(ValueError):
    """Raised when the annotated records cannot satisfy the final dataset contract."""


@dataclass(frozen=True, slots=True)
class DatasetQuotas:
    total: int = 100
    per_source: int = 50
    per_label: int = 50
    cell_count: int = 25
    per_source_cell: int = 2
    h3_resolution: int = 3

    def __post_init__(self) -> None:
        if self.total != self.per_source * len(Source):
            raise ValueError("total must equal the per-source quota for every source")
        if self.total != self.per_label * len(Label):
            raise ValueError("total must equal the per-label quota for every label")
        if self.total != self.cell_count * self.per_source_cell * len(Source):
            raise ValueError("total must equal the cell and source quotas")
        if self.h3_resolution != 3:
            raise ValueError("h3_resolution must be 3")


DEFAULT_QUOTAS = DatasetQuotas()


def validate_final_dataset(
    annotations: Iterable[Annotation],
    quotas: DatasetQuotas = DEFAULT_QUOTAS,
) -> None:
    rows = tuple(annotations)
    _require_row_count(rows, quotas)
    _require_unique_ids(rows)
    _require_source_counts(rows, quotas)
    _require_label_counts(rows, quotas)
    cells = _require_cell_count(rows, quotas)
    _require_cell_source_counts(rows, cells, quotas)
    _require_h3_resolution(rows, quotas)


def _require_row_count(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> None:
    if len(rows) != quotas.total:
        raise FinalDatasetNotReadyError(f"expected {quotas.total} rows, received {len(rows)}")


def _require_unique_ids(rows: tuple[Annotation, ...]) -> None:
    candidate_ids = [row.candidate.candidate_id for row in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise FinalDatasetNotReadyError("candidate IDs must be unique")


def _require_source_counts(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> None:
    counts = Counter(row.candidate.source for row in rows)
    if any(counts[source] != quotas.per_source for source in Source):
        raise FinalDatasetNotReadyError("source quotas are not satisfied")


def _require_label_counts(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> None:
    counts = Counter(row.label for row in rows)
    if any(counts[label] != quotas.per_label for label in Label):
        raise FinalDatasetNotReadyError("label quotas are not satisfied")


def _require_cell_count(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> set[str]:
    cells = {row.candidate.h3_cell for row in rows}
    if len(cells) != quotas.cell_count:
        raise FinalDatasetNotReadyError("the final dataset must cover the configured H3 cells")
    return cells


def _require_cell_source_counts(
    rows: tuple[Annotation, ...],
    cells: set[str],
    quotas: DatasetQuotas,
) -> None:
    for cell in cells:
        if not all(_cell_source_count(rows, cell, source) == quotas.per_source_cell for source in Source):
            raise FinalDatasetNotReadyError("each H3 cell must contain the same source quota")


def _cell_source_count(rows: tuple[Annotation, ...], cell: str, source: Source) -> int:
    return sum(row.candidate.h3_cell == cell and row.candidate.source is source for row in rows)


def _require_h3_resolution(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> None:
    if any(row.candidate.h3_resolution != quotas.h3_resolution for row in rows):
        raise FinalDatasetNotReadyError("all rows must use the configured H3 resolution")
