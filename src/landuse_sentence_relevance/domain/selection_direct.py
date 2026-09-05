from __future__ import annotations

from collections.abc import Mapping
from itertools import chain

from landuse_sentence_relevance.domain.constraints import DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label, Source

Category = tuple[Source, Label]


def select_single_row_cells(
    grouped: Mapping[str, list[Annotation]],
    quotas: DatasetQuotas,
) -> tuple[Annotation, ...] | None:
    """Select a balanced subset without expanding the general state space."""
    rows_by_category = _rows_by_category(grouped)
    diagonal = _feasible_diagonal(rows_by_category, quotas)
    if diagonal is None:
        return None
    return _select_rows(rows_by_category, diagonal, quotas)


def _rows_by_category(grouped: Mapping[str, list[Annotation]]) -> dict[Category, tuple[Annotation, ...]]:
    rows: dict[Category, list[Annotation]] = {(source, label): [] for source in Source for label in Label}
    for row in chain.from_iterable(grouped.values()):
        rows[(row.candidate.source, row.label)].append(row)
    return {
        category: tuple(sorted(category_rows, key=lambda row: row.candidate.candidate_id))
        for category, category_rows in rows.items()
    }


def _feasible_diagonal(
    rows_by_category: Mapping[Category, tuple[Annotation, ...]],
    quotas: DatasetQuotas,
) -> int | None:
    # DatasetQuotas guarantees equal source and label quotas.
    for diagonal in range(quotas.per_source + 1):
        required = _required_counts(diagonal, quotas)
        if all(len(rows_by_category[category]) >= count for category, count in required.items()):
            return diagonal
    return None


def _required_counts(diagonal: int, quotas: DatasetQuotas) -> dict[Category, int]:
    return {
        (Source.WIKIPEDIA, Label.YES): diagonal,
        (Source.WIKIPEDIA, Label.NO): quotas.per_source - diagonal,
        (Source.WEBSITE, Label.YES): quotas.per_label - diagonal,
        (Source.WEBSITE, Label.NO): quotas.per_source - quotas.per_label + diagonal,
    }


def _select_rows(
    rows_by_category: Mapping[Category, tuple[Annotation, ...]],
    diagonal: int,
    quotas: DatasetQuotas,
) -> tuple[Annotation, ...]:
    required = _required_counts(diagonal, quotas)
    selected = tuple(
        row for category, count in required.items() for row in rows_by_category[category][:count]
    )
    return tuple(sorted(selected, key=lambda row: row.candidate.candidate_id))
