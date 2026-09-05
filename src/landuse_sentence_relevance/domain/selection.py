from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from landuse_sentence_relevance.domain.constraints import DEFAULT_QUOTAS, DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.selection_direct import select_single_row_cells

SelectionState = tuple[int, int, int]


def select_final_annotations(
    annotations: Iterable[Annotation],
    quotas: DatasetQuotas = DEFAULT_QUOTAS,
) -> tuple[Annotation, ...] | None:
    """Select one row per distinct cell with balanced source and label quotas."""
    rows = tuple(annotations)
    if not _has_unique_ids(rows):
        return None
    grouped = _group_by_cell(rows)
    if len(grouped) < quotas.cell_count:
        return None
    if _can_use_single_row_path(grouped, quotas):
        return select_single_row_cells(grouped, quotas)
    return _select_with_states(grouped, quotas)


def _can_use_single_row_path(
    grouped: dict[str, list[Annotation]],
    quotas: DatasetQuotas,
) -> bool:
    return quotas.rows_per_cell == 1 and all(len(cell_rows) == 1 for cell_rows in grouped.values())


def _select_with_states(
    grouped: dict[str, list[Annotation]],
    quotas: DatasetQuotas,
) -> tuple[Annotation, ...] | None:
    states: dict[SelectionState, tuple[Annotation, ...]] = {(0, 0, 0): ()}
    for cell in sorted(grouped):
        states = _advance_states(states, tuple(grouped[cell]), quotas)
    selected = states.get((quotas.total, quotas.per_source, quotas.per_label))
    if selected is None:
        return None
    return tuple(sorted(selected, key=lambda row: row.candidate.candidate_id))


def _has_unique_ids(rows: tuple[Annotation, ...]) -> bool:
    return len({row.candidate.candidate_id for row in rows}) == len(rows)


def _group_by_cell(rows: tuple[Annotation, ...]) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = defaultdict(list)
    for row in rows:
        grouped[row.candidate.h3_cell].append(row)
    return grouped


def _advance_states(
    states: dict[SelectionState, tuple[Annotation, ...]],
    cell_rows: tuple[Annotation, ...],
    quotas: DatasetQuotas,
) -> dict[SelectionState, tuple[Annotation, ...]]:
    next_states = dict(states)
    for state, selected in states.items():
        selected_count, wikipedia_count, yes_count = state
        for row in sorted(cell_rows, key=lambda item: item.candidate.candidate_id):
            next_state = (
                selected_count + 1,
                wikipedia_count + (row.candidate.source is Source.WIKIPEDIA),
                yes_count + (row.label is Label.YES),
            )
            if _exceeds_quotas(next_state, quotas):
                continue
            next_states.setdefault(next_state, (*selected, row))
    return next_states


def _exceeds_quotas(state: SelectionState, quotas: DatasetQuotas) -> bool:
    selected_count, wikipedia_count, yes_count = state
    return (
        selected_count > quotas.total or wikipedia_count > quotas.per_source or yes_count > quotas.per_label
    )
