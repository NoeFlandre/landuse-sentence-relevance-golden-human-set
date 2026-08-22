from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from landuse_sentence_relevance.domain.constraints import DEFAULT_QUOTAS, DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label, Source

SelectionOption = tuple[tuple[Annotation, ...], tuple[Annotation, ...]]


def select_final_annotations(
    annotations: Iterable[Annotation],
    quotas: DatasetQuotas = DEFAULT_QUOTAS,
) -> tuple[Annotation, ...] | None:
    """Select a deterministic quota-satisfying subset, or return ``None`` if impossible."""
    rows = tuple(annotations)
    if not _has_unique_ids(rows):
        return None
    cells = _available_cells(rows, quotas)
    if cells is None:
        return None
    options = _selection_options(rows, cells, quotas)
    if options is None:
        return None
    selected = _balanced_selection(options, quotas)
    if selected is None:
        return None
    return tuple(sorted(selected, key=lambda row: row.candidate.candidate_id))


def _has_unique_ids(rows: tuple[Annotation, ...]) -> bool:
    return len({row.candidate.candidate_id for row in rows}) == len(rows)


def _available_cells(rows: tuple[Annotation, ...], quotas: DatasetQuotas) -> list[str] | None:
    cells = sorted({row.candidate.h3_cell for row in rows})
    return cells if len(cells) == quotas.cell_count else None


def _group_by_stratum(rows: tuple[Annotation, ...]) -> dict[tuple[Source, str], list[Annotation]]:
    grouped: dict[tuple[Source, str], list[Annotation]] = defaultdict(list)
    for row in rows:
        grouped[row.candidate.stratum].append(row)
    return grouped


def _selection_options(
    rows: tuple[Annotation, ...],
    cells: list[str],
    quotas: DatasetQuotas,
) -> list[SelectionOption] | None:
    grouped = _group_by_stratum(rows)
    options: list[SelectionOption] = []
    for cell in cells:
        for source in Source:
            option = _option_for_group(grouped.get((source, cell), []), quotas.per_source_cell)
            if option is None:
                return None
            options.append(option)
    return options


def _option_for_group(group: list[Annotation], per_source_cell: int) -> SelectionOption | None:
    ordered = sorted(group, key=lambda row: row.candidate.candidate_id)
    yes_rows, no_rows = _label_rows(ordered)
    if not _has_enough_labels(yes_rows, no_rows, per_source_cell):
        return None
    return yes_rows, no_rows


def _label_rows(rows: list[Annotation]) -> SelectionOption:
    yes_rows: list[Annotation] = []
    no_rows: list[Annotation] = []
    for row in rows:
        (yes_rows if row.label is Label.YES else no_rows).append(row)
    return tuple(yes_rows), tuple(no_rows)


def _has_enough_labels(
    yes_rows: tuple[Annotation, ...],
    no_rows: tuple[Annotation, ...],
    per_source_cell: int,
) -> bool:
    minimum_yes = max(0, per_source_cell - len(no_rows))
    maximum_yes = min(per_source_cell, len(yes_rows))
    return minimum_yes <= maximum_yes


def _balanced_selection(
    options: list[SelectionOption],
    quotas: DatasetQuotas,
) -> tuple[Annotation, ...] | None:
    states: dict[int, tuple[Annotation, ...]] = {0: ()}
    for yes_rows, no_rows in options:
        states = _advance_states(states, yes_rows, no_rows, quotas)
    return states.get(quotas.per_label)


def _advance_states(
    states: dict[int, tuple[Annotation, ...]],
    yes_rows: tuple[Annotation, ...],
    no_rows: tuple[Annotation, ...],
    quotas: DatasetQuotas,
) -> dict[int, tuple[Annotation, ...]]:
    next_states: dict[int, tuple[Annotation, ...]] = {}
    minimum_yes = max(0, quotas.per_source_cell - len(no_rows))
    maximum_yes = min(quotas.per_source_cell, len(yes_rows))
    for current_yes, selected in states.items():
        for chosen_yes in range(minimum_yes, maximum_yes + 1):
            chosen_no = quotas.per_source_cell - chosen_yes
            total_yes = current_yes + chosen_yes
            if total_yes > quotas.per_label or total_yes in next_states:
                continue
            next_states[total_yes] = selected + yes_rows[:chosen_yes] + no_rows[:chosen_no]
    return next_states
