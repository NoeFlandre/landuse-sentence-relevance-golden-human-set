from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import landuse_sentence_relevance.domain.selection as selection
import landuse_sentence_relevance.domain.selection_direct as selection_direct
from landuse_sentence_relevance.domain.constraints import DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.selection import (
    _advance_states,
    _exceeds_quotas,
    _label_rows,
    select_final_annotations,
)
from tests.unit.test_constraints import make_annotations


def test_selection_finds_a_balanced_subset_from_extra_annotations() -> None:
    rows = make_annotations()
    extras = [
        Annotation(candidate=replace(rows[0].candidate, candidate_id="extra-yes"), label=Label.YES),
        Annotation(candidate=replace(rows[1].candidate, candidate_id="extra-no"), label=Label.NO),
    ]

    selected = select_final_annotations(rows + extras)

    assert selected is not None
    assert len(selected) == 100
    assert sum(annotation.label is Label.YES for annotation in selected) == 50
    assert {annotation.candidate.source for annotation in selected} == set(Source)


def test_selection_uses_direct_path_for_one_row_per_cell(monkeypatch) -> None:
    rows = make_annotations()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("state expansion is unnecessary for one row per cell")

    monkeypatch.setattr(selection, "_advance_states", fail_if_called)

    selected = select_final_annotations(rows)

    assert selected is not None
    assert [row.candidate.candidate_id for row in selected] == sorted(
        row.candidate.candidate_id for row in rows
    )


def test_selection_uses_general_path_when_rows_per_cell_is_not_one(monkeypatch) -> None:
    rows = [make_annotations()[index] for index in (0, 1, 50, 51)]
    quotas = DatasetQuotas(total=4, per_source=2, per_label=2, cell_count=2, rows_per_cell=2)
    calls = 0
    advance_states = selection._advance_states

    def observe_advance_states(*args, **kwargs):
        nonlocal calls
        calls += 1
        return advance_states(*args, **kwargs)

    monkeypatch.setattr(selection, "_advance_states", observe_advance_states)

    assert select_final_annotations(rows, quotas) is not None
    assert calls > 0


def test_feasible_diagonal_respects_both_quota_bounds(monkeypatch) -> None:
    quotas = cast(DatasetQuotas, SimpleNamespace(per_source=2, per_label=3))
    calls = []

    def fake_required_counts(diagonal, quotas):
        return {"diagonal": diagonal}

    def fake_has_capacity(rows_by_category, required):
        calls.append(required["diagonal"])
        return required["diagonal"] == 0

    monkeypatch.setattr(selection_direct, "_required_counts", fake_required_counts)
    monkeypatch.setattr(selection_direct, "_has_capacity", fake_has_capacity)

    assert selection_direct._feasible_diagonal({}, quotas) is None
    assert calls == [1, 2]


def test_feasible_diagonal_includes_zero_when_quotas_are_balanced(monkeypatch) -> None:
    quotas = cast(DatasetQuotas, SimpleNamespace(per_source=2, per_label=2))

    monkeypatch.setattr(
        selection_direct,
        "_required_counts",
        lambda diagonal, quotas: {"diagonal": diagonal},
    )
    monkeypatch.setattr(
        selection_direct,
        "_has_capacity",
        lambda rows_by_category, required: required["diagonal"] == 0,
    )

    assert selection_direct._feasible_diagonal({}, quotas) == 0


def test_selection_returns_none_when_label_balance_is_impossible() -> None:
    rows = [Annotation(row.candidate, Label.YES) for row in make_annotations()]

    assert select_final_annotations(rows) is None


def test_selection_returns_none_for_an_impossible_general_path() -> None:
    rows = [Annotation(row.candidate, Label.YES) for row in make_annotations()]
    extra = replace(rows[0].candidate, candidate_id="extra", h3_cell=rows[1].candidate.h3_cell)

    assert select_final_annotations([*rows, Annotation(extra, Label.YES)]) is None


def test_selection_rejects_duplicate_candidate_ids() -> None:
    rows = make_annotations()
    duplicate = replace(rows[0].candidate, candidate_id=rows[1].candidate.candidate_id)

    assert select_final_annotations([*rows[:-1], Annotation(duplicate, rows[-1].label)]) is None


def test_label_rows_separates_yes_and_no_annotations() -> None:
    rows = make_annotations()[:4]

    yes_rows, no_rows = _label_rows(
        [
            Annotation(rows[0].candidate, Label.YES),
            Annotation(rows[1].candidate, Label.NO),
        ]
    )

    assert [row.candidate.candidate_id for row in yes_rows] == [rows[0].candidate.candidate_id]
    assert [row.candidate.candidate_id for row in no_rows] == [rows[1].candidate.candidate_id]


def test_advance_states_discards_states_over_the_label_quota() -> None:
    rows = make_annotations()[:2]
    states: dict[tuple[int, int, int], tuple[Annotation, ...]] = {(100, 50, 50): tuple()}

    assert _advance_states(states, tuple(rows), DatasetQuotas()) == states


def test_advance_states_keeps_alternative_paths_after_a_duplicate_total() -> None:
    rows = make_annotations()
    states: dict[tuple[int, int, int], tuple[Annotation, ...]] = {(0, 0, 0): tuple()}

    advanced = _advance_states(
        states,
        (rows[0], rows[50]),
        DatasetQuotas(total=4, per_source=2, per_label=2, cell_count=4, rows_per_cell=1),
    )

    assert set(advanced) == {(0, 0, 0), (1, 1, 1), (1, 0, 0)}


def test_advance_states_skips_an_over_quota_cell_and_keeps_later_rows() -> None:
    rows = make_annotations()
    states: dict[tuple[int, int, int], tuple[Annotation, ...]] = {(0, 1, 0): tuple()}
    cell_rows = (
        Annotation(replace(rows[0].candidate, candidate_id="a-over"), Label.NO),
        Annotation(replace(rows[50].candidate, candidate_id="z-valid"), Label.YES),
    )

    advanced = _advance_states(
        states,
        cell_rows,
        DatasetQuotas(total=2, per_source=1, per_label=1, cell_count=2, rows_per_cell=1),
    )

    assert (1, 1, 1) in advanced
    assert advanced[(1, 1, 1)][0].candidate.candidate_id == "z-valid"


def test_exceeds_quotas_checks_each_dimension_independently() -> None:
    quotas = DatasetQuotas(total=2, per_source=1, per_label=1, cell_count=2, rows_per_cell=1)

    assert _exceeds_quotas((3, 0, 0), quotas)
    assert _exceeds_quotas((1, 2, 0), quotas)
    assert _exceeds_quotas((1, 0, 2), quotas)
