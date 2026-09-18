from dataclasses import replace

import pytest

import landuse_sentence_relevance.domain.selection as selection
from landuse_sentence_relevance.domain.constraints import DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label
from landuse_sentence_relevance.domain.selection import (
    _advance_states,
    _exceeds_quotas,
    select_final_annotations,
)
from landuse_sentence_relevance.domain.stratification import DEFAULT_SOURCES
from tests.builders import make_annotations


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
    assert {annotation.candidate.source for annotation in selected} == set(DEFAULT_SOURCES)


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


@pytest.mark.parametrize("wikipedia_yes", [0, 1, 2])
def test_selection_balances_real_categories_at_each_capacity_boundary(wikipedia_yes: int) -> None:
    quotas = DatasetQuotas(total=4, per_source=2, per_label=2, cell_count=4)
    rows = make_annotations()
    candidates = [rows[index].candidate for index in (0, 1, 50, 51)]
    labels = [Label.YES] * wikipedia_yes + [Label.NO] * (2 - wikipedia_yes)
    labels += [Label.YES] * (2 - wikipedia_yes) + [Label.NO] * wikipedia_yes
    annotations = [Annotation(candidate, label) for candidate, label in zip(candidates, labels, strict=True)]

    selected = select_final_annotations(reversed(annotations), quotas)

    assert selected == tuple(sorted(annotations, key=lambda row: row.candidate.candidate_id))


@pytest.mark.parametrize("label", list(Label))
def test_selection_returns_none_when_label_balance_is_impossible(label: Label) -> None:
    rows = [Annotation(row.candidate, label) for row in make_annotations()]

    assert select_final_annotations(rows) is None


@pytest.mark.parametrize("wikipedia_label", list(Label))
def test_selection_discards_surplus_rows_at_either_label_boundary(wikipedia_label: Label) -> None:
    quotas = DatasetQuotas(total=4, per_source=2, per_label=2, cell_count=4)
    website_label = Label.NO if wikipedia_label is Label.YES else Label.YES
    pool = make_annotations()
    rows = [Annotation(pool[index].candidate, wikipedia_label) for index in (0, 1, 2)]
    rows += [Annotation(pool[index].candidate, website_label) for index in (50, 51, 52)]

    assert select_final_annotations(reversed(rows), quotas) == (rows[3], rows[4], rows[0], rows[1])


@pytest.mark.parametrize("surplus_index", [1, 2])
def test_selection_does_not_overfill_either_mixed_source_quota(surplus_index: int) -> None:
    quotas = DatasetQuotas(total=4, per_source=2, per_label=2, cell_count=4)
    pool = make_annotations()
    rows = [
        Annotation(pool[index].candidate, label)
        for index, label in ((0, Label.YES), (1, Label.NO), (50, Label.YES), (51, Label.NO))
    ]
    surplus = rows[surplus_index]
    extra = replace(
        surplus,
        candidate=replace(
            surplus.candidate, candidate_id=f"{surplus.candidate.candidate_id}-extra", h3_cell="extra-cell"
        ),
    )

    assert select_final_annotations([extra, *reversed(rows)], quotas) == (rows[2], rows[3], rows[0], rows[1])


@pytest.mark.parametrize(("quota", "expected"), [(-1, None), (0, ())])
def test_selection_preserves_empty_pool_behavior_for_nonpositive_quotas(quota, expected) -> None:
    quotas = DatasetQuotas(total=2 * quota, per_source=quota, per_label=quota, cell_count=2 * quota)

    assert select_final_annotations((), quotas) == expected


def test_selection_returns_none_for_an_impossible_general_path() -> None:
    rows = [Annotation(row.candidate, Label.YES) for row in make_annotations()]
    extra = replace(rows[0].candidate, candidate_id="extra", h3_cell=rows[1].candidate.h3_cell)

    assert select_final_annotations([*rows, Annotation(extra, Label.YES)]) is None


def test_selection_rejects_duplicate_candidate_ids() -> None:
    rows = make_annotations()
    duplicate = replace(rows[0].candidate, candidate_id=rows[1].candidate.candidate_id)

    assert select_final_annotations([*rows[:-1], Annotation(duplicate, rows[-1].label)]) is None


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
