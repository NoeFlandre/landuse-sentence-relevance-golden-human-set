from dataclasses import replace

from tests.unit.test_constraints import make_annotations

from landuse_sentence_relevance.domain.constraints import DatasetQuotas
from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.selection import (
    _advance_states,
    _label_rows,
    select_final_annotations,
)


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


def test_selection_returns_none_when_label_balance_is_impossible() -> None:
    rows = [Annotation(row.candidate, Label.YES) for row in make_annotations()]

    assert select_final_annotations(rows) is None


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
    states: dict[int, tuple[Annotation, ...]] = {50: tuple()}

    assert _advance_states(states, tuple(row for row in rows), (), DatasetQuotas()) == {}


def test_advance_states_keeps_alternative_paths_after_a_duplicate_total() -> None:
    rows = make_annotations()[:3]
    states: dict[int, tuple[Annotation, ...]] = {
        0: tuple(),
        1: (Annotation(rows[0].candidate, Label.YES),),
    }

    advanced = _advance_states(
        states,
        (Annotation(rows[0].candidate, Label.YES),),
        (
            Annotation(rows[1].candidate, Label.NO),
            Annotation(rows[2].candidate, Label.NO),
        ),
        DatasetQuotas(),
    )

    assert set(advanced) == {0, 1, 2}
