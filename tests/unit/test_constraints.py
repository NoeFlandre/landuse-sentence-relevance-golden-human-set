from dataclasses import replace

import pytest
from tests.unit.test_models import make_candidate

from landuse_sentence_relevance.domain.constraints import (
    DatasetQuotas,
    FinalDatasetNotReadyError,
    _cell_source_count,
    validate_final_dataset,
)
from landuse_sentence_relevance.domain.models import Annotation, Label, Source


def make_annotations() -> list[Annotation]:
    annotations: list[Annotation] = []
    for cell_number in range(25):
        cell = f"cell-{cell_number:02d}"
        for source in Source:
            for position in range(2):
                candidate = replace(
                    make_candidate(f"{source.value}-{cell}-{position}"),
                    source=source,
                    h3_cell=cell,
                )
                label = Label.YES if len(annotations) < 50 else Label.NO
                annotations.append(Annotation(candidate=candidate, label=label))
    return annotations


def with_invalid_resolution(rows: list[Annotation]) -> list[Annotation]:
    object.__setattr__(rows[0].candidate, "h3_resolution", 2)
    return rows


def test_valid_final_dataset_meets_every_quota() -> None:
    validate_final_dataset(make_annotations())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[:-1],
        lambda rows: [*rows, rows[0]],
        lambda rows: [Annotation(a.candidate, Label.NO) for a in rows],
    ],
)
def test_invalid_final_dataset_is_rejected(mutator) -> None:
    with pytest.raises(FinalDatasetNotReadyError):
        validate_final_dataset(mutator(make_annotations()))


def test_custom_quotas_are_supported() -> None:
    quotas = DatasetQuotas(
        total=4,
        per_source=2,
        per_label=2,
        cell_count=1,
        per_source_cell=2,
    )
    rows = make_annotations()[:4]
    rows = [
        Annotation(
            candidate=replace(a.candidate, h3_cell="cell-00"),
            label=Label.YES if index < 2 else Label.NO,
        )
        for index, a in enumerate(rows)
    ]

    validate_final_dataset(rows, quotas)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 99},
        {"per_source": 49},
        {"per_label": 49},
        {"cell_count": 24},
        {"h3_resolution": 2},
    ],
)
def test_inconsistent_quotas_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        DatasetQuotas(**kwargs)


@pytest.mark.parametrize("kind", ["duplicate", "source", "label", "cell", "resolution"])
def test_each_final_dataset_invariant_has_a_failing_case(kind: str) -> None:
    rows = make_annotations()
    if kind == "duplicate":
        rows[-1] = replace(
            rows[-1],
            candidate=replace(rows[-1].candidate, candidate_id=rows[0].candidate.candidate_id),
        )
    elif kind == "source":
        rows[0] = replace(rows[0], candidate=replace(rows[0].candidate, source=Source.WEBSITE))
    elif kind == "label":
        rows[0] = replace(rows[0], label=Label.NO)
    elif kind == "cell":
        rows[0] = replace(rows[0], candidate=replace(rows[0].candidate, h3_cell="new-cell"))
    else:
        object.__setattr__(rows[0].candidate, "h3_resolution", 2)

    with pytest.raises(FinalDatasetNotReadyError):
        validate_final_dataset(rows)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[:-1], "expected 100 rows, received 99"),
        (
            lambda rows: [*rows[:-1], Annotation(rows[0].candidate, rows[-1].label)],
            "candidate IDs must be unique",
        ),
        (
            lambda rows: [
                Annotation(replace(row.candidate, source=Source.WEBSITE), row.label) if index == 0 else row
                for index, row in enumerate(rows)
            ],
            "source quotas are not satisfied",
        ),
        (
            lambda rows: [Annotation(row.candidate, Label.NO) for row in rows],
            "label quotas are not satisfied",
        ),
        (
            lambda rows: [
                replace(rows[0], candidate=replace(rows[0].candidate, h3_cell="new-cell")),
                *rows[1:],
            ],
            "the final dataset must cover the configured H3 cells",
        ),
        (
            with_invalid_resolution,
            "all rows must use the configured H3 resolution",
        ),
    ],
)
def test_final_dataset_errors_are_specific(mutator, message: str) -> None:
    with pytest.raises(FinalDatasetNotReadyError) as error:
        validate_final_dataset(mutator(make_annotations()))

    assert str(error.value) == message


def test_cell_source_count_counts_only_the_requested_source() -> None:
    rows = tuple(make_annotations()[:1])

    assert _cell_source_count(rows, "cell-00", Source.WIKIPEDIA) == 1
    assert _cell_source_count(rows, "cell-00", Source.WEBSITE) == 0


def test_final_dataset_reports_an_unbalanced_source_within_a_cell() -> None:
    rows = make_annotations()
    rows[0] = replace(rows[0], candidate=replace(rows[0].candidate, source=Source.WEBSITE))
    rows[6] = replace(rows[6], candidate=replace(rows[6].candidate, source=Source.WIKIPEDIA))

    with pytest.raises(FinalDatasetNotReadyError) as error:
        validate_final_dataset(rows)

    assert str(error.value) == "each H3 cell must contain the same source quota"
