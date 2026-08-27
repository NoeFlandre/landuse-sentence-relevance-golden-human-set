from dataclasses import replace

import pytest

from landuse_sentence_relevance.domain.constraints import (
    DatasetQuotas,
    FinalDatasetNotReadyError,
    validate_final_dataset,
)
from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from tests.unit.test_models import make_candidate


def make_annotations() -> list[Annotation]:
    annotations: list[Annotation] = []
    for cell_number in range(100):
        cell = f"cell-{cell_number:02d}"
        source = Source.WIKIPEDIA if cell_number < 50 else Source.WEBSITE
        candidate = replace(
            make_candidate(f"{source.value}-{cell}"),
            source=source,
            h3_cell=cell,
        )
        label = Label.YES if cell_number < 50 else Label.NO
        annotations.append(Annotation(candidate=candidate, label=label))
    return annotations


def with_invalid_resolution(rows: list[Annotation]) -> list[Annotation]:
    object.__setattr__(rows[0].candidate, "h3_resolution", 2)
    return rows


def test_valid_final_dataset_meets_every_quota() -> None:
    validate_final_dataset(make_annotations())


def test_final_dataset_rejects_two_sentences_from_one_h3_cell() -> None:
    rows = make_annotations()
    rows[-1] = replace(rows[-1], candidate=replace(rows[-1].candidate, h3_cell=rows[0].candidate.h3_cell))

    with pytest.raises(FinalDatasetNotReadyError, match="each H3 cell must contain exactly one row"):
        validate_final_dataset(rows)


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
        cell_count=4,
        rows_per_cell=1,
    )
    rows = [make_annotations()[index] for index in (0, 1, 50, 51)]

    validate_final_dataset(rows, quotas)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 99},
        {"per_source": 49},
        {"per_label": 49},
        {"cell_count": 99},
        {"rows_per_cell": 2},
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
        rows[0] = replace(rows[0], candidate=replace(rows[0].candidate, h3_cell=rows[1].candidate.h3_cell))
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
                replace(rows[0], candidate=replace(rows[0].candidate, h3_cell=rows[1].candidate.h3_cell)),
                *rows[1:],
            ],
            "each H3 cell must contain exactly one row",
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


def test_final_dataset_uses_disjoint_source_cells() -> None:
    rows = make_annotations()

    wikipedia_cells = {row.candidate.h3_cell for row in rows if row.candidate.source is Source.WIKIPEDIA}
    website_cells = {row.candidate.h3_cell for row in rows if row.candidate.source is Source.WEBSITE}

    assert wikipedia_cells.isdisjoint(website_cells)
