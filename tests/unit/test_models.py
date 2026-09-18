from dataclasses import replace
from typing import cast

import pytest

from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from tests.builders import make_candidate


def test_annotation_round_trip_preserves_provenance() -> None:
    annotation = Annotation(candidate=make_candidate(), label=Label.YES)

    restored = Annotation.from_dict(annotation.to_dict())

    assert restored == annotation
    assert restored.to_dict()["source"] == "wikipedia"
    assert restored.to_dict()["label"] == "yes"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sentence", ""),
        ("language", "fr"),
        ("h3_cell", ""),
        ("source_record_id", ""),
        ("source_field", ""),
    ],
)
def test_candidate_rejects_invalid_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=r"non-empty string|candidate language"):
        replace(make_candidate(), **{field: value})


def test_candidate_error_identifies_an_empty_candidate_id() -> None:
    with pytest.raises(ValueError, match=r"^candidate_id must be a non-empty string$"):
        replace(make_candidate(), candidate_id=" ")


def test_candidate_rejects_non_string_sentence_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        replace(make_candidate(), sentence=cast(str, 42))


def test_candidate_stratum_is_source_and_cell() -> None:
    candidate = make_candidate()

    assert candidate.stratum == (Source.WIKIPEDIA, "832830fffffffff")
