from dataclasses import replace

import pytest

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Label, Source


def make_candidate(candidate_id: str = "c-1") -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sentence="A sentence about a visible landscape.",
        source=Source.WIKIPEDIA,
        source_record_id="polygon-1",
        source_field="wikipedia_section",
        h3_cell="832830fffffffff",
        h3_resolution=3,
        latitude=45.0,
        longitude=2.0,
        place_name="A place",
        region="A region",
        source_url="https://example.test/article",
    )


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


def test_candidate_stratum_is_source_and_cell() -> None:
    candidate = make_candidate()

    assert candidate.stratum == (Source.WIKIPEDIA, "832830fffffffff")
