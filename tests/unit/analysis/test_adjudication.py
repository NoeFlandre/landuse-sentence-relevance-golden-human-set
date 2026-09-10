from __future__ import annotations

from pathlib import Path

import pytest

from landuse_sentence_relevance.analysis.adjudication import (
    FINAL_LABEL_COLUMN,
    REMOVE_LABEL,
    AdjudicationError,
    adjudicated_labels,
    adjudication_summary,
    normalise_final_label,
    read_verdicts,
    validate_adjudication,
)

RATERS = {
    "human": {"a": "yes", "b": "no", "c": "yes", "d": "no"},
    "gpt": {"a": "yes", "b": "no", "c": "no", "d": "no"},
    "claude": {"a": "yes", "b": "yes", "c": "yes", "d": "no"},
}
SENTENCES = ("a", "b", "c", "d")
VERDICTS = {"b": "yes", "c": "remove"}


def test_final_label_column_and_removal_token_are_stable() -> None:
    assert FINAL_LABEL_COLUMN == "final_label"
    assert REMOVE_LABEL == "remove"


def test_normalise_final_label_accepts_the_three_verdicts_in_any_case() -> None:
    assert normalise_final_label("a", " Yes ") == "yes"
    assert normalise_final_label("a", "NO") == "no"
    assert normalise_final_label("a", "Remove") == "remove"


def test_normalise_final_label_rejects_anything_else() -> None:
    with pytest.raises(AdjudicationError, match=r"^'a' has an invalid final label 'maybe'$"):
        normalise_final_label("a", "maybe")


def test_validate_adjudication_accepts_exactly_the_disagreeing_sentences() -> None:
    validate_adjudication(RATERS, SENTENCES, VERDICTS)


def test_validate_adjudication_rejects_an_unadjudicated_disagreement() -> None:
    with pytest.raises(AdjudicationError, match=r"unadjudicated disagreements: \['c'\]"):
        validate_adjudication(RATERS, SENTENCES, {"b": "yes"})


def test_validate_adjudication_rejects_a_verdict_on_a_unanimous_sentence() -> None:
    with pytest.raises(AdjudicationError, match=r"verdicts on unanimous sentences: \['a'\]"):
        validate_adjudication(RATERS, SENTENCES, {**VERDICTS, "a": "no"})


def test_validate_adjudication_reports_both_faults_together() -> None:
    with pytest.raises(
        AdjudicationError,
        match=r"^unadjudicated disagreements: \['c'\]; verdicts on unanimous sentences: \['a'\]$",
    ):
        validate_adjudication(RATERS, SENTENCES, {"b": "yes", "a": "no"})


def test_adjudicated_labels_keep_unanimous_rows_and_apply_each_verdict() -> None:
    assert adjudicated_labels(RATERS, SENTENCES, VERDICTS) == {"a": "yes", "b": "yes", "d": "no"}


def test_adjudicated_labels_drop_every_removed_sentence() -> None:
    labels = adjudicated_labels(RATERS, SENTENCES, {"b": "remove", "c": "remove"})

    assert labels == {"a": "yes", "d": "no"}


def test_adjudication_summary_counts_the_outcome_of_every_verdict() -> None:
    assert adjudication_summary(RATERS, SENTENCES, VERDICTS) == {
        "matched_rows": 4,
        "unanimous": 2,
        "adjudicated": 2,
        "removed": 1,
        "final_rows": 3,
        "final_label_counts": {"no": 1, "yes": 2},
        "verdict_counts": {"no": 0, "remove": 1, "yes": 1},
        "upheld": {"claude": 1, "gpt": 0, "human": 0},
    }


def test_adjudication_summary_credits_every_rater_that_matched_the_verdict() -> None:
    summary = adjudication_summary(RATERS, SENTENCES, {"b": "no", "c": "yes"})

    assert summary["removed"] == 0
    assert summary["final_rows"] == 4
    assert summary["upheld"] == {"claude": 1, "gpt": 1, "human": 2}


def test_read_verdicts_reads_and_normalises_every_verdict(tmp_path: Path) -> None:
    path = tmp_path / "adjudication.csv"
    path.write_text("sentence,final_label\nb,Yes\nc,REMOVE\n", encoding="utf-8")

    assert read_verdicts(path) == {"b": "yes", "c": "remove"}


def test_read_verdicts_names_the_sentence_behind_an_invalid_verdict(tmp_path: Path) -> None:
    path = tmp_path / "adjudication.csv"
    path.write_text("sentence,final_label\nb,probably\n", encoding="utf-8")

    with pytest.raises(AdjudicationError, match=r"^'b' has an invalid final label 'probably'$"):
        read_verdicts(path)
