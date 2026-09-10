from __future__ import annotations

import pytest

from landuse_sentence_relevance.analysis.interrater import (
    VALID_LABELS,
    InterraterDataError,
    aligned_sentences,
    cohen_kappa,
    confusion_matrix,
    disagreements,
    fleiss_kappa,
    interrater_report,
    label_counts,
    normalise_label,
    pairwise_agreement,
    rater_labels,
    three_rater_agreement,
)

HUMAN = {"a": "yes", "b": "no", "c": "yes", "d": "no"}
GPT = {"a": "yes", "b": "no", "c": "no", "d": "no"}
CLAUDE = {"a": "yes", "b": "yes", "c": "yes", "d": "no"}


def test_valid_labels_are_the_two_lowercase_relevance_tokens() -> None:
    assert VALID_LABELS == ("no", "yes")


def test_normalise_label_accepts_surrounding_whitespace_and_upper_case() -> None:
    assert normalise_label("human", "a sentence", " YES ") == "yes"
    assert normalise_label("human", "a sentence", "no") == "no"


def test_normalise_label_rejects_a_value_outside_the_allowed_labels() -> None:
    with pytest.raises(
        InterraterDataError, match=r"^human has an invalid label 'maybe' for sentence 'a sentence'$"
    ):
        normalise_label("human", "a sentence", "maybe")


def test_normalise_label_rejects_an_empty_value() -> None:
    with pytest.raises(InterraterDataError, match="invalid label"):
        normalise_label("human", "a sentence", "")


def test_rater_labels_maps_sentences_to_normalised_labels() -> None:
    assert rater_labels("gpt", [("a", "Yes"), ("b", "NO")]) == {"a": "yes", "b": "no"}


def test_rater_labels_rejects_a_duplicated_sentence() -> None:
    with pytest.raises(InterraterDataError, match=r"^gpt has a duplicate sentence: 'a'$"):
        rater_labels("gpt", [("a", "yes"), ("a", "no")])


def test_rater_labels_rejects_an_empty_rater() -> None:
    with pytest.raises(InterraterDataError, match=r"^gpt has no rows$"):
        rater_labels("gpt", [])


def test_rater_labels_names_the_rater_and_sentence_of_an_invalid_label() -> None:
    with pytest.raises(InterraterDataError, match=r"^gpt has an invalid label 'maybe' for sentence 'a'$"):
        rater_labels("gpt", [("a", "maybe")])


def test_aligned_sentences_returns_the_shared_sentences_in_sorted_order() -> None:
    assert aligned_sentences({"human": HUMAN, "gpt": GPT, "claude": CLAUDE}) == ("a", "b", "c", "d")


def test_aligned_sentences_requires_at_least_two_raters() -> None:
    with pytest.raises(InterraterDataError, match=r"^at least two raters are required$"):
        aligned_sentences({"human": HUMAN})


def test_aligned_sentences_rejects_a_rater_missing_a_sentence() -> None:
    with pytest.raises(InterraterDataError, match=r"beta does not align with alpha: missing \['d'\]"):
        aligned_sentences({"alpha": HUMAN, "beta": {"a": "yes", "b": "no", "c": "no"}})


def test_aligned_sentences_rejects_a_rater_with_an_unexpected_sentence() -> None:
    extra = {"a": "yes", "b": "no", "c": "no", "d": "no", "e": "yes"}

    with pytest.raises(InterraterDataError, match=r"beta does not align with alpha: unexpected \['e'\]"):
        aligned_sentences({"alpha": HUMAN, "beta": extra})


def test_aligned_sentences_reports_missing_and_unexpected_sentences_together() -> None:
    swapped = {"a": "yes", "b": "no", "c": "no", "e": "yes"}

    with pytest.raises(InterraterDataError, match=r"missing \['d'\]; unexpected \['e'\]"):
        aligned_sentences({"alpha": HUMAN, "beta": swapped})


def test_aligned_sentences_checks_every_rater_against_the_reference() -> None:
    with pytest.raises(InterraterDataError, match="unexpected"):
        aligned_sentences({"claude": CLAUDE, "gpt": GPT, "human": {**HUMAN, "e": "yes"}})


def test_label_counts_counts_each_label_for_each_rater() -> None:
    counts = label_counts({"human": HUMAN, "gpt": GPT}, ("a", "b", "c", "d"))

    assert counts == {"gpt": {"no": 3, "yes": 1}, "human": {"no": 2, "yes": 2}}


def test_label_counts_is_restricted_to_the_matched_sentences() -> None:
    counts = label_counts({"human": HUMAN}, ("a", "b"))

    assert counts == {"human": {"no": 1, "yes": 1}}


def test_confusion_matrix_cross_tabulates_two_raters() -> None:
    matrix = confusion_matrix(HUMAN, GPT, ("a", "b", "c", "d"))

    assert matrix == {"no": {"no": 2, "yes": 0}, "yes": {"no": 1, "yes": 1}}


def test_cohen_kappa_scales_observed_agreement_against_chance() -> None:
    assert cohen_kappa(0.75, 0.5) == 0.5


def test_cohen_kappa_is_negative_below_chance_agreement() -> None:
    assert cohen_kappa(0.25, 0.5) == -0.5


def test_cohen_kappa_is_one_when_chance_agreement_is_total() -> None:
    assert cohen_kappa(1.0, 1.0) == 1.0


def test_pairwise_agreement_reports_counts_rates_kappa_and_confusion() -> None:
    result = pairwise_agreement("human", HUMAN, "gpt", GPT, ("a", "b", "c", "d"))

    assert result.left == "human"
    assert result.right == "gpt"
    assert result.matched == 4
    assert result.agreements == 3
    assert result.observed_agreement == 0.75
    assert result.expected_agreement == 0.5
    assert result.cohen_kappa == 0.5
    assert result.confusion_matrix == {"no": {"no": 2, "yes": 0}, "yes": {"no": 1, "yes": 1}}


def test_pairwise_agreement_serialises_to_a_sorted_mapping() -> None:
    result = pairwise_agreement("human", HUMAN, "gpt", GPT, ("a", "b", "c", "d"))

    assert result.as_dict() == {
        "left": "human",
        "right": "gpt",
        "matched": 4,
        "agreements": 3,
        "observed_agreement": 0.75,
        "expected_agreement": 0.5,
        "cohen_kappa": 0.5,
        "confusion_matrix": {"no": {"no": 2, "yes": 0}, "yes": {"no": 1, "yes": 1}},
    }


def test_fleiss_kappa_is_one_for_unanimous_and_balanced_ratings() -> None:
    raters = {
        "one": {"a": "yes", "b": "no"},
        "two": {"a": "yes", "b": "no"},
        "three": {"a": "yes", "b": "no"},
    }

    assert fleiss_kappa(raters, ("a", "b")) == 1.0


def test_fleiss_kappa_is_one_when_every_rating_uses_a_single_label() -> None:
    raters = {
        "one": {"a": "yes", "b": "yes"},
        "two": {"a": "yes", "b": "yes"},
        "three": {"a": "yes", "b": "yes"},
    }

    assert fleiss_kappa(raters, ("a", "b")) == 1.0


def test_fleiss_kappa_is_negative_when_items_split_against_a_skewed_marginal() -> None:
    raters = {
        "one": {"a": "yes", "b": "yes"},
        "two": {"a": "yes", "b": "no"},
        "three": {"a": "no", "b": "yes"},
    }

    assert fleiss_kappa(raters, ("a", "b")) == pytest.approx(-0.5)


def test_three_rater_agreement_counts_unanimous_rows() -> None:
    result = three_rater_agreement({"human": HUMAN, "gpt": GPT, "claude": CLAUDE}, ("a", "b", "c", "d"))

    assert result.matched == 4
    assert result.unanimous == 2
    assert result.unanimous_agreement == 0.5
    assert result.fleiss_kappa == pytest.approx(1 / 3)
    assert result.as_dict() == {
        "matched": 4,
        "unanimous": 2,
        "unanimous_agreement": 0.5,
        "fleiss_kappa": pytest.approx(1 / 3),
    }


def test_three_rater_agreement_separates_unanimous_rows_from_split_rows() -> None:
    raters = {
        "one": {"a": "yes", "b": "no", "c": "yes"},
        "two": {"a": "yes", "b": "no", "c": "no"},
        "three": {"a": "yes", "b": "no", "c": "yes"},
    }

    result = three_rater_agreement(raters, ("a", "b", "c"))

    assert result.matched == 3
    assert result.unanimous == 2
    assert result.unanimous_agreement == pytest.approx(2 / 3)


def test_disagreements_lists_every_non_unanimous_sentence_in_order() -> None:
    assert disagreements({"human": HUMAN, "gpt": GPT, "claude": CLAUDE}, ("a", "b", "c", "d")) == [
        {"sentence": "b", "labels": {"claude": "yes", "gpt": "no", "human": "no"}},
        {"sentence": "c", "labels": {"claude": "yes", "gpt": "no", "human": "yes"}},
    ]


def test_disagreements_is_empty_when_every_rater_matches() -> None:
    assert disagreements({"human": HUMAN, "copy": dict(HUMAN)}, ("a", "b")) == []


def test_interrater_report_assembles_every_requested_metric() -> None:
    report = interrater_report({"human": HUMAN, "gpt": GPT, "claude": CLAUDE})

    assert report["raters"] == ["claude", "gpt", "human"]
    assert report["labels"] == ["no", "yes"]
    assert report["matched_rows"] == 4
    assert report["label_counts"]["claude"] == {"no": 1, "yes": 3}
    assert [(pair["left"], pair["right"]) for pair in report["pairwise"]] == [
        ("claude", "gpt"),
        ("claude", "human"),
        ("gpt", "human"),
    ]
    assert report["three_rater"]["unanimous"] == 2
    assert [entry["sentence"] for entry in report["disagreements"]] == ["b", "c"]


def test_interrater_report_rejects_misaligned_raters() -> None:
    with pytest.raises(InterraterDataError, match="does not align"):
        interrater_report({"human": HUMAN, "gpt": {"a": "yes"}})
