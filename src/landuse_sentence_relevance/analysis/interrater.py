from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

VALID_LABELS = ("no", "yes")

RaterTable = Mapping[str, Mapping[str, str]]


class InterraterDataError(ValueError):
    """Raised when rater rows cannot be aligned one-to-one on sentence identity."""


def normalise_label(rater: str, sentence: str, value: str) -> str:
    """Return the canonical label, rejecting anything outside the allowed tokens."""

    label = value.strip().lower()
    if label not in VALID_LABELS:
        raise InterraterDataError(f"{rater} has an invalid label {value!r} for sentence {sentence!r}")
    return label


def rater_labels(rater: str, rows: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Index one rater's rows by exact sentence identity."""

    labels: dict[str, str] = {}
    for sentence, value in rows:
        if sentence in labels:
            raise InterraterDataError(f"{rater} has a duplicate sentence: {sentence!r}")
        labels[sentence] = normalise_label(rater, sentence, value)
    if not labels:
        raise InterraterDataError(f"{rater} has no rows")
    return labels


def _alignment_problems(expected: set[str], current: set[str]) -> list[str]:
    """Describe how one rater's sentence set departs from the reference set."""

    problems: list[str] = []
    missing = sorted(expected - current)
    unexpected = sorted(current - expected)
    if missing:
        problems.append(f"missing {missing}")
    if unexpected:
        problems.append(f"unexpected {unexpected}")
    return problems


def aligned_sentences(raters: RaterTable) -> tuple[str, ...]:
    """Return the sentences every rater labelled, refusing any one-to-one violation."""

    names = sorted(raters)
    if len(names) < 2:
        raise InterraterDataError("at least two raters are required")
    reference = names[0]
    expected = set(raters[reference])
    for name in names[1:]:
        problems = _alignment_problems(expected, set(raters[name]))
        if problems:
            raise InterraterDataError(f"{name} does not align with {reference}: {'; '.join(problems)}")
    return tuple(sorted(expected))


def label_counts(raters: RaterTable, sentences: Sequence[str]) -> dict[str, dict[str, int]]:
    """Count each label per rater over the matched sentences."""

    return {
        name: {label: sum(labels[sentence] == label for sentence in sentences) for label in VALID_LABELS}
        for name, labels in sorted(raters.items())
    }


def confusion_matrix(
    left: Mapping[str, str], right: Mapping[str, str], sentences: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Cross-tabulate two raters as rows of left labels by columns of right labels."""

    pairs = Counter((left[sentence], right[sentence]) for sentence in sentences)
    return {row: {column: pairs[(row, column)] for column in VALID_LABELS} for row in VALID_LABELS}


def cohen_kappa(observed_agreement: float, expected_agreement: float) -> float:
    """Return chance-corrected agreement, treating total expected agreement as perfect."""

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


@dataclass(frozen=True, slots=True)
class PairwiseAgreement:
    """Agreement between two raters over the matched sentences."""

    left: str
    right: str
    matched: int
    agreements: int
    observed_agreement: float
    expected_agreement: float
    cohen_kappa: float
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "matched": self.matched,
            "agreements": self.agreements,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "cohen_kappa": self.cohen_kappa,
            "confusion_matrix": self.confusion_matrix,
        }


def pairwise_agreement(
    left_name: str,
    left: Mapping[str, str],
    right_name: str,
    right: Mapping[str, str],
    sentences: Sequence[str],
) -> PairwiseAgreement:
    """Compute raw agreement, chance agreement, Cohen's kappa, and the confusion matrix."""

    matched = len(sentences)
    matrix = confusion_matrix(left, right, sentences)
    agreements = sum(matrix[label][label] for label in VALID_LABELS)
    observed = agreements / matched
    expected = sum(
        sum(matrix[label].values()) * sum(row[label] for row in matrix.values()) for label in VALID_LABELS
    ) / (matched * matched)
    return PairwiseAgreement(
        left=left_name,
        right=right_name,
        matched=matched,
        agreements=agreements,
        observed_agreement=observed,
        expected_agreement=expected,
        cohen_kappa=cohen_kappa(observed, expected),
        confusion_matrix=matrix,
    )


def _rating_counts(raters: RaterTable, sentences: Sequence[str]) -> list[Counter[str]]:
    """Count, per sentence, how many raters chose each label."""

    return [Counter(labels[sentence] for labels in raters.values()) for sentence in sentences]


def _mean_item_agreement(counts: Sequence[Counter[str]], rater_count: int) -> float:
    """Return the mean proportion of agreeing rater pairs per sentence."""

    ordered_pairs = rater_count * (rater_count - 1)
    return sum(sum(count * count for count in row.values()) - rater_count for row in counts) / (
        len(counts) * ordered_pairs
    )


def _chance_agreement(counts: Sequence[Counter[str]], rater_count: int) -> float:
    """Return the agreement expected from the pooled label marginals."""

    assignments = len(counts) * rater_count
    return sum((sum(row[label] for row in counts) / assignments) ** 2 for label in VALID_LABELS)


def fleiss_kappa(raters: RaterTable, sentences: Sequence[str]) -> float:
    """Return Fleiss' kappa across all raters, treating a single-label corpus as perfect."""

    counts = _rating_counts(raters, sentences)
    observed = _mean_item_agreement(counts, len(raters))
    expected = _chance_agreement(counts, len(raters))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True, slots=True)
class MultiRaterAgreement:
    """Agreement across every rater at once."""

    matched: int
    unanimous: int
    unanimous_agreement: float
    fleiss_kappa: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "unanimous": self.unanimous,
            "unanimous_agreement": self.unanimous_agreement,
            "fleiss_kappa": self.fleiss_kappa,
        }


def three_rater_agreement(raters: RaterTable, sentences: Sequence[str]) -> MultiRaterAgreement:
    """Count sentences where every rater chose the same label, plus Fleiss' kappa."""

    matched = len(sentences)
    unanimous = sum(len({labels[sentence] for labels in raters.values()}) == 1 for sentence in sentences)
    return MultiRaterAgreement(
        matched=matched,
        unanimous=unanimous,
        unanimous_agreement=unanimous / matched,
        fleiss_kappa=fleiss_kappa(raters, sentences),
    )


def disagreements(raters: RaterTable, sentences: Sequence[str]) -> list[dict[str, Any]]:
    """List every sentence the raters did not label unanimously, in matched order."""

    rows: list[dict[str, Any]] = []
    for sentence in sentences:
        labels = {name: raters[name][sentence] for name in sorted(raters)}
        if len(set(labels.values())) > 1:
            rows.append({"sentence": sentence, "labels": labels})
    return rows


def interrater_report(raters: RaterTable) -> dict[str, Any]:
    """Assemble the full deterministic agreement report for the aligned raters."""

    sentences = aligned_sentences(raters)
    names = sorted(raters)
    return {
        "raters": names,
        "labels": list(VALID_LABELS),
        "matched_rows": len(sentences),
        "label_counts": label_counts(raters, sentences),
        "pairwise": [
            pairwise_agreement(left, raters[left], right, raters[right], sentences).as_dict()
            for left, right in combinations(names, 2)
        ],
        "three_rater": three_rater_agreement(raters, sentences).as_dict(),
        "disagreements": disagreements(raters, sentences),
    }
