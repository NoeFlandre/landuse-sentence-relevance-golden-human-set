from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.analysis.annotation_csv import read_sentence_context
from landuse_sentence_relevance.analysis.interrater import VALID_LABELS, RaterTable, disagreements

FINAL_LABEL_COLUMN = "final_label"
REMOVE_LABEL = "remove"
FINAL_LABELS = (*VALID_LABELS, REMOVE_LABEL)


class AdjudicationError(ValueError):
    """Raised when the adjudicated verdicts do not resolve the disagreements exactly."""


def normalise_final_label(sentence: str, value: str) -> str:
    """Return the canonical verdict, rejecting anything but yes, no, or remove."""

    label = value.strip().lower()
    if label not in FINAL_LABELS:
        raise AdjudicationError(f"{sentence!r} has an invalid final label {value!r}")
    return label


def read_verdicts(path: Path) -> dict[str, str]:
    """Read the adjudicated verdicts of a review CSV, keyed by exact sentence identity."""

    rows = read_sentence_context(path, (FINAL_LABEL_COLUMN,))
    return {
        sentence: normalise_final_label(sentence, row[FINAL_LABEL_COLUMN]) for sentence, row in rows.items()
    }


def _disagreeing_sentences(raters: RaterTable, sentences: Sequence[str]) -> set[str]:
    return {entry["sentence"] for entry in disagreements(raters, sentences)}


def validate_adjudication(raters: RaterTable, sentences: Sequence[str], verdicts: Mapping[str, str]) -> None:
    """Refuse verdicts that miss a disagreement or judge an already unanimous sentence."""

    disagreeing = _disagreeing_sentences(raters, sentences)
    problems: list[str] = []
    missing = sorted(disagreeing - set(verdicts))
    if missing:
        problems.append(f"unadjudicated disagreements: {missing}")
    extra = sorted(set(verdicts) - disagreeing)
    if extra:
        problems.append(f"verdicts on unanimous sentences: {extra}")
    if problems:
        raise AdjudicationError("; ".join(problems))


def _unanimous_label(raters: RaterTable, sentence: str) -> str:
    """Return the single label every rater gave this sentence."""

    agreed = {labels[sentence] for labels in raters.values()}
    return sorted(agreed)[0]


def adjudicated_labels(
    raters: RaterTable, sentences: Sequence[str], verdicts: Mapping[str, str]
) -> dict[str, str]:
    """Resolve every sentence to one label, dropping the ones the adjudicator removed."""

    validate_adjudication(raters, sentences, verdicts)
    resolved: dict[str, str] = {}
    for sentence in sentences:
        label = verdicts[sentence] if sentence in verdicts else _unanimous_label(raters, sentence)
        if label != REMOVE_LABEL:
            resolved[sentence] = label
    return resolved


def _final_label_counts(resolved: Mapping[str, str]) -> dict[str, int]:
    """Count each label in the resolved benchmark."""

    return {label: sum(value == label for value in resolved.values()) for label in VALID_LABELS}


def _verdict_counts(verdicts: Mapping[str, str]) -> dict[str, int]:
    """Count how many sentences each verdict resolved."""

    return {label: sum(verdict == label for verdict in verdicts.values()) for label in sorted(FINAL_LABELS)}


def _upheld_counts(raters: RaterTable, verdicts: Mapping[str, str]) -> dict[str, int]:
    """Count, per rater, how many verdicts confirmed that rater's original label."""

    return {
        name: sum(
            labels[sentence] == verdict for sentence, verdict in verdicts.items() if verdict != REMOVE_LABEL
        )
        for name, labels in sorted(raters.items())
    }


def adjudication_summary(
    raters: RaterTable, sentences: Sequence[str], verdicts: Mapping[str, str]
) -> dict[str, Any]:
    """Summarise how the adjudication resolved the disagreements."""

    resolved = adjudicated_labels(raters, sentences, verdicts)
    removed = sum(verdict == REMOVE_LABEL for verdict in verdicts.values())
    return {
        "matched_rows": len(sentences),
        "unanimous": len(sentences) - len(verdicts),
        "adjudicated": len(verdicts),
        "removed": removed,
        "final_rows": len(resolved),
        "final_label_counts": _final_label_counts(resolved),
        "verdict_counts": _verdict_counts(verdicts),
        "upheld": _upheld_counts(raters, verdicts),
    }
