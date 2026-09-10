from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from landuse_sentence_relevance.analysis.annotation_csv import SENTENCE_COLUMN
from landuse_sentence_relevance.analysis.interrater import RaterTable, disagreements

CONTEXT_COLUMNS = ("source", "region", "polygon_name", "source_url")

MINORITY_RATER_COLUMN = "minority_rater"
MINORITY_LABEL_COLUMN = "minority_label"

SentenceContext = Mapping[str, Mapping[str, str]]


def majority_label(labels: Mapping[str, str]) -> str | None:
    """Return the label a strict majority of raters chose, or None when none exists."""

    if not labels:
        return None
    ranked = Counter(labels.values()).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def minority_raters(labels: Mapping[str, str]) -> tuple[str, ...]:
    """Name the raters the majority outvoted, in sorted order."""

    majority = majority_label(labels)
    if majority is None:
        return ()
    return tuple(name for name in sorted(labels) if labels[name] != majority)


def review_header(rater_names: Sequence[str], context_columns: Sequence[str]) -> tuple[str, ...]:
    """Order the review columns: the verdict, the labels, the sentence, then its provenance."""

    return (MINORITY_RATER_COLUMN, MINORITY_LABEL_COLUMN, *rater_names, SENTENCE_COLUMN, *context_columns)


def _review_row(
    entry: Mapping[str, Any],
    context: SentenceContext,
    rater_names: Sequence[str],
    context_columns: Sequence[str],
) -> dict[str, str]:
    """Render one disagreement as a flat, reviewable row."""

    labels = entry["labels"]
    sentence = entry[SENTENCE_COLUMN]
    outvoted = minority_raters(labels)
    return {
        MINORITY_RATER_COLUMN: "; ".join(outvoted),
        MINORITY_LABEL_COLUMN: "; ".join(sorted({labels[name] for name in outvoted})),
        **{name: labels[name] for name in rater_names},
        SENTENCE_COLUMN: sentence,
        **{column: context[sentence][column] for column in context_columns},
    }


def _review_sort_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (row[MINORITY_RATER_COLUMN], row[MINORITY_LABEL_COLUMN], row[SENTENCE_COLUMN])


def review_rows(
    raters: RaterTable,
    sentences: Sequence[str],
    context: SentenceContext,
    rater_names: Sequence[str] | None = None,
    context_columns: Sequence[str] = CONTEXT_COLUMNS,
) -> list[dict[str, str]]:
    """Build one row per disagreement, grouped by the rater the majority outvoted."""

    names = sorted(raters) if rater_names is None else rater_names
    rows = [_review_row(entry, context, names, context_columns) for entry in disagreements(raters, sentences)]
    return sorted(rows, key=_review_sort_key)
