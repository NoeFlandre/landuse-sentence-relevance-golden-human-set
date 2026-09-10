from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from landuse_sentence_relevance.analysis.interrater import InterraterDataError, rater_labels

SENTENCE_COLUMN = "sentence"


def _validate_columns(path: Path, fieldnames: Iterable[str], label_column: str) -> None:
    """Refuse a file that does not carry both the join key and the label column."""

    present = set(fieldnames)
    for column in (SENTENCE_COLUMN, label_column):
        if column not in present:
            raise InterraterDataError(f"{path} is missing column {column!r}")


def _row_pair(path: Path, row: Mapping[str, str | None], label_column: str) -> tuple[str, str]:
    """Extract the sentence and label of one row, refusing a truncated row."""

    sentence = row.get(SENTENCE_COLUMN)
    value = row.get(label_column)
    if sentence is None or value is None:
        raise InterraterDataError(f"{path} has a row missing {SENTENCE_COLUMN!r} or {label_column!r}")
    return sentence, value


def read_rater_labels(path: Path, rater: str, label_column: str) -> dict[str, str]:
    """Read one rater's labels from a CSV, keyed by exact sentence identity."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(path, reader.fieldnames or (), label_column)
        rows = [_row_pair(path, row, label_column) for row in reader]
    return rater_labels(rater, rows)
