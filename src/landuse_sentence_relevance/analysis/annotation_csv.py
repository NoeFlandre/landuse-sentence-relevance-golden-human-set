from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from landuse_sentence_relevance.analysis.interrater import InterraterDataError, rater_labels

SENTENCE_COLUMN = "sentence"


def _validate_columns(path: Path, fieldnames: Iterable[str], columns: Sequence[str]) -> None:
    """Refuse a file that does not carry the join key and every requested column."""

    present = set(fieldnames)
    for column in (SENTENCE_COLUMN, *columns):
        if column not in present:
            raise InterraterDataError(f"{path} is missing column {column!r}")


def _row_mapping(path: Path, row: Mapping[str, str | None], columns: Sequence[str]) -> dict[str, str]:
    """Extract the requested values of one row, refusing a truncated row."""

    values: dict[str, str] = {}
    for column in columns:
        value = row.get(column)
        if value is None:
            raise InterraterDataError(f"{path} has a row missing one of {list(columns)}")
        values[column] = value
    return values


def _read_columns(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read the sentence and the requested columns of every row, in file order."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(path, reader.fieldnames or (), columns)
        return [_row_mapping(path, row, (SENTENCE_COLUMN, *columns)) for row in reader]


def read_rater_labels(path: Path, rater: str, label_column: str) -> dict[str, str]:
    """Read one rater's labels from a CSV, keyed by exact sentence identity."""

    rows = _read_columns(path, (label_column,))
    return rater_labels(rater, [(row[SENTENCE_COLUMN], row[label_column]) for row in rows])


def read_sentence_context(path: Path, columns: Sequence[str]) -> dict[str, dict[str, str]]:
    """Read the requested provenance columns of a CSV, keyed by exact sentence identity."""

    context: dict[str, dict[str, str]] = {}
    for row in _read_columns(path, columns):
        sentence = row[SENTENCE_COLUMN]
        if sentence in context:
            raise InterraterDataError(f"{path} has a duplicate sentence: {sentence!r}")
        context[sentence] = {column: row[column] for column in columns}
    if not context:
        raise InterraterDataError(f"{path} has no rows")
    return context
