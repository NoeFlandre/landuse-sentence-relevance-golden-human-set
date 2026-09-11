from __future__ import annotations

import csv
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.analysis.annotation_csv import read_table

LABEL_COLUMN = "label"
ROUND_SCHEMA_VERSION = 1


class RoundPreparationError(ValueError):
    """Raised when a reproducible LLM evaluation round cannot be prepared."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_labels(path: Path, rows: list[dict[str, str]]) -> None:
    if any(row[LABEL_COLUMN] not in {"yes", "no"} for row in rows):
        raise RoundPreparationError(f"benchmark {path} has a label other than 'yes' or 'no'")


def _validate_unique_sentences(path: Path, rows: list[dict[str, str]]) -> None:
    sentences = [row["sentence"] for row in rows]
    if len(sentences) != len(set(sentences)):
        raise RoundPreparationError(f"benchmark {path} contains duplicate sentences")


def _validate_benchmark(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        header, rows = read_table(path)
    except (OSError, ValueError) as error:
        raise RoundPreparationError(f"could not read benchmark {path}: {error}") from error
    if LABEL_COLUMN not in header:
        raise RoundPreparationError(f"benchmark {path} is missing the {LABEL_COLUMN!r} column")
    _validate_labels(path, rows)
    _validate_unique_sentences(path, rows)
    return header, rows


def _write_blinded_input(path: Path, header: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    blinded_header = tuple(column for column in header if column != LABEL_COLUMN)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=blinded_header, lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row[column] for column in blinded_header} for row in rows)
    path.write_bytes(buffer.getvalue().encode())


def _manifest(
    round_id: str,
    benchmark: Path,
    input_path: Path,
    prompt: Path,
    row_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": ROUND_SCHEMA_VERSION,
        "round_id": round_id,
        "status": "prepared",
        "benchmark": {"path": benchmark.name, "rows": row_count, "sha256": _sha256(benchmark)},
        "input": {"path": input_path.name, "rows": row_count, "sha256": _sha256(input_path)},
        "prompt": {"path": prompt.name, "sha256": _sha256(prompt)},
        "outputs": {"gpt": None, "claude": None},
        "analysis": {"path": "analysis/agreement.json"},
    }


def prepare_round(
    round_directory: Path,
    benchmark_path: Path,
    prompt_path: Path,
    *,
    round_id: str,
) -> dict[str, Any]:
    """Create one immutable, blinded LLM evaluation round from the benchmark."""

    if round_directory.exists():
        raise RoundPreparationError(f"round directory already exists: {round_directory}")
    if not prompt_path.is_file():
        raise RoundPreparationError(f"prompt file does not exist: {prompt_path}")
    header, rows = _validate_benchmark(benchmark_path)
    round_directory.mkdir(parents=True)
    benchmark = round_directory / "benchmark.csv"
    input_path = round_directory / "input.csv"
    prompt = round_directory / "prompt.md"
    benchmark.write_bytes(benchmark_path.read_bytes())
    prompt.write_bytes(prompt_path.read_bytes())
    _write_blinded_input(input_path, header, rows)
    (round_directory / "outputs").mkdir()
    (round_directory / "analysis").mkdir()
    manifest = _manifest(round_id, benchmark, input_path, prompt, len(rows))
    manifest_json = json.dumps(manifest, indent=2) + "\n"
    (round_directory / "manifest.json").write_bytes(manifest_json.encode())
    return manifest
