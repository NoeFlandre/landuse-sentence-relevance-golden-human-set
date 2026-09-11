from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.analysis.adjudication import (
    AdjudicationError,
    adjudicated_labels,
    adjudication_summary,
    read_verdicts,
)
from landuse_sentence_relevance.analysis.annotation_csv import (
    SENTENCE_COLUMN,
    read_rater_labels,
    read_sentence_context,
    read_table,
)
from landuse_sentence_relevance.analysis.disagreement_review import (
    CONTEXT_COLUMNS,
    review_header,
    review_rows,
)
from landuse_sentence_relevance.analysis.interrater import (
    InterraterDataError,
    aligned_sentences,
    interrater_report,
)

ROUND_ONE_DIRECTORY = Path("results/evaluations/round-01")
BENCHMARK_CSV = ROUND_ONE_DIRECTORY / "human.csv"
GPT_CSV = ROUND_ONE_DIRECTORY / "outputs/gpt.csv"
CLAUDE_CSV = ROUND_ONE_DIRECTORY / "outputs/claude.csv"

DEFAULT_OUTPUT_DIRECTORY = ROUND_ONE_DIRECTORY / "analysis"
DEFAULT_REVIEW_CSV = Path("data/interrater/disagreements.csv")
DEFAULT_ADJUDICATION_CSV = Path("data/interrater/adjudication.csv")
DEFAULT_BENCHMARK_CSV = Path("data/benchmark/v2-adjudicated.csv")

JSON_FILENAME = "agreement.json"


@dataclass(frozen=True, slots=True)
class RaterSource:
    """One labelled CSV and the column that carries its labels."""

    name: str
    path: Path
    label_column: str


DEFAULT_SOURCES = (
    RaterSource("human", BENCHMARK_CSV, "label"),
    RaterSource("gpt", GPT_CSV, "llm_label"),
    RaterSource("claude", CLAUDE_CSV, "llm_label"),
)


def file_digest(path: Path) -> str:
    """Return the SHA-256 of the file bytes, pinning the inputs of a report."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_raters(sources: Sequence[RaterSource]) -> dict[str, dict[str, str]]:
    return {
        source.name: read_rater_labels(source.path, source.name, source.label_column) for source in sources
    }


def build_report(sources: Sequence[RaterSource]) -> dict[str, Any]:
    """Read every source, align it, and return the agreement report plus provenance."""

    raters = _read_raters(sources)
    report = interrater_report(raters)
    report["sources"] = [
        {
            "name": source.name,
            "path": source.path.as_posix(),
            "label_column": source.label_column,
            "rows": len(raters[source.name]),
            "sha256": file_digest(source.path),
        }
        for source in sources
    ]
    return report


def build_review_rows(
    sources: Sequence[RaterSource],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Build the reviewable disagreement table, keeping the raters in source order."""

    raters = _read_raters(sources)
    sentences = aligned_sentences(raters)
    names = [source.name for source in sources]
    context = read_sentence_context(sources[0].path, CONTEXT_COLUMNS)
    rows = review_rows(raters, sentences, context, rater_names=names)
    return review_header(names, CONTEXT_COLUMNS), rows


def build_adjudication(
    sources: Sequence[RaterSource], adjudication: Path
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, str]]]:
    """Summarise the adjudication and rebuild the human benchmark from its verdicts.

    The benchmark keeps the human file's own columns and row order. Every removed
    sentence is dropped and every adjudicated label replaces the human's original.
    """

    raters = _read_raters(sources)
    sentences = aligned_sentences(raters)
    verdicts = read_verdicts(adjudication)
    summary = adjudication_summary(raters, sentences, verdicts)
    resolved = adjudicated_labels(raters, sentences, verdicts)
    label_column = sources[0].label_column
    header, source_rows = read_table(sources[0].path)
    rows = [
        {**row, label_column: resolved[row[SENTENCE_COLUMN]]}
        for row in source_rows
        if row[SENTENCE_COLUMN] in resolved
    ]
    return summary, header, rows


def write_report(report: dict[str, Any], directory: Path) -> Path:
    """Write the deterministic machine-readable JSON report."""

    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_FILENAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def _write_csv(header: Sequence[str], rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write a deterministic CSV with the given header and rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_review_csv(header: Sequence[str], rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write one CSV row per disagreement, ready to open in a spreadsheet."""

    return _write_csv(header, rows, path)


def write_benchmark_csv(header: Sequence[str], rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write the adjudicated benchmark, one row per surviving sentence."""

    return _write_csv(header, rows, path)


def _summary(report: dict[str, Any]) -> str:
    """Render a short human-readable digest of the report."""

    lines = [f"matched rows: {report['matched_rows']}"]
    lines += [
        f"{pair['left']} vs {pair['right']}: "
        f"agreement {pair['observed_agreement']:.4f} kappa {pair['cohen_kappa']:.4f}"
        for pair in report["pairwise"]
    ]
    three = report["three_rater"]
    lines.append(
        f"all raters unanimous: {three['unanimous']}/{three['matched']} "
        f"({three['unanimous_agreement']:.4f}), Fleiss kappa {three['fleiss_kappa']:.4f}"
    )
    lines.append(f"disagreements: {len(report['disagreements'])}")
    adjudication = report["adjudication"]
    lines.append(
        f"adjudicated: {adjudication['adjudicated']}, removed: {adjudication['removed']}, "
        f"final rows: {adjudication['final_rows']}"
    )
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute interrater agreement over labelled CSVs.")
    parser.add_argument("--human", type=Path, default=BENCHMARK_CSV)
    parser.add_argument("--gpt", type=Path, default=GPT_CSV)
    parser.add_argument("--claude", type=Path, default=CLAUDE_CSV)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    sources = (
        RaterSource("human", arguments.human, "label"),
        RaterSource("gpt", arguments.gpt, "llm_label"),
        RaterSource("claude", arguments.claude, "llm_label"),
    )
    try:
        report = build_report(sources)
        header, rows = build_review_rows(sources)
        summary, benchmark_header, benchmark_rows = build_adjudication(sources, arguments.adjudication)
    except InterraterDataError as error:
        print(f"interrater validation failed: {error}", file=sys.stderr)
        return 1
    except AdjudicationError as error:
        print(f"adjudication failed: {error}", file=sys.stderr)
        return 1
    report["adjudication"] = summary
    json_path = write_report(report, arguments.output_directory)
    review_path = write_review_csv(header, rows, arguments.review_csv)
    benchmark_path = write_benchmark_csv(benchmark_header, benchmark_rows, arguments.benchmark_csv)
    print(_summary(report))
    for written in (json_path, review_path, benchmark_path):
        print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
