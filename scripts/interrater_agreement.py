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

from landuse_sentence_relevance.analysis.annotation_csv import read_rater_labels, read_sentence_context
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

BENCHMARK_CSV = Path("results/annotations/benchmark/v2-wikipedia-website-combined.csv")
GPT_CSV = Path("results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv")
CLAUDE_CSV = Path("results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv")

DEFAULT_OUTPUT_DIRECTORY = Path("results/analysis/interrater")
DEFAULT_REVIEW_CSV = Path("docs/data/interrater-disagreements.csv")

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


def write_report(report: dict[str, Any], directory: Path) -> Path:
    """Write the deterministic machine-readable JSON report."""

    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_FILENAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def write_review_csv(header: Sequence[str], rows: Sequence[Mapping[str, str]], path: Path) -> Path:
    """Write one CSV row per disagreement, ready to open in a spreadsheet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


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
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute interrater agreement over labelled CSVs.")
    parser.add_argument("--human", type=Path, default=BENCHMARK_CSV)
    parser.add_argument("--gpt", type=Path, default=GPT_CSV)
    parser.add_argument("--claude", type=Path, default=CLAUDE_CSV)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
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
    except InterraterDataError as error:
        print(f"interrater validation failed: {error}", file=sys.stderr)
        return 1
    json_path = write_report(report, arguments.output_directory)
    review_path = write_review_csv(header, rows, arguments.review_csv)
    print(_summary(report))
    print(f"wrote {json_path}")
    print(f"wrote {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
