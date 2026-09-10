from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.analysis.annotation_csv import SENTENCE_COLUMN, read_rater_labels
from landuse_sentence_relevance.analysis.interrater import InterraterDataError, interrater_report

BENCHMARK_CSV = Path("results/annotations/benchmark/v2-wikipedia-website-combined.csv")
GPT_CSV = Path("results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv")
CLAUDE_CSV = Path("results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv")

DEFAULT_OUTPUT_DIRECTORY = Path("results/analysis/interrater")

JSON_FILENAME = "agreement.json"
DISAGREEMENT_FILENAME = "disagreements.csv"


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


def build_report(sources: Sequence[RaterSource]) -> dict[str, Any]:
    """Read every source, align it, and return the agreement report plus provenance."""

    raters = {
        source.name: read_rater_labels(source.path, source.name, source.label_column) for source in sources
    }
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


def write_report(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Write the deterministic JSON report and its companion disagreement CSV."""

    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_FILENAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = directory / DISAGREEMENT_FILENAME
    names = list(report["raters"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([SENTENCE_COLUMN, *names])
        for entry in report["disagreements"]:
            writer.writerow([entry["sentence"], *(entry["labels"][name] for name in names)])
    return json_path, csv_path


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
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute interrater agreement over labelled CSVs.")
    parser.add_argument("--human", type=Path, default=BENCHMARK_CSV)
    parser.add_argument("--gpt", type=Path, default=GPT_CSV)
    parser.add_argument("--claude", type=Path, default=CLAUDE_CSV)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
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
    except InterraterDataError as error:
        print(f"interrater validation failed: {error}", file=sys.stderr)
        return 1
    json_path, csv_path = write_report(report, arguments.output_directory)
    print(_summary(report))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
