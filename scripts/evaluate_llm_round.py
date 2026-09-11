from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from landuse_sentence_relevance.analysis.annotation_csv import read_rater_labels
from landuse_sentence_relevance.analysis.interrater import InterraterDataError
from scripts.interrater_agreement import (
    RaterSource,
    build_report,
    build_review_rows,
    file_digest,
    write_report,
    write_review_csv,
)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one GPT/Claude LLM labeling round.")
    parser.add_argument("--round-directory", type=Path, required=True)
    parser.add_argument("--gpt", type=Path)
    parser.add_argument("--claude", type=Path)
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--claude-model", required=True)
    parser.add_argument("--gpt-run-at")
    parser.add_argument("--claude-run-at")
    return parser.parse_args(argv)


def _read_manifest(round_directory: Path) -> dict[str, Any]:
    manifest_path = round_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InterraterDataError(f"could not read round manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("round_id"), str):
        raise InterraterDataError(f"round manifest {manifest_path} is invalid")
    return manifest


def _output_metadata(path: Path, model: str, run_at: str | None) -> dict[str, Any]:
    labels = read_rater_labels(path, path.stem, "llm_label")
    return {
        "path": path.name,
        "rows": len(labels),
        "sha256": file_digest(path),
        "model": model,
        "run_at": run_at,
    }


def _summary(report: dict[str, Any]) -> str:
    lines = [f"round: {report['round_id']}", f"matched rows: {report['matched_rows']}"]
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


def evaluate_round(
    round_directory: Path,
    *,
    gpt_model: str,
    claude_model: str,
    gpt_run_at: str | None = None,
    claude_run_at: str | None = None,
    gpt_path: Path | None = None,
    claude_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate two model outputs against the round's labeled benchmark snapshot."""

    manifest = _read_manifest(round_directory)
    benchmark_path = round_directory / "benchmark.csv"
    gpt_output = gpt_path or round_directory / "outputs" / "gpt.csv"
    claude_output = claude_path or round_directory / "outputs" / "claude.csv"
    sources = (
        RaterSource("human", benchmark_path, "label"),
        RaterSource("gpt", gpt_output, "llm_label"),
        RaterSource("claude", claude_output, "llm_label"),
    )
    report = build_report(sources)
    report["round_id"] = manifest["round_id"]
    report["input"] = manifest["input"]
    report["prompt"] = manifest["prompt"]
    header, rows = build_review_rows(sources)
    analysis_directory = round_directory / "analysis"
    write_report(report, analysis_directory)
    write_review_csv(header, rows, analysis_directory / "disagreements.csv")
    manifest["status"] = "evaluated"
    manifest["outputs"] = {
        "gpt": _output_metadata(gpt_output, gpt_model, gpt_run_at),
        "claude": _output_metadata(claude_output, claude_model, claude_run_at),
    }
    manifest["analysis"] = {
        "agreement": "analysis/agreement.json",
        "disagreements": "analysis/disagreements.csv",
    }
    (round_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        report = evaluate_round(
            arguments.round_directory,
            gpt_model=arguments.gpt_model,
            claude_model=arguments.claude_model,
            gpt_run_at=arguments.gpt_run_at,
            claude_run_at=arguments.claude_run_at,
            gpt_path=arguments.gpt,
            claude_path=arguments.claude,
        )
    except (InterraterDataError, OSError) as error:
        print(f"LLM round evaluation failed: {error}", file=sys.stderr)
        return 1
    print(_summary(report))
    print(f"wrote {arguments.round_directory / 'analysis' / 'agreement.json'}")
    print(f"wrote {arguments.round_directory / 'analysis' / 'disagreements.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
