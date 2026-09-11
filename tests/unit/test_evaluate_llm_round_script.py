from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_llm_round import main

from landuse_sentence_relevance.analysis.llm_rounds import prepare_round

BENCHMARK = (
    "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
    "A field.,yes,Field,cell-1,1.0,2.0,wikipedia,region-a,https://example.invalid/field\n"
    "A road.,no,Road,cell-2,3.0,4.0,website,region-b,https://example.invalid/road\n"
)


def test_main_evaluates_a_round_and_records_output_provenance(tmp_path: Path, capsys) -> None:
    benchmark = tmp_path / "benchmark.csv"
    prompt = tmp_path / "prompt.md"
    benchmark.write_text(BENCHMARK, encoding="utf-8")
    prompt.write_text("Prompt version 2\n", encoding="utf-8")
    round_directory = tmp_path / "round-02"
    prepare_round(round_directory, benchmark, prompt, round_id="round-02")
    (round_directory / "outputs" / "gpt.csv").write_text(
        "sentence,llm_label\nA field.,yes\nA road.,yes\n", encoding="utf-8"
    )
    (round_directory / "outputs" / "claude.csv").write_text(
        "sentence,llm_label\nA field.,no\nA road.,no\n", encoding="utf-8"
    )

    exit_code = main(
        [
            "--round-directory",
            str(round_directory),
            "--gpt-model",
            "GPT test",
            "--claude-model",
            "Claude test",
            "--gpt-run-at",
            "2026-09-11T10:00:00+02:00",
            "--claude-run-at",
            "2026-09-11T10:01:00+02:00",
        ]
    )

    report = json.loads((round_directory / "analysis" / "agreement.json").read_text(encoding="utf-8"))
    manifest = json.loads((round_directory / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["round_id"] == "round-02"
    assert report["matched_rows"] == 2
    assert (round_directory / "analysis" / "disagreements.csv").is_file()
    assert manifest["status"] == "evaluated"
    assert manifest["outputs"]["gpt"]["model"] == "GPT test"
    assert manifest["outputs"]["claude"]["model"] == "Claude test"
    assert manifest["outputs"]["gpt"]["run_at"] == "2026-09-11T10:00:00+02:00"
    assert manifest["outputs"]["claude"]["run_at"] == "2026-09-11T10:01:00+02:00"
    output = capsys.readouterr().out
    assert "round: round-02\nmatched rows: 2\n" in output
