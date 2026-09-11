from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from scripts.prepare_llm_round import main

from landuse_sentence_relevance.analysis.llm_rounds import (
    RoundPreparationError,
    prepare_round,
)

BENCHMARK = (
    "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
    "A field.,yes,Field,cell-1,1.0,2.0,wikipedia,region-a,https://example.invalid/field\n"
    "A road.,no,Road,cell-2,3.0,4.0,website,region-b,https://example.invalid/road\n"
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    benchmark = tmp_path / "benchmark.csv"
    prompt = tmp_path / "prompt.md"
    benchmark.write_text(BENCHMARK, encoding="utf-8")
    prompt.write_text("Prompt version 2\n", encoding="utf-8")
    return benchmark, prompt


def test_prepare_round_writes_a_blinded_input_and_manifest(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    round_directory = tmp_path / "round-02"

    manifest = prepare_round(round_directory, benchmark, prompt, round_id="round-02")

    input_path = round_directory / "input.csv"
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "sentence",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    ]
    assert rows == [
        {
            "sentence": "A field.",
            "polygon_name": "Field",
            "h3_cell": "cell-1",
            "latitude": "1.0",
            "longitude": "2.0",
            "source": "wikipedia",
            "region": "region-a",
            "source_url": "https://example.invalid/field",
        },
        {
            "sentence": "A road.",
            "polygon_name": "Road",
            "h3_cell": "cell-2",
            "latitude": "3.0",
            "longitude": "4.0",
            "source": "website",
            "region": "region-b",
            "source_url": "https://example.invalid/road",
        },
    ]
    assert (round_directory / "benchmark.csv").read_bytes() == benchmark.read_bytes()
    assert (round_directory / "prompt.md").read_bytes() == prompt.read_bytes()
    assert (round_directory / "outputs").is_dir()
    assert (round_directory / "analysis").is_dir()
    assert json.loads((round_directory / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["round_id"] == "round-02"
    assert manifest["status"] == "prepared"
    assert manifest["benchmark"]["rows"] == 2
    assert manifest["input"]["rows"] == 2
    assert manifest["outputs"] == {"gpt": None, "claude": None}


def test_prepare_round_writes_a_complete_round_contract(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    benchmark.write_text(BENCHMARK.replace("A field.", "A field — région."), encoding="utf-8")
    prompt.write_text("Prompt version 2 — région\n", encoding="utf-8")
    round_directory = tmp_path / "round-02"

    manifest = prepare_round(round_directory, benchmark, prompt, round_id="round-02-é")

    expected_input = (
        "sentence,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A field — région.,Field,cell-1,1.0,2.0,wikipedia,region-a,https://example.invalid/field\n"
        "A road.,Road,cell-2,3.0,4.0,website,region-b,https://example.invalid/road\n"
    ).encode()
    assert (round_directory / "input.csv").read_bytes() == expected_input

    expected_manifest = {
        "schema_version": 1,
        "round_id": "round-02-é",
        "status": "prepared",
        "benchmark": {
            "path": "benchmark.csv",
            "rows": 2,
            "sha256": hashlib.sha256((round_directory / "benchmark.csv").read_bytes()).hexdigest(),
        },
        "input": {
            "path": "input.csv",
            "rows": 2,
            "sha256": hashlib.sha256((round_directory / "input.csv").read_bytes()).hexdigest(),
        },
        "prompt": {
            "path": "prompt.md",
            "sha256": hashlib.sha256((round_directory / "prompt.md").read_bytes()).hexdigest(),
        },
        "outputs": {"gpt": None, "claude": None},
        "analysis": {"path": "analysis/agreement.json"},
    }
    assert manifest == expected_manifest
    assert (round_directory / "manifest.json").read_bytes() == (
        json.dumps(expected_manifest, indent=2) + "\n"
    ).encode()
    assert {path.name for path in round_directory.iterdir()} == {
        "analysis",
        "benchmark.csv",
        "input.csv",
        "manifest.json",
        "outputs",
        "prompt.md",
    }


def test_prepare_round_refuses_to_overwrite_an_existing_round(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    round_directory = tmp_path / "round-02"
    prepare_round(round_directory, benchmark, prompt, round_id="round-02")

    with pytest.raises(RoundPreparationError, match="already exists"):
        prepare_round(round_directory, benchmark, prompt, round_id="round-02")


def test_prepare_round_rejects_a_benchmark_without_human_labels(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    benchmark.write_text(BENCHMARK.replace(",label,", ",human_label,"), encoding="utf-8")

    with pytest.raises(RoundPreparationError, match="label"):
        prepare_round(tmp_path / "round-02", benchmark, prompt, round_id="round-02")


def test_main_prepares_a_round_from_command_line_arguments(tmp_path: Path, capsys) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    output_root = tmp_path / "evaluations"

    exit_code = main(
        [
            "--round-id",
            "round-02",
            "--benchmark",
            str(benchmark),
            "--prompt-file",
            str(prompt),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert (output_root / "round-02" / "manifest.json").is_file()
    assert "prepared round-02" in capsys.readouterr().out


def test_prepare_round_rejects_a_missing_benchmark(tmp_path: Path) -> None:
    _, prompt = _write_inputs(tmp_path)

    with pytest.raises(RoundPreparationError, match="could not read benchmark"):
        prepare_round(tmp_path / "round-02", tmp_path / "missing.csv", prompt, round_id="round-02")


def test_prepare_round_rejects_an_invalid_benchmark_label(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    benchmark.write_text(BENCHMARK.replace(",yes,", ",maybe,"), encoding="utf-8")

    with pytest.raises(RoundPreparationError, match="other than 'yes' or 'no'") as error:
        prepare_round(tmp_path / "round-02", benchmark, prompt, round_id="round-02")

    assert f"benchmark {benchmark}" in str(error.value)


def test_prepare_round_rejects_duplicate_benchmark_sentences(tmp_path: Path) -> None:
    benchmark, prompt = _write_inputs(tmp_path)
    benchmark.write_text(BENCHMARK.replace("A road.", "A field."), encoding="utf-8")

    with pytest.raises(RoundPreparationError, match="duplicate sentences") as error:
        prepare_round(tmp_path / "round-02", benchmark, prompt, round_id="round-02")

    assert f"benchmark {benchmark}" in str(error.value)


def test_prepare_round_rejects_a_missing_prompt(tmp_path: Path) -> None:
    benchmark, _ = _write_inputs(tmp_path)

    with pytest.raises(RoundPreparationError, match="prompt file does not exist"):
        prepare_round(tmp_path / "round-02", benchmark, tmp_path / "missing.md", round_id="round-02")
