from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.interrater_agreement import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SOURCES,
    RaterSource,
    build_report,
    file_digest,
    main,
    write_report,
)

from landuse_sentence_relevance.analysis.interrater import InterraterDataError

HUMAN_CSV = "sentence,label\nA field.,yes\nA meeting.,no\nA road.,yes\n"
GPT_CSV = "sentence,llm_label\nA field.,yes\nA meeting.,no\nA road.,no\n"
CLAUDE_CSV = "sentence,llm_label\nA field.,yes\nA meeting.,yes\nA road.,yes\n"


@pytest.fixture
def sources(tmp_path: Path) -> tuple[RaterSource, ...]:
    (tmp_path / "human.csv").write_text(HUMAN_CSV, encoding="utf-8")
    (tmp_path / "gpt.csv").write_text(GPT_CSV, encoding="utf-8")
    (tmp_path / "claude.csv").write_text(CLAUDE_CSV, encoding="utf-8")
    return (
        RaterSource("human", tmp_path / "human.csv", "label"),
        RaterSource("gpt", tmp_path / "gpt.csv", "llm_label"),
        RaterSource("claude", tmp_path / "claude.csv", "llm_label"),
    )


def test_default_sources_point_at_the_human_benchmark_and_both_llm_runs() -> None:
    assert [source.name for source in DEFAULT_SOURCES] == ["human", "gpt", "claude"]
    assert [source.label_column for source in DEFAULT_SOURCES] == ["label", "llm_label", "llm_label"]
    assert DEFAULT_SOURCES[0].path == Path("results/annotations/benchmark/v2-wikipedia-website-combined.csv")
    assert DEFAULT_SOURCES[1].path == Path(
        "results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv"
    )
    assert DEFAULT_SOURCES[2].path == Path(
        "results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv"
    )


def test_default_output_directory_is_the_machine_readable_analysis_folder() -> None:
    assert Path("results/analysis/interrater") == DEFAULT_OUTPUT_DIRECTORY


def test_file_digest_is_the_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "digest.txt"
    path.write_text("abc", encoding="utf-8")

    assert file_digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_build_report_records_sources_and_metrics(sources: tuple[RaterSource, ...]) -> None:
    report = build_report(sources)

    assert report["matched_rows"] == 3
    assert report["raters"] == ["claude", "gpt", "human"]
    assert [source["name"] for source in report["sources"]] == ["human", "gpt", "claude"]
    assert report["sources"][0]["rows"] == 3
    assert report["sources"][0]["label_column"] == "label"
    assert report["sources"][0]["sha256"] == file_digest(sources[0].path)
    assert report["sources"][0]["path"] == sources[0].path.as_posix()


def test_build_report_fails_loudly_on_misaligned_sources(tmp_path: Path) -> None:
    (tmp_path / "human.csv").write_text(HUMAN_CSV, encoding="utf-8")
    (tmp_path / "gpt.csv").write_text("sentence,llm_label\nA field.,yes\n", encoding="utf-8")
    misaligned = (
        RaterSource("human", tmp_path / "human.csv", "label"),
        RaterSource("gpt", tmp_path / "gpt.csv", "llm_label"),
    )

    with pytest.raises(InterraterDataError, match="does not align"):
        build_report(misaligned)


def test_write_report_writes_deterministic_json_and_a_disagreement_csv(
    sources: tuple[RaterSource, ...], tmp_path: Path
) -> None:
    report = build_report(sources)
    directory = tmp_path / "out" / "interrater"

    json_path, csv_path = write_report(report, directory)

    assert json_path == directory / "agreement.json"
    assert csv_path == directory / "disagreements.csv"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "sentence,claude,gpt,human",
        "A meeting.,yes,no,no",
        "A road.,yes,no,yes",
    ]


def test_write_report_is_byte_identical_when_run_twice(
    sources: tuple[RaterSource, ...], tmp_path: Path
) -> None:
    first_json, first_csv = write_report(build_report(sources), tmp_path / "first")
    second_json, second_csv = write_report(build_report(sources), tmp_path / "second")

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()


def test_main_writes_the_report_and_reports_success(
    sources: tuple[RaterSource, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "report"

    exit_code = main(
        [
            "--human",
            str(sources[0].path),
            "--gpt",
            str(sources[1].path),
            "--claude",
            str(sources[2].path),
            "--output-directory",
            str(directory),
        ]
    )

    assert exit_code == 0
    assert (directory / "agreement.json").is_file()
    assert "matched rows: 3" in capsys.readouterr().out


def test_main_reports_a_validation_failure_without_writing_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "human.csv").write_text(HUMAN_CSV, encoding="utf-8")
    (tmp_path / "gpt.csv").write_text("sentence,llm_label\nA field.,yes\n", encoding="utf-8")
    (tmp_path / "claude.csv").write_text(CLAUDE_CSV, encoding="utf-8")
    directory = tmp_path / "report"

    exit_code = main(
        [
            "--human",
            str(tmp_path / "human.csv"),
            "--gpt",
            str(tmp_path / "gpt.csv"),
            "--claude",
            str(tmp_path / "claude.csv"),
            "--output-directory",
            str(directory),
        ]
    )

    assert exit_code == 1
    assert not directory.exists()
    assert "interrater validation failed" in capsys.readouterr().err
