from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.interrater_agreement import (
    DEFAULT_ADJUDICATION_CSV,
    DEFAULT_BENCHMARK_CSV,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_REVIEW_CSV,
    DEFAULT_SOURCES,
    RaterSource,
    _parse_arguments,
    build_adjudication,
    build_report,
    build_review_rows,
    file_digest,
    main,
    write_benchmark_csv,
    write_report,
    write_review_csv,
)

from landuse_sentence_relevance.analysis.adjudication import AdjudicationError
from landuse_sentence_relevance.analysis.interrater import InterraterDataError

HUMAN_CSV = (
    "sentence,label,source,region,polygon_name,source_url\n"
    "A field.,yes,wikipedia,fiji,Rabi,https://example.invalid/field\n"
    "A meeting.,no,wikipedia,fiji,Rabi,https://example.invalid/meeting\n"
    "A road.,yes,website,peru,Cusco,https://example.invalid/road\n"
)
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
    assert DEFAULT_SOURCES[0].path == Path("results/evaluations/round-01/human.csv")
    assert DEFAULT_SOURCES[1].path == Path("results/evaluations/round-01/outputs/gpt.csv")
    assert DEFAULT_SOURCES[2].path == Path("results/evaluations/round-01/outputs/claude.csv")


def test_default_output_directory_is_the_machine_readable_analysis_folder() -> None:
    assert Path("results/evaluations/round-01/analysis") == DEFAULT_OUTPUT_DIRECTORY


def test_default_review_csv_lives_in_the_committed_data_tree() -> None:
    assert Path("data/interrater/disagreements.csv") == DEFAULT_REVIEW_CSV


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


def test_build_review_rows_reads_context_from_the_first_source(
    sources: tuple[RaterSource, ...],
) -> None:
    header, rows = build_review_rows(sources)

    assert header == (
        "minority_rater",
        "minority_label",
        "human",
        "gpt",
        "claude",
        "sentence",
        "source",
        "region",
        "polygon_name",
        "source_url",
    )
    assert [(row["minority_rater"], row["sentence"]) for row in rows] == [
        ("claude", "A meeting."),
        ("gpt", "A road."),
    ]
    assert rows[1]["polygon_name"] == "Cusco"
    assert rows[1]["source"] == "website"


def test_write_report_writes_deterministic_json(sources: tuple[RaterSource, ...], tmp_path: Path) -> None:
    report = build_report(sources)
    directory = tmp_path / "out" / "interrater"

    json_path = write_report(report, directory)

    assert json_path == directory / "agreement.json"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert json_path.read_text(encoding="utf-8").endswith("\n")


def test_write_review_csv_writes_a_header_and_one_row_per_disagreement(
    sources: tuple[RaterSource, ...], tmp_path: Path
) -> None:
    header, rows = build_review_rows(sources)
    path = tmp_path / "review" / "disagreements.csv"

    assert write_review_csv(header, rows, path) == path
    assert path.read_text(encoding="utf-8").splitlines() == [
        "minority_rater,minority_label,human,gpt,claude,sentence,source,region,polygon_name,source_url",
        "claude,yes,no,no,yes,A meeting.,wikipedia,fiji,Rabi,https://example.invalid/meeting",
        "gpt,no,yes,no,yes,A road.,website,peru,Cusco,https://example.invalid/road",
    ]


def test_write_review_csv_is_byte_identical_when_run_twice(
    sources: tuple[RaterSource, ...], tmp_path: Path
) -> None:
    header, rows = build_review_rows(sources)

    first = write_review_csv(header, rows, tmp_path / "first.csv")
    second = write_review_csv(header, rows, tmp_path / "second.csv")

    assert first.read_bytes() == second.read_bytes()


def test_main_writes_the_report_and_the_review_csv(
    sources: tuple[RaterSource, ...],
    adjudication: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "report"
    review = tmp_path / "review" / "disagreements.csv"
    benchmark = tmp_path / "benchmark.csv"

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
            "--review-csv",
            str(review),
            "--adjudication",
            str(adjudication),
            "--benchmark-csv",
            str(benchmark),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert (directory / "agreement.json").is_file()
    assert review.is_file()
    assert benchmark.is_file()
    assert "matched rows: 3" in output
    assert "disagreements: 2" in output


def test_main_reports_a_validation_failure_without_writing_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "human.csv").write_text(HUMAN_CSV, encoding="utf-8")
    (tmp_path / "gpt.csv").write_text("sentence,llm_label\nA field.,yes\n", encoding="utf-8")
    (tmp_path / "claude.csv").write_text(CLAUDE_CSV, encoding="utf-8")
    directory = tmp_path / "report"
    review = tmp_path / "review" / "disagreements.csv"

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
            "--review-csv",
            str(review),
        ]
    )

    assert exit_code == 1
    assert not directory.exists()
    assert not review.exists()
    assert "interrater validation failed" in capsys.readouterr().err


ADJUDICATION_CSV = (
    "minority_rater,minority_label,human,gpt,claude,final_label,sentence,source,region,polygon_name,source_url\n"
    "claude,yes,no,no,yes,no,A meeting.,wikipedia,fiji,Rabi,https://example.invalid/meeting\n"
    "gpt,no,yes,no,yes,remove,A road.,website,peru,Cusco,https://example.invalid/road\n"
)


@pytest.fixture
def adjudication(tmp_path: Path) -> Path:
    path = tmp_path / "adjudication.csv"
    path.write_text(ADJUDICATION_CSV, encoding="utf-8")
    return path


def test_default_adjudication_csv_lives_in_the_committed_data_tree() -> None:
    assert Path("data/interrater/adjudication.csv") == DEFAULT_ADJUDICATION_CSV


def test_build_adjudication_summarises_and_resolves_the_benchmark(
    sources: tuple[RaterSource, ...], adjudication: Path
) -> None:
    summary, header, rows = build_adjudication(sources, adjudication)

    assert summary["matched_rows"] == 3
    assert summary["adjudicated"] == 2
    assert summary["removed"] == 1
    assert summary["final_rows"] == 2
    assert header == ("sentence", "label", "source", "region", "polygon_name", "source_url")
    assert [(row["sentence"], row["label"]) for row in rows] == [
        ("A field.", "yes"),
        ("A meeting.", "no"),
    ]
    assert rows[1]["polygon_name"] == "Rabi"


def test_build_adjudication_fails_loudly_on_a_missing_verdict(
    sources: tuple[RaterSource, ...], tmp_path: Path
) -> None:
    partial = tmp_path / "partial.csv"
    partial.write_text(
        "sentence,final_label\nA meeting.,no\n",
        encoding="utf-8",
    )

    with pytest.raises(AdjudicationError, match="unadjudicated disagreements"):
        build_adjudication(sources, partial)


def test_write_benchmark_csv_writes_the_resolved_rows(
    sources: tuple[RaterSource, ...], adjudication: Path, tmp_path: Path
) -> None:
    _, header, rows = build_adjudication(sources, adjudication)
    path = tmp_path / "out" / "benchmark.csv"

    assert write_benchmark_csv(header, rows, path) == path
    assert path.read_text(encoding="utf-8").splitlines() == [
        "sentence,label,source,region,polygon_name,source_url",
        "A field.,yes,wikipedia,fiji,Rabi,https://example.invalid/field",
        "A meeting.,no,wikipedia,fiji,Rabi,https://example.invalid/meeting",
    ]


def test_default_benchmark_csv_is_the_committed_final_dataset() -> None:
    assert Path("data/benchmark/v2-adjudicated.csv") == DEFAULT_BENCHMARK_CSV


def test_build_adjudication_keeps_the_human_schema_and_row_order(
    sources: tuple[RaterSource, ...], adjudication: Path
) -> None:
    _, header, rows = build_adjudication(sources, adjudication)
    source_header = HUMAN_CSV.splitlines()[0].split(",")

    assert list(header) == source_header
    assert all(list(row) == source_header for row in rows)


def test_main_records_the_adjudication_and_writes_the_resolved_benchmark(
    sources: tuple[RaterSource, ...],
    adjudication: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "report"
    benchmark = tmp_path / "benchmark.csv"

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
            "--review-csv",
            str(tmp_path / "review.csv"),
            "--adjudication",
            str(adjudication),
            "--benchmark-csv",
            str(benchmark),
        ]
    )

    report = json.loads((directory / "agreement.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["adjudication"]["final_rows"] == 2
    assert benchmark.is_file()
    assert "final rows: 2" in capsys.readouterr().out


def test_main_reports_an_adjudication_failure(
    sources: tuple[RaterSource, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    partial = tmp_path / "partial.csv"
    partial.write_text("sentence,final_label\nA meeting.,no\n", encoding="utf-8")
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
            "--review-csv",
            str(tmp_path / "review.csv"),
            "--adjudication",
            str(partial),
            "--benchmark-csv",
            str(tmp_path / "benchmark.csv"),
        ]
    )

    assert exit_code == 1
    assert not directory.exists()
    assert "adjudication failed" in capsys.readouterr().err


ROOT = Path(__file__).parents[2]
COMMITTED_DATA_ROOT = ROOT / "data"
REBUILD_ROOT = ROOT / "results" / "interrater-rebuild"
# The fresh-clone command in docs/interrater-agreement.md. tests/unit/test_public_project_files.py checks
# that the doc still gives these arguments. This file must not read the doc: the mutation copy has no docs/.
FRESH_CLONE_ARGUMENTS = [
    "--human",
    "data/provenance/round-01/human.csv",
    "--gpt",
    "data/provenance/round-01/outputs/gpt.csv",
    "--claude",
    "data/provenance/round-01/outputs/claude.csv",
    "--adjudication",
    "data/interrater/adjudication.csv",
    "--output-directory",
    "results/interrater-rebuild/round-01",
    "--review-csv",
    "results/interrater-rebuild/round-01/disagreements.csv",
    "--benchmark-csv",
    "results/interrater-rebuild/round-01/v2-adjudicated.csv",
]
RECIPE_INPUTS = {
    "human": HUMAN_CSV,
    "gpt": GPT_CSV,
    "claude": CLAUDE_CSV,
    "adjudication": ADJUDICATION_CSV,
}
RECIPE_OUTPUTS = ("output_directory", "review_csv", "benchmark_csv")


def _recipe_violations(arguments: list[str]) -> list[str]:
    """Return every way the recipe breaks the read and write contract.

    Inputs must be files under data/, which git commits and the mutation gate copies. The check does not
    call git: the mutation copy sits in the git-ignored mutants/ directory, where git lists no tracked files.
    """

    parsed = _parse_arguments(arguments)
    violations = []
    for name in RECIPE_INPUTS:
        path = getattr(parsed, name)
        source = (ROOT / path).resolve()
        if not (source.is_file() and source.is_relative_to(COMMITTED_DATA_ROOT)):
            violations.append(f"--{name} must name a committed file under data/, not {path}")
    for name in RECIPE_OUTPUTS:
        flag = name.replace("_", "-")
        destination = (ROOT / getattr(parsed, name)).resolve()
        if not destination.is_relative_to(REBUILD_ROOT):
            violations.append(f"--{flag} must write under results/interrater-rebuild/, not {destination}")
    return violations


def _data_files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in (root / "data").rglob("*") if path.is_file()}


def test_fresh_clone_recipe_obeys_the_read_and_write_contract() -> None:
    assert _recipe_violations(FRESH_CLONE_ARGUMENTS) == []


@pytest.mark.parametrize(
    ("flag", "mutant", "violation"),
    [
        (
            "--human",
            "results/evaluations/round-01/human.csv",
            "--human must name a committed file under data/, not results/evaluations/round-01/human.csv",
        ),
        (
            "--output-directory",
            "results/evaluations/round-01/analysis",
            "--output-directory must write under results/interrater-rebuild/",
        ),
        (
            "--benchmark-csv",
            "data/benchmark/v2-adjudicated.csv",
            "--benchmark-csv must write under results/interrater-rebuild/",
        ),
    ],
)
def test_contract_check_rejects_each_mutated_recipe(flag: str, mutant: str, violation: str) -> None:
    arguments = list(FRESH_CLONE_ARGUMENTS)
    arguments[arguments.index(flag) + 1] = mutant

    assert any(violation in found for found in _recipe_violations(arguments))


def test_fresh_clone_recipe_reads_committed_inputs_and_writes_outside_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = list(FRESH_CLONE_ARGUMENTS)
    assert _recipe_violations(arguments) == []
    parsed = _parse_arguments(arguments)

    clone = tmp_path / "clone"
    for name, content in RECIPE_INPUTS.items():
        path = clone / getattr(parsed, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    committed = _data_files(clone)
    monkeypatch.chdir(clone)

    assert main(arguments) == 0
    assert "final rows: 2" in capsys.readouterr().out
    assert _data_files(clone) == committed
    assert (clone / parsed.output_directory / "agreement.json").is_file()
    assert (clone / parsed.review_csv).is_file()
    assert (clone / parsed.benchmark_csv).is_file()
