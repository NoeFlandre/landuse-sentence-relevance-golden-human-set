from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from typing import cast

import pytest

from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_seed_plan

_REPOSITORY_ROOT = Path(__file__).parents[3]


def test_v2_seed_loader_reserves_excluded_rows_and_reports_the_v3_shortfall(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "wiki yes,yes,Wiki yes,wiki-yes,1,2,wikipedia,region,https://example.test/wiki-yes\n"
        "wiki no,no,Wiki no,wiki-no,3,4,wikipedia,region,https://example.test/wiki-no\n"
        "wiki surplus,no,Wiki surplus,wiki-surplus,5,6,wikipedia,region,https://example.test/wiki-surplus\n"
        "web yes,yes,Web yes,web-yes,7,8,website,region,https://example.test/web-yes\n"
        "web no,no,Web no,web-no,9,10,website,region,https://example.test/web-no\n",
        encoding="utf-8",
    )

    plan = load_v2_seed_plan(
        benchmark,
        quotas=balanced_quotas(V3_SOURCES, rows_per_source_label=1),
        seed="test-seed",
    )

    assert len(plan.reused) == 4
    assert len(plan.excluded) == 1
    assert [annotation.candidate.candidate_id for annotation in plan.reused] == [
        "v2:000000",
        "v2:000001",
        "v2:000003",
        "v2:000004",
    ]
    assert plan.excluded[0].candidate.candidate_id == "v2:000002"
    assert plan.reserved_cells == frozenset({"wiki-yes", "wiki-no", "wiki-surplus", "web-yes", "web-no"})
    assert plan.remaining == {
        (Source.DESCRIPTION, Label.YES): 1,
        (Source.DESCRIPTION, Label.NO): 1,
    }


def test_v2_seed_loader_rejects_duplicate_h3_cells(tmp_path: Path) -> None:
    benchmark = tmp_path / "duplicate-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "first,yes,First,duplicate-cell,1,2,wikipedia,region,https://example.test/first\n"
        "second,no,Second,duplicate-cell,3,4,website,region,https://example.test/second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} contains a duplicate H3 cell"


def test_v2_seed_loader_rejects_an_empty_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "empty-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no rows"):
        load_v2_seed_plan(benchmark, seed="test-seed")


def test_v2_seed_loader_rejects_missing_columns_with_a_precise_error(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-column-v2.csv"
    benchmark.write_text(
        "sentence,label\na sentence,yes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == (
        f"{benchmark} is missing benchmark columns: "
        "['h3_cell', 'latitude', 'longitude', 'polygon_name', 'region', 'source', 'source_url']"
    )


def test_v2_seed_loader_preserves_csv_newlines_and_utf8_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "multiline-v2.csv"
    benchmark.write_bytes(
        b"sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\r\n"
        b'"Ligne \xc3\xa9\r\navec retour",yes,Place,cell-1,1,2,wikipedia,Region,https://example.test/1\r\n'
    )
    original_open = Path.open

    def checked_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> TextIOWrapper:
        if self == benchmark:
            assert newline == ""
            assert encoding == "utf-8"
        return cast(TextIOWrapper, original_open(self, mode, buffering, encoding, errors, newline))

    monkeypatch.setattr(Path, "open", checked_open)

    plan = load_v2_seed_plan(benchmark, seed="test-seed")

    candidate = plan.reused[0].candidate
    assert candidate.sentence == "Ligne é\r\navec retour"
    assert candidate.place_name == "Place"
    assert candidate.region == "Region"
    assert candidate.source_url == "https://example.test/1"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "source",
            "description",
            "contains a non-V2 source: description",
        ),
        (
            "label",
            "maybe",
            "has an invalid label value: 'maybe'",
        ),
    ],
)
def test_v2_seed_loader_reports_invalid_enum_rows_precisely(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    benchmark = tmp_path / f"invalid-{field}-v2.csv"
    source = "wikipedia"
    label = "yes"
    if field == "source":
        source = value
    else:
        label = value
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        f"first,yes,First,cell-1,1,2,wikipedia,region,https://example.test/1\n"
        f"second,{label},Second,cell-2,3,4,{source},region,https://example.test/2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    if field == "source":
        assert str(error.value) == f"{benchmark} row 3 {expected}"
    else:
        assert str(error.value) == f"{benchmark} has an invalid label value: 'maybe'"


def test_v2_seed_loader_reports_an_unknown_source_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "unknown-source-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,cell-1,1,2,not-a-source,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has an invalid source value: 'not-a-source'"


def test_v2_seed_loader_reports_a_missing_source_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-source-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,cell-1,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has an invalid source value: None"


def test_v2_seed_loader_preserves_all_candidate_provenance_fields(tmp_path: Path) -> None:
    benchmark = tmp_path / "provenance-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Named place,cell-1,12.5,-3.25,wikipedia,Named region,https://example.test/1\n",
        encoding="utf-8",
    )

    annotation = load_v2_seed_plan(benchmark, seed="test-seed").reused[0]
    candidate = annotation.candidate

    assert annotation.label is Label.YES
    assert candidate.candidate_id == "v2:000000"
    assert candidate.sentence == "A sentence"
    assert candidate.source is Source.WIKIPEDIA
    assert candidate.source_record_id == "v2:000000"
    assert candidate.source_field == "v2_benchmark"
    assert candidate.h3_cell == "cell-1"
    assert candidate.h3_resolution == 3
    assert candidate.latitude == 12.5
    assert candidate.longitude == -3.25
    assert candidate.place_name == "Named place"
    assert candidate.region == "Named region"
    assert candidate.source_url == "https://example.test/1"
    assert candidate.language == "en"


def test_v2_seed_loader_reports_missing_required_text_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-sentence-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        ",yes,Place,cell-1,1,2,wikipedia,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has a row missing sentence"


def test_v2_seed_loader_reports_missing_h3_cell_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-h3-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,,1,2,wikipedia,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has a row missing h3_cell"


def test_v2_seed_loader_reports_missing_float_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-latitude-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,cell-1,,2,wikipedia,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has a row missing latitude"


def test_v2_seed_loader_reports_missing_longitude_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing-longitude-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,cell-1,1,,wikipedia,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has a row missing longitude"


def test_v2_seed_loader_reports_invalid_float_with_the_benchmark_path(tmp_path: Path) -> None:
    benchmark = tmp_path / "invalid-latitude-v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "A sentence,yes,Place,cell-1,not-a-number,2,wikipedia,region,https://example.test/1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_v2_seed_plan(benchmark, seed="test-seed")

    assert str(error.value) == f"{benchmark} has an invalid latitude value: 'not-a-number'"


def test_real_v2_benchmark_seed_arithmetic_matches_the_v3_contract() -> None:
    plan = load_v2_seed_plan(
        _REPOSITORY_ROOT / "data/benchmark/v2-adjudicated.csv",
        quotas=V3_QUOTAS,
        seed="landuse-sentence-relevance-golden-human-set-v3",
    )

    assert len(plan.reused) == 153
    assert len(plan.excluded) == 1
    assert len(plan.reserved_cells) == 154
    assert plan.remaining == {
        (Source.WIKIPEDIA, Label.YES): 1,
        (Source.WEBSITE, Label.YES): 19,
        (Source.WEBSITE, Label.NO): 27,
        (Source.DESCRIPTION, Label.YES): 50,
        (Source.DESCRIPTION, Label.NO): 50,
    }
