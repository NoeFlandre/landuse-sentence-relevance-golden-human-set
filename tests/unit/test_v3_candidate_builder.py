from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from landuse_sentence_relevance.bootstrap import build_v3_candidate_pool
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas
from landuse_sentence_relevance.domain.v3_pool import V3_SOURCES
from landuse_sentence_relevance.sources.v3 import V3SourceAdapters
from tests.unit.test_models import make_candidate


@dataclass
class FakeSource:
    rows: tuple[Candidate, ...]
    on_iter: Callable[[], None] | None = None

    def iter_candidates(self) -> Iterable[Candidate]:
        if self.on_iter is not None:
            self.on_iter()
        return iter(self.rows)


def candidate(identifier: str, source: Source, cell: str, latitude: float) -> Candidate:
    return replace(
        make_candidate(identifier),
        source=source,
        source_record_id=identifier,
        h3_cell=cell,
        latitude=latitude,
        longitude=0.0,
    )


def quotas() -> SourceLabelQuotas:
    return SourceLabelQuotas({(source, label): 1 for source in V3_SOURCES for label in (Label.YES, Label.NO)})


def settings(tmp_path: Path, benchmark_path: Path) -> V3Settings:
    return V3Settings(
        data_root=tmp_path,
        benchmark_path=benchmark_path,
        candidate_pool_path=tmp_path / "results" / "pool.json",
        candidate_progress_path=tmp_path / "results" / "progress.json",
        candidate_cells_per_source=2,
        candidate_reservoir_cells_per_source=3,
    )


def adapters(rows_by_source: dict[Source, tuple[Candidate, ...]]) -> V3SourceAdapters:
    return V3SourceAdapters(
        description=FakeSource(rows_by_source[Source.DESCRIPTION]),
        wikipedia=FakeSource(rows_by_source[Source.WIKIPEDIA]),
        website=FakeSource(rows_by_source[Source.WEBSITE]),
    )


def centers_for(rows_by_source: dict[Source, tuple[Candidate, ...]]) -> dict[str, tuple[float, float]]:
    return {row.h3_cell: (row.latitude, row.longitude) for rows in rows_by_source.values() for row in rows}


def make_rows() -> dict[Source, tuple[Candidate, ...]]:
    return {
        source: tuple(
            candidate(f"{source.value}-{index}", source, f"{source.value}-{index}", float(index))
            for index in range(3)
        )
        for source in V3_SOURCES
    }


def test_build_v3_candidate_pool_streams_each_source_and_saves_a_reusable_pool(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n",
        encoding="utf-8",
    )
    rows = make_rows()
    settings_ = settings(tmp_path, benchmark)
    centers = centers_for(rows)

    pool = build_v3_candidate_pool(
        settings_,
        adapters=adapters(rows),
        center_of_cell=centers.__getitem__,
        quotas=quotas(),
    )

    assert len(pool.candidates) == 6
    assert {row.source for row in pool.candidates} == set(V3_SOURCES)
    assert len(set(pool.cells)) == len(pool.cells)
    assert settings_.candidate_pool_path.is_file()


def test_build_v3_candidate_pool_resumes_after_a_completed_source_pass(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n",
        encoding="utf-8",
    )
    rows = make_rows()
    settings_ = settings(tmp_path, benchmark)
    centers = centers_for(rows)
    failures = {Source.WEBSITE: True}

    def fail_website() -> None:
        if failures[Source.WEBSITE]:
            raise RuntimeError("interrupted website stream")

    interrupted = V3SourceAdapters(
        description=FakeSource(rows[Source.DESCRIPTION]),
        wikipedia=FakeSource(rows[Source.WIKIPEDIA]),
        website=FakeSource(rows[Source.WEBSITE], fail_website),
    )

    with pytest.raises(RuntimeError, match="interrupted website stream"):
        build_v3_candidate_pool(
            settings_,
            adapters=interrupted,
            center_of_cell=centers.__getitem__,
            quotas=quotas(),
        )

    checkpoint = json.loads(settings_.candidate_progress_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_sources"] == ["wikipedia"]
    assert len(checkpoint["candidates"]) == 3

    failures[Source.WEBSITE] = False

    def unexpected_completed_source() -> None:
        raise AssertionError("completed V3 sources must not be streamed again")

    resumed = build_v3_candidate_pool(
        settings_,
        adapters=V3SourceAdapters(
            description=FakeSource(rows[Source.DESCRIPTION]),
            wikipedia=FakeSource(rows[Source.WIKIPEDIA], unexpected_completed_source),
            website=FakeSource(rows[Source.WEBSITE]),
        ),
        center_of_cell=centers.__getitem__,
        quotas=quotas(),
    )

    assert len(resumed.candidates) == 6


def test_build_v3_candidate_pool_reuses_a_saved_pool_without_streaming_again(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n",
        encoding="utf-8",
    )
    rows = make_rows()
    settings_ = settings(tmp_path, benchmark)
    centers = centers_for(rows)
    build_v3_candidate_pool(
        settings_,
        adapters=adapters(rows),
        center_of_cell=centers.__getitem__,
        quotas=quotas(),
    )

    def unexpected_stream() -> None:
        raise AssertionError("saved V3 pool must be reused")

    reused = build_v3_candidate_pool(
        settings_,
        adapters=V3SourceAdapters(
            description=FakeSource((), unexpected_stream),
            wikipedia=FakeSource((), unexpected_stream),
            website=FakeSource((), unexpected_stream),
        ),
        center_of_cell=centers.__getitem__,
        quotas=quotas(),
    )

    assert reused == build_v3_candidate_pool(
        settings_,
        adapters=adapters(rows),
        center_of_cell=centers.__getitem__,
        quotas=quotas(),
    )
