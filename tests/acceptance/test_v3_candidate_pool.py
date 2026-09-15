from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from landuse_sentence_relevance.bootstrap import build_v3_candidate_pool
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Label, Source
from landuse_sentence_relevance.domain.profile import SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.v3_pool import V3_SOURCES, preflight_v3_candidate_pool
from landuse_sentence_relevance.sources.v3 import V3SourceAdapters
from landuse_sentence_relevance.storage.v3_candidate_pool import load_v2_benchmark
from tests.unit.test_models import make_candidate

scenarios("features/v3_candidate_pool.feature")
pytestmark = pytest.mark.acceptance


@dataclass
class FakeSource:
    rows: tuple[Candidate, ...]
    on_iter: Callable[[], None] | None = None

    def iter_candidates(self) -> tuple[Candidate, ...]:
        if self.on_iter is not None:
            self.on_iter()
        return self.rows


@dataclass
class V3Context:
    settings: V3Settings
    adapters: V3SourceAdapters
    centers: dict[str, tuple[float, float]]
    pool: FinalizedCandidatePool | None = None


def _candidate(identifier: str, source: Source, cell: str, latitude: float) -> Candidate:
    return replace(
        make_candidate(identifier),
        source=source,
        source_record_id=identifier,
        h3_cell=cell,
        latitude=latitude,
        longitude=0.0,
    )


def _context(tmp_path: Path) -> V3Context:
    benchmark = tmp_path / "v2.csv"
    benchmark.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        '"Frozen sentence",yes,Frozen,reserved,0.0,0.0,wikipedia,region,https://example.test/frozen\n',
        encoding="utf-8",
    )
    rows_by_source = {
        source: tuple(
            _candidate(f"{source.value}-{index}", source, f"{source.value}-{index}", float(index + 1))
            for index in range(3)
        )
        for source in V3_SOURCES
    }
    rows_by_source[Source.DESCRIPTION] = (
        _candidate("description-reserved", Source.DESCRIPTION, "reserved", 0.0),
        *rows_by_source[Source.DESCRIPTION],
    )
    settings = V3Settings(
        data_root=tmp_path,
        benchmark_path=benchmark,
        candidate_pool_path=tmp_path / "results" / "pool.json",
        candidate_progress_path=tmp_path / "results" / "progress.json",
        candidate_cells_per_source=2,
        candidate_reservoir_cells_per_source=3,
    )
    return V3Context(
        settings=settings,
        adapters=V3SourceAdapters(
            description=FakeSource(rows_by_source[Source.DESCRIPTION]),
            wikipedia=FakeSource(rows_by_source[Source.WIKIPEDIA]),
            website=FakeSource(rows_by_source[Source.WEBSITE]),
        ),
        centers={
            row.h3_cell: (row.latitude, row.longitude)
            for rows in rows_by_source.values()
            for row in rows
            if row.h3_cell != "reserved"
        },
    )


@pytest.fixture
def v3_context(tmp_path: Path) -> V3Context:
    return _context(tmp_path)


@given("a frozen V2 benchmark with reserved cells")
def frozen_v2_benchmark(v3_context: V3Context) -> None:
    assert v3_context.settings.benchmark_path.is_file()


@given("three streamed V3 sources with enough English candidates")
def streamed_v3_sources(v3_context: V3Context) -> None:
    assert cast(FakeSource, v3_context.adapters.description).rows
    assert cast(FakeSource, v3_context.adapters.wikipedia).rows
    assert cast(FakeSource, v3_context.adapters.website).rows


@when("I build the V3 candidate pool")
def build_pool(v3_context: V3Context) -> None:
    v3_context.pool = build_v3_candidate_pool(
        v3_context.settings,
        adapters=v3_context.adapters,
        center_of_cell=v3_context.centers.__getitem__,
        quotas=SourceLabelQuotas(
            {(source, label): 1 for source in V3_SOURCES for label in (Label.YES, Label.NO)}
        ),
    )


@then(parsers.parse("every source has at least {count:d} fresh candidates"))
def every_source_has_enough_candidates(v3_context: V3Context, count: int) -> None:
    assert v3_context.pool is not None
    for source in V3_SOURCES:
        assert sum(row.source is source for row in v3_context.pool.candidates) >= count


@then("no candidate uses a V2 H3 cell")
def no_candidate_uses_reserved_cell(v3_context: V3Context) -> None:
    assert v3_context.pool is not None
    reserved = load_v2_benchmark(v3_context.settings.benchmark_path).reserved_cells
    assert not reserved.intersection(v3_context.pool.cells)


@then("the V3 quota preflight passes")
def v3_quota_preflight_passes(v3_context: V3Context) -> None:
    assert v3_context.pool is not None
    benchmark = load_v2_benchmark(v3_context.settings.benchmark_path)
    report = preflight_v3_candidate_pool(
        v3_context.pool,
        existing_v2=benchmark.annotations,
        quotas=SourceLabelQuotas(
            {(source, label): 1 for source in V3_SOURCES for label in (Label.YES, Label.NO)}
        ),
        target_cells_per_source=2,
        seed=v3_context.settings.seed,
    )
    assert report.total_candidates == 6
