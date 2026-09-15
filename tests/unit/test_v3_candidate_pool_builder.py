from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from landuse_sentence_relevance.bootstrap import (
    _finalize_v3_pool,
    _resolve_v3_geometry,
    _v3_candidate_pool_metadata,
    build_v3_candidate_pool,
)
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, SourceLabelQuotas, balanced_quotas
from landuse_sentence_relevance.domain.sampling import BoundedCandidatePool
from landuse_sentence_relevance.domain.v3_preflight import V3PreflightError
from landuse_sentence_relevance.sources.v3 import V3SourceAdapters
from landuse_sentence_relevance.storage.candidate_progress import CandidateProgressStore
from tests.unit.test_models import make_candidate


@dataclass(frozen=True, slots=True)
class StaticSource:
    candidates: tuple[Candidate, ...]

    def iter_candidates(self) -> Iterator[Candidate]:
        return iter(self.candidates)


@dataclass(frozen=True, slots=True)
class FailingSource:
    first: Candidate

    def iter_candidates(self) -> Iterator[Candidate]:
        yield self.first
        raise RuntimeError("synthetic source interruption")


def _candidate(source: Source, index: int, cell: str | None = None) -> Candidate:
    return replace(
        make_candidate(f"{source.value}-{index}"),
        source=source,
        h3_cell=cell or f"{source.value}-cell-{index}",
    )


def _adapters() -> V3SourceAdapters:
    return V3SourceAdapters(
        description=StaticSource(tuple(_candidate(Source.DESCRIPTION, index) for index in range(2))),
        wikipedia=StaticSource(
            (
                _candidate(Source.WIKIPEDIA, 0, "reserved-cell"),
                _candidate(Source.WIKIPEDIA, 1),
                _candidate(Source.WIKIPEDIA, 2),
            )
        ),
        website=StaticSource(tuple(_candidate(Source.WEBSITE, index) for index in range(2))),
    )


def _benchmark(path: Path) -> None:
    path.write_text(
        "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
        "seed,yes,Seed,reserved-cell,1,2,wikipedia,region,https://example.test/seed\n",
        encoding="utf-8",
    )


def test_builder_filters_v2_cells_and_reuses_a_finalized_pool(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    _benchmark(benchmark)
    quotas: SourceLabelQuotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    settings = V3Settings(
        data_root=tmp_path,
        benchmark_path=benchmark,
        candidate_cells_per_source=2,
        minimum_cell_distance_km=0.0,
    )
    centers = {
        candidate.h3_cell: (0.0, float(index))
        for index, candidate in enumerate(
            (
                _candidate(Source.WIKIPEDIA, 1),
                _candidate(Source.WIKIPEDIA, 2),
                _candidate(Source.WEBSITE, 0),
                _candidate(Source.WEBSITE, 1),
                _candidate(Source.DESCRIPTION, 0),
                _candidate(Source.DESCRIPTION, 1),
            )
        )
    }

    result = build_v3_candidate_pool(
        settings,
        quotas=quotas,
        adapters=_adapters(),
        cell_for_location=lambda latitude, longitude: "unused",
        center_of_cell=centers.__getitem__,
    )

    assert len(result.pool.candidates) == 6
    assert set(result.pool.cells).isdisjoint({"reserved-cell"})
    assert result.preflight.candidate_cells_by_source == {source: 2 for source in V3_SOURCES}
    assert result.preflight.required_new_rows_by_source == {
        Source.WIKIPEDIA: 1,
        Source.WEBSITE: 2,
        Source.DESCRIPTION: 2,
    }
    assert settings.candidate_pool_path.exists()
    assert settings.candidate_progress_path.exists()

    cached = build_v3_candidate_pool(
        settings,
        quotas=quotas,
        adapters=V3SourceAdapters(
            description=StaticSource(()),
            wikipedia=StaticSource(()),
            website=StaticSource(()),
        ),
        cell_for_location=lambda latitude, longitude: "unused",
        center_of_cell=centers.__getitem__,
    )

    assert cached.pool == result.pool
    assert cached.preflight == result.preflight


def test_builder_checkpoints_candidates_when_a_source_stream_is_interrupted(tmp_path: Path) -> None:
    benchmark = tmp_path / "v2.csv"
    _benchmark(benchmark)
    quotas: SourceLabelQuotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    settings = V3Settings(
        data_root=tmp_path,
        benchmark_path=benchmark,
        candidate_cells_per_source=2,
        minimum_cell_distance_km=0.0,
    )
    first_wikipedia = _candidate(Source.WIKIPEDIA, 0)

    with pytest.raises(RuntimeError, match="synthetic source interruption"):
        build_v3_candidate_pool(
            settings,
            quotas=quotas,
            adapters=V3SourceAdapters(
                description=StaticSource(()),
                wikipedia=FailingSource(first_wikipedia),
                website=StaticSource(()),
            ),
            cell_for_location=lambda latitude, longitude: "unused",
            center_of_cell=lambda cell: (0.0, 0.0),
        )

    metadata = _v3_candidate_pool_metadata(settings, quotas)
    checkpoint = CandidateProgressStore(settings.candidate_progress_path).load(metadata)
    assert checkpoint is not None
    assert tuple(candidate.candidate_id for candidate in checkpoint) == (first_wikipedia.candidate_id,)

    candidates = (
        _candidate(Source.WIKIPEDIA, 1),
        _candidate(Source.WIKIPEDIA, 2),
        _candidate(Source.WEBSITE, 0),
        _candidate(Source.WEBSITE, 1),
        _candidate(Source.DESCRIPTION, 0),
        _candidate(Source.DESCRIPTION, 1),
    )
    all_candidates = (first_wikipedia, *candidates)
    centers = {candidate.h3_cell: (0.0, float(index)) for index, candidate in enumerate(all_candidates)}
    resumed = build_v3_candidate_pool(
        settings,
        quotas=quotas,
        adapters=V3SourceAdapters(
            description=StaticSource(tuple(candidates[-2:])),
            wikipedia=StaticSource((first_wikipedia, *candidates[:2])),
            website=StaticSource(tuple(candidates[2:4])),
        ),
        cell_for_location=lambda latitude, longitude: "unused",
        center_of_cell=centers.__getitem__,
    )

    assert len(resumed.pool.candidates) == 6


def test_builder_resolves_default_h3_geometry_when_callers_do_not_inject_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def expected_cell(latitude: float, longitude: float) -> str:
        return "default-cell"

    def expected_center(cell: str) -> tuple[float, float]:
        return (1.0, 2.0)

    monkeypatch.setattr(
        "landuse_sentence_relevance.bootstrap._h3_geometry",
        lambda settings: (expected_cell, expected_center),
    )

    resolved = _resolve_v3_geometry(V3Settings(data_root=tmp_path), None, None)

    assert resolved == (expected_cell, expected_center)


def test_builder_reports_available_cells_when_geographic_finalization_is_impossible(
    tmp_path: Path,
) -> None:
    settings = V3Settings(data_root=tmp_path, candidate_cells_per_source=2)
    pool = BoundedCandidatePool(
        capacity_per_stratum=1,
        seed=settings.seed,
        sources=V3_SOURCES,
    )
    for source in V3_SOURCES:
        pool.add(_candidate(source, 0))

    with pytest.raises(V3PreflightError, match="available"):
        _finalize_v3_pool(pool, settings, lambda cell: (0.0, 0.0))
