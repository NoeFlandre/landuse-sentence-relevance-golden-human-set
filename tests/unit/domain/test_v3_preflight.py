from __future__ import annotations

from dataclasses import replace

import pytest

from landuse_sentence_relevance.domain.models import Annotation, Label, Source
from landuse_sentence_relevance.domain.profile import V3_SOURCES, balanced_quotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan, plan_seed
from landuse_sentence_relevance.domain.v3_preflight import (
    V3PreflightError,
    preflight_v3_candidate_pool,
)
from tests.builders import make_candidate


def _candidate(source: Source, index: int):
    return replace(
        make_candidate(f"{source.value}-{index}"),
        source=source,
        h3_cell=f"{source.value}-cell-{index}",
    )


def test_preflight_reports_available_cells_and_exact_new_rows() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    seed_plan = plan_seed((), quotas, seed="test-seed")
    candidates = tuple(
        candidate for source in V3_SOURCES for candidate in (_candidate(source, 0), _candidate(source, 1))
    )
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    report = preflight_v3_candidate_pool(
        pool,
        seed_plan,
        quotas=quotas,
        candidate_cells_per_source=2,
    )

    assert report.candidate_cells_by_source == {source: 2 for source in V3_SOURCES}
    assert report.required_new_rows_by_source == {source: 2 for source in V3_SOURCES}
    assert report.remaining_by_source_label == {
        (source, label): 1 for source in V3_SOURCES for label in (Label.YES, Label.NO)
    }
    assert report.reserved_v2_cells == frozenset()


def test_preflight_uses_the_default_four_hundred_cell_target_and_counts_candidates() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=0)
    seed_plan = plan_seed((), quotas, seed="test-seed")
    candidates = tuple(_candidate(source, index) for source in V3_SOURCES for index in range(400))
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    report = preflight_v3_candidate_pool(pool, seed_plan, quotas=quotas)

    assert report.candidate_count_by_source == {source: 400 for source in V3_SOURCES}
    assert report.candidate_cells_by_source == {source: 400 for source in V3_SOURCES}
    assert report.required_new_rows_by_source == {source: 0 for source in V3_SOURCES}
    assert report.total_candidate_count == 1200
    assert report.total_required_new_rows == 0


def test_preflight_treats_unrepresented_seed_quota_keys_as_zero() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    seed_plan = plan_seed(
        (_annotation(Source.WIKIPEDIA, Label.YES, "already-filled"),),
        quotas,
        seed="test-seed",
    )
    candidates = tuple(
        [_candidate(Source.WIKIPEDIA, 0)]
        + [_candidate(Source.WEBSITE, index) for index in range(2)]
        + [_candidate(Source.DESCRIPTION, index) for index in range(2)]
    )
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    report = preflight_v3_candidate_pool(
        pool,
        seed_plan,
        quotas=quotas,
        candidate_cells_per_source=1,
    )

    assert report.required_new_rows_by_source == {
        Source.WIKIPEDIA: 1,
        Source.WEBSITE: 2,
        Source.DESCRIPTION: 2,
    }


def test_preflight_rejects_a_candidate_in_a_reserved_v2_cell() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    seed_plan = plan_seed(
        (_annotation(Source.WIKIPEDIA, Label.YES, "reserved-cell"),),
        quotas,
        seed="test-seed",
    )
    candidates = (
        replace(_candidate(Source.WIKIPEDIA, 0), h3_cell="reserved-cell"),
        _candidate(Source.WIKIPEDIA, 1),
        _candidate(Source.WEBSITE, 0),
        _candidate(Source.WEBSITE, 1),
        _candidate(Source.DESCRIPTION, 0),
        _candidate(Source.DESCRIPTION, 1),
    )
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    with pytest.raises(V3PreflightError, match="V2-reserved"):
        preflight_v3_candidate_pool(pool, seed_plan, quotas=quotas, candidate_cells_per_source=2)


def test_preflight_reports_only_the_first_five_reserved_cell_collisions() -> None:
    reserved_cells = frozenset(f"reserved-cell-{index}" for index in range(6))
    seed_plan = SeedPlan((), (), {}, reserved_cells)
    candidates = tuple(
        replace(_candidate(Source.WIKIPEDIA, index), h3_cell=cell)
        for index, cell in enumerate(sorted(reserved_cells))
    )
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    with pytest.raises(V3PreflightError) as error:
        preflight_v3_candidate_pool(
            pool,
            seed_plan,
            quotas=balanced_quotas((Source.WIKIPEDIA,), rows_per_source_label=0),
            candidate_cells_per_source=1,
        )

    assert str(error.value) == (
        "candidate pool reuses V2-reserved H3 cells: "
        "['reserved-cell-0', 'reserved-cell-1', 'reserved-cell-2', "
        "'reserved-cell-3', 'reserved-cell-4']"
    )


def test_preflight_rejects_a_non_positive_reservoir_target() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    seed_plan = plan_seed((), quotas, seed="test-seed")

    with pytest.raises(ValueError) as error:
        preflight_v3_candidate_pool(
            FinalizedCandidatePool((), ()),
            seed_plan,
            quotas=quotas,
            candidate_cells_per_source=0,
        )

    assert str(error.value) == "candidate_cells_per_source must be positive"


def test_preflight_rejects_sources_outside_the_quota_matrix() -> None:
    quotas = balanced_quotas((Source.WIKIPEDIA, Source.WEBSITE), rows_per_source_label=1)
    seed_plan = plan_seed((), quotas, seed="test-seed")
    candidates = tuple(_candidate(source, 0) for source in V3_SOURCES)
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    with pytest.raises(V3PreflightError, match="outside the quota matrix"):
        preflight_v3_candidate_pool(pool, seed_plan, quotas=quotas, candidate_cells_per_source=1)


def test_preflight_rejects_a_reservoir_smaller_than_the_requested_pool() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=1)
    seed_plan = plan_seed((), quotas, seed="test-seed")
    candidates = tuple(_candidate(source, 0) for source in V3_SOURCES)
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    with pytest.raises(V3PreflightError, match="at least 2 fresh cells"):
        preflight_v3_candidate_pool(pool, seed_plan, quotas=quotas, candidate_cells_per_source=2)


def test_preflight_rejects_a_reservoir_smaller_than_a_quota_shortfall() -> None:
    quotas = balanced_quotas(V3_SOURCES, rows_per_source_label=5)
    seed_plan = plan_seed((), quotas, seed="test-seed")
    candidates = tuple(_candidate(source, 0) for source in V3_SOURCES)
    pool = FinalizedCandidatePool(
        candidates=candidates,
        cells=tuple(candidate.h3_cell for candidate in candidates),
    )

    with pytest.raises(V3PreflightError, match="quota shortfall"):
        preflight_v3_candidate_pool(pool, seed_plan, quotas=quotas, candidate_cells_per_source=1)


def _annotation(source: Source, label: Label, cell: str):
    return Annotation(candidate=replace(_candidate(source, 99), h3_cell=cell), label=label)
