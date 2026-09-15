"""Quota and geographic checks for the V3 candidate reservoir."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, QuotaKey, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import SeedPlan


class V3PreflightError(ValueError):
    """Raised when a V3 candidate reservoir cannot support annotation quotas."""


@dataclass(frozen=True, slots=True)
class V3PreflightReport:
    """Compact evidence that a fresh V3 reservoir can support its seed plan."""

    candidate_count_by_source: Mapping[Source, int]
    candidate_cells_by_source: Mapping[Source, int]
    required_new_rows_by_source: Mapping[Source, int]
    remaining_by_source_label: Mapping[QuotaKey, int]
    reserved_v2_cells: frozenset[str]

    @property
    def total_candidate_count(self) -> int:
        return sum(self.candidate_count_by_source.values())

    @property
    def total_required_new_rows(self) -> int:
        return sum(self.required_new_rows_by_source.values())


def preflight_v3_candidate_pool(
    pool: FinalizedCandidatePool,
    seed_plan: SeedPlan,
    *,
    quotas: SourceLabelQuotas = V3_QUOTAS,
    candidate_cells_per_source: int = 400,
) -> V3PreflightReport:
    """Verify fresh cells, V2 reservations, and exact downstream quota needs.

    ``pool`` is already finalized, so its constructor has established unique
    candidate IDs, one candidate per global H3 cell, and matching cell order.
    This check adds the V3-specific constraints: every quota source has the
    oversized reservoir requested by the settings, no fresh cell collides with
    any V2 row, and enough fresh rows remain for the seeded quota shortfall.
    """

    if candidate_cells_per_source < 1:
        raise ValueError("candidate_cells_per_source must be positive")
    _require_expected_sources(pool, quotas)
    _require_no_reserved_cells(pool, seed_plan.reserved_cells)

    candidate_count_by_source = _candidate_counts(pool, quotas.sources)
    candidate_cells_by_source = _candidate_cells(pool, quotas.sources)
    required_new_rows_by_source = _required_new_rows(seed_plan, quotas)
    _require_reservoir_sizes(
        candidate_cells_by_source,
        required_new_rows_by_source,
        candidate_cells_per_source,
    )

    return V3PreflightReport(
        candidate_count_by_source=MappingProxyType(candidate_count_by_source),
        candidate_cells_by_source=MappingProxyType(candidate_cells_by_source),
        required_new_rows_by_source=MappingProxyType(required_new_rows_by_source),
        remaining_by_source_label=MappingProxyType(dict(seed_plan.remaining)),
        reserved_v2_cells=seed_plan.reserved_cells,
    )


def _candidate_counts(pool: FinalizedCandidatePool, sources: tuple[Source, ...]) -> dict[Source, int]:
    return {source: sum(candidate.source is source for candidate in pool.candidates) for source in sources}


def _candidate_cells(pool: FinalizedCandidatePool, sources: tuple[Source, ...]) -> dict[Source, int]:
    return {
        source: len({candidate.h3_cell for candidate in pool.candidates if candidate.source is source})
        for source in sources
    }


def _required_new_rows(seed_plan: SeedPlan, quotas: SourceLabelQuotas) -> dict[Source, int]:
    return {
        source: sum(seed_plan.remaining.get((source, label), 0) for label in quotas.labels)
        for source in quotas.sources
    }


def _require_expected_sources(pool: FinalizedCandidatePool, quotas: SourceLabelQuotas) -> None:
    expected = set(quotas.sources)
    unexpected = sorted(
        {candidate.source.value for candidate in pool.candidates if candidate.source not in expected}
    )
    if unexpected:
        raise V3PreflightError(f"candidate pool contains sources outside the quota matrix: {unexpected}")


def _require_no_reserved_cells(pool: FinalizedCandidatePool, reserved_cells: frozenset[str]) -> None:
    collisions = sorted({candidate.h3_cell for candidate in pool.candidates} & reserved_cells)
    if collisions:
        raise V3PreflightError(f"candidate pool reuses V2-reserved H3 cells: {collisions[:5]}")


def _require_reservoir_sizes(
    candidate_cells_by_source: Mapping[Source, int],
    required_new_rows_by_source: Mapping[Source, int],
    target_cells_per_source: int,
) -> None:
    for source, available_cells in candidate_cells_by_source.items():
        if available_cells < target_cells_per_source:
            raise V3PreflightError(
                f"V3 {source.value} reservoir needs at least {target_cells_per_source} fresh cells; "
                f"found {available_cells}"
            )
        required = required_new_rows_by_source[source]
        if available_cells < required:
            raise V3PreflightError(
                f"V3 {source.value} reservoir needs {required} fresh rows for its quota shortfall; "
                f"found {available_cells}"
            )
