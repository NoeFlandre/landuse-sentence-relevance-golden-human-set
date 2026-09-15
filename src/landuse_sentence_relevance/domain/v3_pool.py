"""Bounded V3 candidate reservoirs and preflight rules."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from landuse_sentence_relevance.domain.models import Annotation, Candidate, Source
from landuse_sentence_relevance.domain.profile import V3_QUOTAS, V3_SOURCES, QuotaKey, SourceLabelQuotas
from landuse_sentence_relevance.domain.sampling import FinalizedCandidatePool
from landuse_sentence_relevance.domain.seeding import plan_seed
from landuse_sentence_relevance.domain.stratification import select_distinct_source_cells


class V3PreflightError(ValueError):
    """Raised when a fresh V3 pool cannot satisfy the downstream contract."""


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


def _candidate_key(seed: str, candidate: Candidate) -> tuple[str, str, str]:
    return _rank(seed, candidate.candidate_id), candidate.candidate_id, candidate.h3_cell


@dataclass(frozen=True, slots=True)
class V3Preflight:
    """Evidence that a fresh candidate pool can feed the exact V3 quotas."""

    candidate_count_by_source: Mapping[Source, int]
    candidate_cells_by_source: Mapping[Source, frozenset[str]]
    required_new_by_source: Mapping[Source, int]
    remaining_quotas: Mapping[QuotaKey, int]
    reserved_cells: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_count_by_source", MappingProxyType(dict(self.candidate_count_by_source))
        )
        object.__setattr__(
            self, "candidate_cells_by_source", MappingProxyType(dict(self.candidate_cells_by_source))
        )
        object.__setattr__(
            self, "required_new_by_source", MappingProxyType(dict(self.required_new_by_source))
        )
        object.__setattr__(self, "remaining_quotas", MappingProxyType(dict(self.remaining_quotas)))

    @property
    def total_candidates(self) -> int:
        return sum(self.candidate_count_by_source.values())


class V3CandidateReservoir:
    """Keep a deterministic bounded candidate sample for every V3 source."""

    def __init__(
        self,
        *,
        capacity_per_source: int,
        seed: str,
        reserved_cells: Iterable[str] = (),
        sources: tuple[Source, ...] = V3_SOURCES,
    ) -> None:
        if capacity_per_source < 1:
            raise ValueError("capacity_per_source must be positive")
        if not sources:
            raise ValueError("sources must not be empty")
        if len(set(sources)) != len(sources):
            raise ValueError("sources must be unique")
        self.capacity_per_source = capacity_per_source
        self.seed = seed
        self.sources = sources
        self.reserved_cells = frozenset(reserved_cells)
        self._buckets: dict[Source, dict[str, Candidate]] = {source: {} for source in sources}

    def add(self, candidate: Candidate) -> bool:
        """Add a candidate unless its cell is reserved or its source is unsupported."""

        bucket = self._bucket_for(candidate.source)
        if candidate.h3_cell in self.reserved_cells:
            return False
        _retain_candidate(bucket, candidate, self.seed, self.capacity_per_source)
        return bucket.get(candidate.h3_cell) == candidate

    def _bucket_for(self, source: Source) -> dict[str, Candidate]:
        bucket = self._buckets.get(source)
        if bucket is None:
            raise ValueError(f"candidate source is outside the V3 profile: {source.value}")
        return bucket

    def snapshot(self) -> tuple[Candidate, ...]:
        return tuple(
            self._buckets[source][cell] for source in self.sources for cell in sorted(self._buckets[source])
        )

    def candidates_for_source(self, source: Source) -> tuple[Candidate, ...]:
        if source not in self._buckets:
            raise ValueError(f"candidate source is outside the V3 profile: {source.value}")
        return tuple(self._buckets[source][cell] for cell in sorted(self._buckets[source]))

    def cells_for_source(self, source: Source) -> frozenset[str]:
        return frozenset(candidate.h3_cell for candidate in self.candidates_for_source(source))

    def count_for_source(self, source: Source) -> int:
        return len(self.candidates_for_source(source))

    def finalize(
        self,
        *,
        target_cells_per_source: int,
        center_of_cell: Callable[[str], tuple[float, float]],
        minimum_distance_km: float = 0.0,
    ) -> FinalizedCandidatePool:
        """Choose one candidate per globally unique, geographically spread cell."""

        _validate_target_cells(target_cells_per_source, self.capacity_per_source)
        selected = self._select_cells(
            target_count_per_source=target_cells_per_source,
            center_of_cell=center_of_cell,
            seed=self.seed,
            minimum_distance_km=minimum_distance_km,
        )
        candidates = _selected_candidates(self, selected)
        return FinalizedCandidatePool(
            candidates=candidates,
            cells=_selected_cells(self.sources, selected),
        )

    def _select_cells(
        self,
        *,
        target_count_per_source: int,
        center_of_cell: Callable[[str], tuple[float, float]],
        seed: str,
        minimum_distance_km: float,
    ) -> dict[Source, tuple[str, ...]]:
        available = {source: self.cells_for_source(source) for source in self.sources}
        return select_distinct_source_cells(
            available,
            target_count_per_source=target_count_per_source,
            center_of_cell=center_of_cell,
            seed=seed,
            minimum_distance_km=minimum_distance_km,
            sources=self.sources,
        )

    def _candidate_for_cell(self, source: Source, cell: str) -> Candidate:
        return next(
            candidate for candidate in self.candidates_for_source(source) if candidate.h3_cell == cell
        )


def preflight_v3_candidate_pool(
    pool: FinalizedCandidatePool,
    *,
    existing_v2: Iterable[Annotation],
    quotas: SourceLabelQuotas = V3_QUOTAS,
    target_cells_per_source: int,
    seed: str,
) -> V3Preflight:
    """Verify fresh cells and candidates cover every remaining V3 quota."""

    _validate_target_cells(target_cells_per_source)
    seed_plan = plan_seed(existing_v2, quotas, seed)
    reserved_cells = seed_plan.reserved_cells
    _reject_reserved_cells(pool, reserved_cells)
    _reject_unexpected_sources(pool, quotas.sources)
    candidate_count_by_source = _candidate_counts(pool, quotas.sources)
    required_new_by_source = _required_new_counts(seed_plan.remaining, quotas)
    _reject_source_shortfalls(
        candidate_count_by_source,
        required_new_by_source,
        quotas.sources,
        target_cells_per_source,
    )

    cells_by_source = _candidate_cells_by_source(pool, quotas.sources)
    return V3Preflight(
        candidate_count_by_source=candidate_count_by_source,
        candidate_cells_by_source=cells_by_source,
        required_new_by_source=required_new_by_source,
        remaining_quotas=seed_plan.remaining,
        reserved_cells=reserved_cells,
    )


def _retain_candidate(
    bucket: dict[str, Candidate],
    candidate: Candidate,
    seed: str,
    capacity: int,
) -> None:
    current = bucket.get(candidate.h3_cell)
    if current is not None and _candidate_key(seed, current) <= _candidate_key(seed, candidate):
        return
    bucket[candidate.h3_cell] = candidate
    if len(bucket) > capacity:
        worst_cell = max(bucket, key=lambda cell: _candidate_key(seed, bucket[cell]))
        del bucket[worst_cell]


def _selected_candidates(
    reservoir: V3CandidateReservoir,
    selected: Mapping[Source, tuple[str, ...]],
) -> tuple[Candidate, ...]:
    return tuple(
        reservoir._candidate_for_cell(source, cell)
        for source in reservoir.sources
        for cell in selected[source]
    )


def _selected_cells(
    sources: tuple[Source, ...],
    selected: Mapping[Source, tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(cell for source in sources for cell in selected[source])


def _candidate_counts(
    pool: FinalizedCandidatePool,
    sources: tuple[Source, ...],
) -> dict[Source, int]:
    return {source: sum(candidate.source is source for candidate in pool.candidates) for source in sources}


def _required_new_counts(
    remaining: Mapping[QuotaKey, int],
    quotas: SourceLabelQuotas,
) -> dict[Source, int]:
    return {
        source: sum(remaining.get((source, label), 0) for label in quotas.labels) for source in quotas.sources
    }


def _candidate_cells_by_source(
    pool: FinalizedCandidatePool,
    sources: tuple[Source, ...],
) -> dict[Source, frozenset[str]]:
    return {
        source: frozenset(candidate.h3_cell for candidate in pool.candidates if candidate.source is source)
        for source in sources
    }


def _validate_target_cells(target_cells_per_source: int, capacity: int | None = None) -> None:
    if target_cells_per_source < 1:
        raise ValueError("target_cells_per_source must be positive")
    if capacity is not None and target_cells_per_source > capacity:
        raise ValueError("target_cells_per_source must not exceed reservoir capacity")


def _reject_reserved_cells(pool: FinalizedCandidatePool, reserved_cells: frozenset[str]) -> None:
    reserved = sorted({candidate.h3_cell for candidate in pool.candidates} & reserved_cells)
    if not reserved:
        return
    preview = ", ".join(reserved[:3])
    suffix = "..." if len(reserved) > 3 else ""
    raise V3PreflightError(f"candidate pool contains a reserved V2 H3 cell: {preview}{suffix}")


def _reject_unexpected_sources(pool: FinalizedCandidatePool, sources: tuple[Source, ...]) -> None:
    unexpected_sources = {candidate.source for candidate in pool.candidates} - set(sources)
    if not unexpected_sources:
        return
    names = ", ".join(sorted(source.value for source in unexpected_sources))
    raise V3PreflightError(f"candidate pool contains sources outside the V3 quota matrix: {names}")


def _reject_source_shortfalls(
    candidate_count_by_source: Mapping[Source, int],
    required_new_by_source: Mapping[Source, int],
    sources: tuple[Source, ...],
    target_cells_per_source: int,
) -> None:
    for source in sources:
        count = candidate_count_by_source[source]
        required_new = required_new_by_source[source]
        if count < required_new:
            raise V3PreflightError(
                f"V3 {source.value} source has {count} candidates; need {required_new} new "
                "candidates for its exact quota"
            )
        if count < target_cells_per_source:
            raise V3PreflightError(
                f"V3 {source.value} source has {count} candidates; need at least "
                f"{target_cells_per_source} fresh candidates"
            )
