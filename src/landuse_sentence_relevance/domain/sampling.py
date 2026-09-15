from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.stratification import (
    DEFAULT_SOURCES,
    select_distinct_source_cells,
)


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FinalizedCandidatePool:
    candidates: tuple[Candidate, ...]
    cells: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_unique_candidate_ids(self.candidates)
        _require_unique_candidate_cells(self.candidates)
        _require_matching_candidate_cells(self.candidates, self.cells)

    def next_unannotated(self, annotated_ids: set[str]) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.candidate_id not in annotated_ids:
                return candidate
        return None


def _require_unique_candidate_ids(candidates: tuple[Candidate, ...]) -> None:
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate pool must contain unique candidate IDs")


def _require_unique_candidate_cells(candidates: tuple[Candidate, ...]) -> None:
    candidate_cells = {candidate.h3_cell for candidate in candidates}
    if len(candidate_cells) != len(candidates):
        raise ValueError("candidate pool must contain one candidate per H3 cell")


def _require_matching_candidate_cells(candidates: tuple[Candidate, ...], cells: tuple[str, ...]) -> None:
    candidate_cells = tuple(candidate.h3_cell for candidate in candidates)
    if tuple(cells) != candidate_cells:
        raise ValueError("candidate pool cells must match candidate order")


class BoundedCandidatePool:
    """Keep only a deterministic bounded number of candidates per source and H3 cell."""

    def __init__(
        self,
        capacity_per_stratum: int,
        seed: str,
        sources: tuple[Source, ...] = DEFAULT_SOURCES,
    ) -> None:
        if capacity_per_stratum < 1:
            raise ValueError("capacity_per_stratum must be positive")
        self._capacity = capacity_per_stratum
        self._seed = seed
        self._sources = sources
        self._buckets: dict[tuple[Source, str], dict[str, Candidate]] = {}

    def add(self, candidate: Candidate) -> None:
        bucket = self._buckets.setdefault(candidate.stratum, {})
        bucket[candidate.candidate_id] = candidate
        if len(bucket) > self._capacity:
            worst_id = max(bucket, key=lambda candidate_id: (_rank(self._seed, candidate_id), candidate_id))
            del bucket[worst_id]

    def snapshot(self) -> tuple[Candidate, ...]:
        return tuple(
            candidate
            for key in sorted(self._buckets, key=lambda item: (item[1], item[0].value))
            for candidate in sorted(self._buckets[key].values(), key=lambda row: row.candidate_id)
        )

    def finalize(
        self,
        target_cells_per_source: int,
        center_of_cell: Callable[[str], tuple[float, float]],
        minimum_distance_km: float = 0.0,
    ) -> FinalizedCandidatePool:
        eligible_cells = _eligible_cells(self._buckets, self._sources)
        selected_by_source = select_distinct_source_cells(
            eligible_cells,
            target_count_per_source=target_cells_per_source,
            center_of_cell=center_of_cell,
            seed=self._seed,
            minimum_distance_km=minimum_distance_km,
            sources=self._sources,
        )
        cells = tuple(cell for source in self._sources for cell in selected_by_source[source])
        return FinalizedCandidatePool(
            candidates=_selected_candidates(self._buckets, selected_by_source, self._sources),
            cells=cells,
        )


def _eligible_cells(
    buckets: dict[tuple[Source, str], dict[str, Candidate]],
    sources: tuple[Source, ...],
) -> dict[Source, set[str]]:
    eligible: dict[Source, set[str]] = {source: set() for source in sources}
    for source, cell in buckets:
        if source in eligible:
            eligible[source].add(cell)
    return eligible


def _selected_candidates(
    buckets: dict[tuple[Source, str], dict[str, Candidate]],
    cells_by_source: dict[Source, tuple[str, ...]],
    sources: tuple[Source, ...],
) -> tuple[Candidate, ...]:
    return tuple(
        min(buckets[(source, cell)].values(), key=lambda row: row.candidate_id)
        for source in sources
        for cell in cells_by_source[source]
    )
