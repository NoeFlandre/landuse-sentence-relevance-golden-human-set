from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from landuse_sentence_relevance.domain.models import Candidate, Source
from landuse_sentence_relevance.domain.stratification import select_common_cells


def _rank(seed: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FinalizedCandidatePool:
    candidates: tuple[Candidate, ...]
    cells: tuple[str, ...]

    def next_unannotated(self, annotated_ids: set[str]) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.candidate_id not in annotated_ids:
                return candidate
        return None


class BoundedCandidatePool:
    """Keep only a deterministic bounded number of candidates per source and H3 cell."""

    def __init__(self, capacity_per_stratum: int, seed: str) -> None:
        if capacity_per_stratum < 1:
            raise ValueError("capacity_per_stratum must be positive")
        self._capacity = capacity_per_stratum
        self._seed = seed
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
        target_cell_count: int,
        center_of_cell: Callable[[str], tuple[float, float]],
    ) -> FinalizedCandidatePool:
        eligible_cells = _eligible_cells(self._buckets)
        cells = select_common_cells(
            eligible_cells,
            target_count=target_cell_count,
            center_of_cell=center_of_cell,
            seed=self._seed,
        )
        return FinalizedCandidatePool(candidates=_selected_candidates(self._buckets, cells), cells=cells)


def _eligible_cells(
    buckets: dict[tuple[Source, str], dict[str, Candidate]],
) -> dict[Source, set[str]]:
    eligible = {source: set() for source in Source}
    for source, cell in buckets:
        eligible[source].add(cell)
    return eligible


def _selected_candidates(
    buckets: dict[tuple[Source, str], dict[str, Candidate]],
    cells: tuple[str, ...],
) -> tuple[Candidate, ...]:
    selected: list[Candidate] = []
    for cell in cells:
        for source in Source:
            selected.extend(_sorted_bucket(buckets[(source, cell)]))
    return tuple(selected)


def _sorted_bucket(bucket: dict[str, Candidate]) -> list[Candidate]:
    return sorted(bucket.values(), key=lambda row: row.candidate_id)
