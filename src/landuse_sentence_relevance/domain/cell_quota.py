from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CellQuota:
    """Apply a candidate-count threshold to a collection of H3 cells."""

    capacity: int
    completion_threshold: int | None = None

    def is_full(self, cell: str, counts: Mapping[str, int]) -> bool:
        return counts.get(cell, 0) >= self.capacity

    def filled_cells(self, cells: Iterable[str], counts: Mapping[str, int]) -> int:
        threshold = self.capacity if self.completion_threshold is None else self.completion_threshold
        return sum(counts.get(cell, 0) >= threshold for cell in cells)

    def is_reached(
        self,
        cells: Iterable[str],
        counts: Mapping[str, int],
        target_cells: int,
    ) -> bool:
        return self.filled_cells(cells, counts) >= target_cells
