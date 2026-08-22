from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping

from landuse_sentence_relevance.domain.models import Source


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    latitude_1, longitude_1 = map(math.radians, first)
    latitude_2, longitude_2 = map(math.radians, second)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(haversine))


def _stable_key(seed: str, cell: str) -> str:
    return hashlib.sha256(f"{seed}:{cell}".encode()).hexdigest()


def select_common_cells(
    eligible_cells: Mapping[Source, Iterable[str]],
    target_count: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
) -> tuple[str, ...]:
    """Choose a deterministic, geographically spread subset of shared H3 cells."""
    _require_target_count(target_count)
    common = _shared_cells(eligible_cells)
    _require_enough_cells(common, target_count)
    return _spread_cells(sorted(common), target_count, center_of_cell, seed)


def _require_target_count(target_count: int) -> None:
    if target_count < 1:
        raise ValueError("target_count must be positive")


def _shared_cells(eligible_cells: Mapping[Source, Iterable[str]]) -> set[str]:
    return set(eligible_cells[Source.WIKIPEDIA]) & set(eligible_cells[Source.WEBSITE])


def _require_enough_cells(common: set[str], target_count: int) -> None:
    if len(common) < target_count:
        raise ValueError(f"need {target_count} shared cells, found {len(common)}")


def _spread_cells(
    cells: list[str],
    target_count: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
) -> tuple[str, ...]:
    first = min(cells, key=lambda cell: _stable_key(seed, cell))
    selected = [first]
    remaining = [cell for cell in cells if cell != first]
    while len(selected) < target_count:
        best = max(remaining, key=lambda cell: _distance_to_selection(cell, selected, center_of_cell, seed))
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)


def _distance_to_selection(
    cell: str,
    selected: list[str],
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
) -> tuple[float, str]:
    distance = min(_haversine_km(center_of_cell(cell), center_of_cell(chosen)) for chosen in selected)
    return distance, _stable_key(seed, cell)
