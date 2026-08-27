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


def select_spread_cells(
    cells: Iterable[str],
    target_count: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
) -> tuple[str, ...]:
    """Choose up to ``target_count`` cells with deterministic maximin spacing."""
    _require_target_count(target_count)
    available = sorted(set(cells))
    if len(available) <= target_count:
        return tuple(available)
    return _spread_cells(available, target_count, center_of_cell, seed)


def select_distinct_source_cells(
    eligible_cells: Mapping[Source, Iterable[str]],
    target_count_per_source: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
    minimum_distance_km: float = 0.0,
) -> dict[Source, tuple[str, ...]]:
    """Choose disjoint, globally spread cells for each source."""
    _require_target_count(target_count_per_source)
    _require_nonnegative_distance(minimum_distance_km)
    available = _available_source_cells(eligible_cells)
    centers = _cell_centers(available, center_of_cell)
    selected_by_source = _empty_source_selection()
    selected_cells: list[str] = []
    for _ in range(target_count_per_source):
        _append_selection_round(
            available,
            selected_by_source,
            selected_cells,
            centers,
            target_count_per_source,
            minimum_distance_km,
            seed,
        )
    return _freeze_source_selection(selected_by_source)


def _available_source_cells(eligible_cells: Mapping[Source, Iterable[str]]) -> dict[Source, set[str]]:
    return {source: set(eligible_cells[source]) for source in Source}


def _cell_centers(
    available: Mapping[Source, set[str]],
    center_of_cell: Callable[[str], tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    cells: set[str] = set()
    for source_cells in available.values():
        cells.update(source_cells)
    return {cell: center_of_cell(cell) for cell in cells}


def _empty_source_selection() -> dict[Source, list[str]]:
    return {source: [] for source in Source}


def _freeze_source_selection(selected: Mapping[Source, list[str]]) -> dict[Source, tuple[str, ...]]:
    return {source: tuple(cells) for source, cells in selected.items()}


def _require_nonnegative_distance(minimum_distance_km: float) -> None:
    if minimum_distance_km < 0:
        raise ValueError("minimum_distance_km must not be negative")


def _append_selection_round(
    available: Mapping[Source, set[str]],
    selected_by_source: dict[Source, list[str]],
    selected_cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    target_count: int,
    minimum_distance_km: float,
    seed: str,
) -> None:
    for source in Source:
        _append_next_source_cell(
            source,
            available,
            selected_by_source,
            selected_cells,
            centers,
            target_count,
            minimum_distance_km,
            seed,
        )


def _append_next_source_cell(
    source: Source,
    available: Mapping[Source, set[str]],
    selected_by_source: dict[Source, list[str]],
    selected_cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    target_count: int,
    minimum_distance_km: float,
    seed: str,
) -> None:
    candidates = _feasible_source_cells(
        source,
        available,
        selected_by_source,
        selected_cells,
        target_count,
    )
    if not candidates:
        raise ValueError("need enough disjoint source cells")
    distant = _distant_cells(candidates, selected_cells, centers, minimum_distance_km)
    if not distant:
        raise ValueError("cannot satisfy the minimum distance between H3 cells")
    cell = _farthest_or_stable_cell(distant, selected_cells, centers, seed)
    selected_by_source[source].append(cell)
    selected_cells.append(cell)


def _feasible_source_cells(
    source: Source,
    available: Mapping[Source, set[str]],
    selected_by_source: Mapping[Source, list[str]],
    selected_cells: list[str],
    target_count: int,
) -> list[str]:
    candidates = sorted(available[source] - set(selected_cells))
    feasible: list[str] = []
    for candidate in candidates:
        used = set(selected_cells) | {candidate}
        if all(
            len(available[other] - used)
            >= target_count - len(selected_by_source[other]) - (1 if other is source else 0)
            for other in Source
        ):
            feasible.append(candidate)
    return feasible


def _distant_cells(
    cells: list[str],
    selected: list[str],
    centers: Mapping[str, tuple[float, float]],
    minimum_distance_km: float,
) -> list[str]:
    if minimum_distance_km == 0:
        return cells
    if not selected:
        return cells
    return [
        cell
        for cell in cells
        if _minimum_distance_to_selection(cell, selected, centers.__getitem__) >= minimum_distance_km
    ]


def _minimum_distance_to_selection(
    cell: str,
    selected: list[str],
    center_of_cell: Callable[[str], tuple[float, float]],
) -> float:
    return min(_haversine_km(center_of_cell(cell), center_of_cell(chosen)) for chosen in selected)


def _farthest_or_stable_cell(
    cells: list[str],
    selected: list[str],
    centers: Mapping[str, tuple[float, float]],
    seed: str,
) -> str:
    if not selected:
        return min(cells, key=lambda cell: _stable_key(seed, cell))
    return max(cells, key=lambda cell: _distance_to_selection(cell, selected, centers.__getitem__, seed))


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
    if target_count == 1:
        return (first,)
    return _spread_remaining(cells, target_count, center_of_cell, seed, first)


def _spread_remaining(
    cells: list[str],
    target_count: int,
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
    first: str,
) -> tuple[str, ...]:
    selected = [first]
    centers = {cell: center_of_cell(cell) for cell in cells}
    remaining = [cell for cell in cells if cell != first]
    nearest_distances = _initial_distances(remaining, centers, first)
    for _ in range(target_count - 1):
        best = _farthest_cell(remaining, nearest_distances, seed)
        selected.append(best)
        remaining.remove(best)
        _update_nearest_distances(remaining, nearest_distances, centers, best)
    return tuple(selected)


def _initial_distances(
    cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    first: str,
) -> dict[str, float]:
    return {cell: _haversine_km(centers[cell], centers[first]) for cell in cells}


def _farthest_cell(cells: list[str], distances: Mapping[str, float], seed: str) -> str:
    return max(cells, key=lambda cell: (distances[cell], _stable_key(seed, cell)))


def _update_nearest_distances(
    cells: list[str],
    distances: dict[str, float],
    centers: Mapping[str, tuple[float, float]],
    selected: str,
) -> None:
    for cell in cells:
        distances[cell] = min(distances[cell], _haversine_km(centers[cell], centers[selected]))


def _distance_to_selection(
    cell: str,
    selected: list[str],
    center_of_cell: Callable[[str], tuple[float, float]],
    seed: str,
) -> tuple[float, str]:
    distance = min(_haversine_km(center_of_cell(cell), center_of_cell(chosen)) for chosen in selected)
    return distance, _stable_key(seed, cell)
