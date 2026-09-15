from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from itertools import product

from landuse_sentence_relevance.domain.models import Source

DEFAULT_SOURCES: tuple[Source, ...] = (Source.WIKIPEDIA, Source.WEBSITE)

_EARTH_RADIUS_KM = 6371.0088
_UNIT_CUBE_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    latitude_1, longitude_1 = map(math.radians, first)
    latitude_2, longitude_2 = map(math.radians, second)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(haversine))


def _conflict_counts(
    centers: Mapping[str, tuple[float, float]],
    minimum_distance_km: float,
) -> dict[str, int]:
    """Count cells that would conflict with each cell at the distance floor."""
    cells = tuple(sorted(centers))
    if not cells:
        return {}
    if minimum_distance_km <= 0:
        return dict.fromkeys(cells, 0)
    if minimum_distance_km >= math.pi * _EARTH_RADIUS_KM:
        return {cell: len(cells) - 1 for cell in cells}

    return _spatial_conflict_counts(cells, centers, minimum_distance_km)


def _spatial_conflict_counts(
    cells: tuple[str, ...],
    centers: Mapping[str, tuple[float, float]],
    minimum_distance_km: float,
) -> dict[str, int]:
    counts = dict.fromkeys(cells, 0)
    chord_radius = 2 * math.sin(minimum_distance_km / _EARTH_RADIUS_KM / 2)
    vectors = _cell_vectors(cells, centers)
    buckets = _spatial_buckets(cells, vectors, chord_radius)
    for cell in cells:
        _count_cell_conflicts(
            cell,
            centers,
            vectors,
            buckets,
            chord_radius,
            minimum_distance_km,
            counts,
        )
    return counts


def _cell_vectors(
    cells: tuple[str, ...],
    centers: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float, float]]:
    return {cell: _unit_vector(centers[cell]) for cell in cells}


def _spatial_buckets(
    cells: tuple[str, ...],
    vectors: Mapping[str, tuple[float, float, float]],
    width: float,
) -> dict[tuple[int, int, int], list[str]]:
    buckets: dict[tuple[int, int, int], list[str]] = {}
    for cell in cells:
        buckets.setdefault(_cube_bin(vectors[cell], width), []).append(cell)
    return buckets


def _count_cell_conflicts(
    cell: str,
    centers: Mapping[str, tuple[float, float]],
    vectors: Mapping[str, tuple[float, float, float]],
    buckets: Mapping[tuple[int, int, int], list[str]],
    width: float,
    minimum_distance_km: float,
    counts: dict[str, int],
) -> None:
    bucket = _cube_bin(vectors[cell], width)
    for offset in _UNIT_CUBE_NEIGHBOR_OFFSETS:
        neighbor_bucket = tuple(sum((bucket[index], offset[index])) for index in range(3))
        _count_bucket_conflicts(cell, centers, buckets.get(neighbor_bucket, ()), minimum_distance_km, counts)


def _count_bucket_conflicts(
    cell: str,
    centers: Mapping[str, tuple[float, float]],
    other_cells: Iterable[str],
    minimum_distance_km: float,
    counts: dict[str, int],
) -> None:
    for other in other_cells:
        if other <= cell:
            continue
        if _haversine_km(centers[cell], centers[other]) < minimum_distance_km:
            counts[cell] += 1
            counts[other] += 1


def _unit_vector(center: tuple[float, float]) -> tuple[float, float, float]:
    latitude, longitude = map(math.radians, center)
    cosine_latitude = math.cos(latitude)
    return (
        cosine_latitude * math.cos(longitude),
        cosine_latitude * math.sin(longitude),
        math.sin(latitude),
    )


def _cube_bin(vector: tuple[float, float, float], width: float) -> tuple[int, int, int]:
    return (
        math.floor((vector[0] + 1) / width),
        math.floor((vector[1] + 1) / width),
        math.floor((vector[2] + 1) / width),
    )


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
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
) -> dict[Source, tuple[str, ...]]:
    """Choose disjoint, globally spread cells for each of ``sources``."""
    _require_target_count(target_count_per_source)
    _require_nonnegative_distance(minimum_distance_km)
    available = _available_source_cells(eligible_cells, sources)
    centers = _cell_centers(available, center_of_cell)
    conflict_counts = _conflict_counts(centers, minimum_distance_km) if minimum_distance_km > 0 else None
    selected_by_source = _empty_source_selection(sources)
    selected_cells: list[str] = []
    nearest_distances = {} if minimum_distance_km > 0 else None
    for _ in range(target_count_per_source):
        _append_selection_round(
            available,
            selected_by_source,
            selected_cells,
            centers,
            target_count_per_source,
            minimum_distance_km,
            seed,
            nearest_distances,
            conflict_counts,
            sources,
        )
    return _freeze_source_selection(selected_by_source)


def _available_source_cells(
    eligible_cells: Mapping[Source, Iterable[str]],
    sources: tuple[Source, ...],
) -> dict[Source, set[str]]:
    return {source: set(eligible_cells[source]) for source in sources}


def _cell_centers(
    available: Mapping[Source, set[str]],
    center_of_cell: Callable[[str], tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    cells: set[str] = set()
    for source_cells in available.values():
        cells.update(source_cells)
    return {cell: center_of_cell(cell) for cell in cells}


def _empty_source_selection(sources: tuple[Source, ...]) -> dict[Source, list[str]]:
    return {source: [] for source in sources}


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
    nearest_distances: dict[str, float] | None = None,
    conflict_counts: Mapping[str, int] | None = None,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
) -> None:
    for source in sources:
        _append_next_source_cell(
            source,
            available,
            selected_by_source,
            selected_cells,
            centers,
            target_count,
            minimum_distance_km,
            seed,
            nearest_distances,
            conflict_counts,
            sources,
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
    nearest_distances: dict[str, float] | None = None,
    conflict_counts: Mapping[str, int] | None = None,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
) -> None:
    candidates = _required_source_candidates(
        source,
        available,
        selected_by_source,
        selected_cells,
        target_count,
        sources,
    )
    distant = _required_distant_candidates(
        candidates,
        selected_cells,
        centers,
        minimum_distance_km,
        nearest_distances,
    )
    cell = _next_source_cell(distant, selected_cells, centers, seed, nearest_distances, conflict_counts)
    _update_distance_cache_if_enabled(centers, nearest_distances, selected_cells, cell)
    selected_by_source[source].append(cell)
    selected_cells.append(cell)


def _required_source_candidates(
    source: Source,
    available: Mapping[Source, set[str]],
    selected_by_source: Mapping[Source, list[str]],
    selected_cells: list[str],
    target_count: int,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
) -> list[str]:
    candidates = _feasible_source_cells(
        source,
        available,
        selected_by_source,
        selected_cells,
        target_count,
        sources,
    )
    if not candidates:
        raise ValueError("need enough disjoint source cells")
    return candidates


def _required_distant_candidates(
    candidates: list[str],
    selected_cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    minimum_distance_km: float,
    nearest_distances: Mapping[str, float] | None,
) -> list[str]:
    distant = _distant_candidates(
        candidates,
        selected_cells,
        centers,
        minimum_distance_km,
        nearest_distances,
    )
    if not distant:
        raise ValueError("cannot satisfy the minimum distance between H3 cells")
    return distant


def _distant_candidates(
    candidates: list[str],
    selected_cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    minimum_distance_km: float,
    nearest_distances: Mapping[str, float] | None,
) -> list[str]:
    if nearest_distances is None or not selected_cells:
        return _distant_cells(candidates, selected_cells, centers, minimum_distance_km)
    return _cached_distant_cells(candidates, nearest_distances, minimum_distance_km)


def _next_source_cell(
    candidates: list[str],
    selected_cells: list[str],
    centers: Mapping[str, tuple[float, float]],
    seed: str,
    nearest_distances: Mapping[str, float] | None,
    conflict_counts: Mapping[str, int] | None = None,
) -> str:
    if conflict_counts is not None:
        return _least_conflicting_cell(candidates, conflict_counts, seed)
    if nearest_distances is None or not selected_cells:
        return _farthest_or_stable_cell(candidates, selected_cells, centers, seed)
    return _farthest_cached_cell(candidates, nearest_distances, seed)


def _least_conflicting_cell(
    cells: list[str],
    conflict_counts: Mapping[str, int],
    seed: str,
) -> str:
    return min(cells, key=lambda cell: (conflict_counts[cell], _stable_key(seed, cell)))


def _update_distance_cache_if_enabled(
    centers: Mapping[str, tuple[float, float]],
    nearest_distances: dict[str, float] | None,
    selected_cells: list[str],
    selected: str,
) -> None:
    if nearest_distances is not None:
        _update_selected_distance_cache(centers, nearest_distances, selected_cells, selected)


def _cached_distant_cells(
    cells: list[str],
    nearest_distances: Mapping[str, float],
    minimum_distance_km: float,
) -> list[str]:
    return [cell for cell in cells if nearest_distances[cell] >= minimum_distance_km]


def _farthest_cached_cell(
    cells: list[str],
    nearest_distances: Mapping[str, float],
    seed: str,
) -> str:
    return max(cells, key=lambda cell: (nearest_distances[cell], _stable_key(seed, cell)))


def _update_selected_distance_cache(
    centers: Mapping[str, tuple[float, float]],
    nearest_distances: dict[str, float],
    selected_cells: list[str],
    selected: str,
) -> None:
    selected_center = centers[selected]
    for cell, center in centers.items():
        if cell == selected or cell in selected_cells:
            continue
        distance = _haversine_km(center, selected_center)
        nearest_distances[cell] = min(nearest_distances.get(cell, math.inf), distance)
    nearest_distances[selected] = 0.0


def _feasible_source_cells(
    source: Source,
    available: Mapping[Source, set[str]],
    selected_by_source: Mapping[Source, list[str]],
    selected_cells: list[str],
    target_count: int,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
) -> list[str]:
    candidates = sorted(available[source] - set(selected_cells))
    feasible: list[str] = []
    for candidate in candidates:
        used = set(selected_cells) | {candidate}
        if all(
            len(available[other] - used)
            >= target_count - len(selected_by_source[other]) - (1 if other is source else 0)
            for other in sources
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
