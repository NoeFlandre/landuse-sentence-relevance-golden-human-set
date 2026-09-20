"""Deterministic geographic coverage map generation for the V3 benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "sentence",
    "label",
    "polygon_name",
    "h3_cell",
    "latitude",
    "longitude",
    "source",
    "region",
    "source_url",
)
SOURCE_ORDER = ("description", "website", "wikipedia")
V3_SOURCE_COUNTS = {source: 100 for source in SOURCE_ORDER}
SOURCE_COLORS = {
    "description": "#e58a13",
    "website": "#477ff0",
    "wikipedia": "#28a67d",
}
SOURCE_LABELS = {
    "description": "Description",
    "website": "Website",
    "wikipedia": "Wikipedia",
}
type BoundaryRing = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class BenchmarkPoint:
    """The geographic fields required to render one benchmark point."""

    latitude: float
    longitude: float
    source: str


def load_benchmark_points(
    path: Path,
    *,
    expected_counts: Mapping[str, int] = V3_SOURCE_COUNTS,
) -> tuple[BenchmarkPoint, ...]:
    """Load and validate benchmark coordinates and source assignments."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"{path} has an unexpected benchmark CSV header")

        points = tuple(
            _point_from_row(row, path, row_number) for row_number, row in enumerate(reader, start=2)
        )

    _validate_source_counts(points, expected_counts, path)
    return points


def load_boundary_rings(path: Path) -> tuple[BoundaryRing, ...]:
    """Read exterior Polygon and MultiPolygon rings from a GeoJSON file."""

    document = _read_json(path)
    features = _feature_collection_features(document, path)
    return tuple(
        ring
        for feature_number, feature in enumerate(features, start=1)
        for ring in _feature_boundary_rings(feature, path, feature_number)
    )


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _feature_collection_features(document: Any, path: Path) -> list[Any]:
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError(f"{path} must contain a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain at least one GeoJSON feature")
    return features


def _feature_boundary_rings(feature: Any, path: Path, feature_number: int) -> tuple[BoundaryRing, ...]:
    if not isinstance(feature, dict):
        raise ValueError(f"{path} feature {feature_number} is not an object")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"{path} feature {feature_number} has no geometry")
    return tuple(
        _parse_boundary_ring(_polygon_exterior(polygon, path, feature_number), path, feature_number)
        for polygon in _geometry_polygons(geometry, path, feature_number)
    )


def _geometry_polygons(geometry: dict[str, Any], path: Path, feature_number: int) -> list[Any]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError(f"{path} feature {feature_number} uses unsupported geometry {geometry_type!r}")
    if not isinstance(polygons, list):
        raise ValueError(f"{path} feature {feature_number} has malformed polygon coordinates")
    return polygons


def _polygon_exterior(polygon: Any, path: Path, feature_number: int) -> Any:
    if not isinstance(polygon, list) or not polygon:
        raise ValueError(f"{path} feature {feature_number} has an empty polygon")
    return polygon[0]


def render_world_map(
    points: Sequence[BenchmarkPoint],
    boundary_rings: Sequence[BoundaryRing],
    output_path: Path,
) -> None:
    """Render the fixed V3 coverage map without network or clock input."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    figure, canvas, polygon_class = _new_canvas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        axes = _configure_axes(figure)
        _draw_boundaries(axes, boundary_rings, polygon_class)
        _draw_points(axes, points)
        _add_figure_labels(figure)
        canvas.print_png(output_path, metadata={"Software": "Matplotlib 3.11.1"})
    finally:
        figure.clear()


def _new_canvas() -> tuple[Any, Any, Any]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    figure = Figure(figsize=(14.28, 7.72), dpi=100, facecolor="#f5f7fa")
    return figure, FigureCanvasAgg(figure), Polygon


def _configure_axes(figure: Any) -> Any:
    axes = figure.add_axes((0.035, 0.10, 0.93, 0.80), facecolor="#e7f2f8")
    axes.set_xlim(-180, 180)
    axes.set_ylim(-60, 85)
    axes.set_xticks(range(-180, 181, 30))
    axes.set_yticks(range(-60, 81, 20))
    axes.grid(color="white", linewidth=0.8, alpha=0.85)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color("#b7cbd4")
        spine.set_linewidth(0.8)
    return axes


def _draw_boundaries(axes: Any, boundary_rings: Sequence[BoundaryRing], polygon_class: Any) -> None:
    for ring in boundary_rings:
        axes.add_patch(
            polygon_class(
                ring,
                closed=True,
                fill=True,
                facecolor="#dce6d7",
                edgecolor="#becabd",
                linewidth=0.45,
                antialiased=True,
                zorder=1,
            )
        )


def _draw_points(axes: Any, points: Sequence[BenchmarkPoint]) -> None:
    counts = Counter(point.source for point in points)
    for source in SOURCE_ORDER:
        source_points = [point for point in points if point.source == source]
        axes.scatter(
            [point.longitude for point in source_points],
            [point.latitude for point in source_points],
            s=18,
            color=SOURCE_COLORS[source],
            edgecolors="none",
            label=f"{SOURCE_LABELS[source]} ({counts[source]})",
            zorder=3,
        )
    axes.legend(loc="lower left", framealpha=0.85, fontsize=9)


def _add_figure_labels(figure: Any) -> None:
    figure.text(
        0.035,
        0.965,
        "V3 multilingual benchmark - sentence locations",
        color="#20385d",
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.035,
        0.928,
        "300 English benchmark records; coordinates retained across all 85 language files",
        color="#52657d",
        fontsize=10,
        va="top",
    )
    figure.text(
        0.755,
        0.022,
        "Coordinates: benchmark metadata | WGS84 / Plate Carree",
        color="#52657d",
        fontsize=8,
        ha="left",
    )


def _point_from_row(row: dict[str, str | None], path: Path, row_number: int) -> BenchmarkPoint:
    sentence = (row.get("sentence") or "").strip()
    if not sentence:
        raise ValueError(f"{path} row {row_number} has a blank sentence")
    source = (row.get("source") or "").strip()
    if source not in SOURCE_ORDER:
        raise ValueError(f"{path} row {row_number} has unsupported source {source!r}")
    latitude = _coordinate(row.get("latitude"), "latitude", path, row_number)
    longitude = _coordinate(row.get("longitude"), "longitude", path, row_number)
    return BenchmarkPoint(latitude=latitude, longitude=longitude, source=source)


def _coordinate(value: str | None, name: str, path: Path, row_number: int) -> float:
    try:
        coordinate = float(value or "")
    except ValueError as error:
        raise ValueError(f"{path} row {row_number} has invalid {name}") from error
    return _validated_coordinate(coordinate, name, path, row_number)


def _validated_coordinate(coordinate: float, name: str, path: Path, row_number: int) -> float:
    if not math.isfinite(coordinate):
        raise ValueError(f"{path} row {row_number} has non-finite {name}")
    limit = 90.0 if name == "latitude" else 180.0
    if not -limit <= coordinate <= limit:
        raise ValueError(f"{path} row {row_number} has out-of-range {name}")
    return coordinate


def _validate_source_counts(
    points: Sequence[BenchmarkPoint], expected_counts: Mapping[str, int], path: Path
) -> None:
    actual = dict(Counter(point.source for point in points))
    normalized_expected = dict(expected_counts)
    if actual != normalized_expected:
        raise ValueError(
            f"{path} source counts {actual} do not match expected source counts {normalized_expected}"
        )


def _parse_boundary_ring(raw_ring: Any, path: Path, feature_number: int) -> BoundaryRing:
    if not isinstance(raw_ring, list) or len(raw_ring) < 4:
        raise ValueError(f"{path} feature {feature_number} has an invalid exterior ring")
    return tuple(_boundary_point(coordinate, path, feature_number) for coordinate in raw_ring)


def _boundary_point(coordinate: Any, path: Path, feature_number: int) -> tuple[float, float]:
    if not isinstance(coordinate, list) or len(coordinate) < 2:
        raise ValueError(f"{path} feature {feature_number} has an invalid boundary coordinate")
    longitude, latitude = float(coordinate[0]), float(coordinate[1])
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ValueError(f"{path} feature {feature_number} has a non-finite boundary coordinate")
    return longitude, latitude
