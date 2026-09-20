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

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError(f"{path} must contain a GeoJSON FeatureCollection")

    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain at least one GeoJSON feature")

    rings: list[BoundaryRing] = []
    for feature_number, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise ValueError(f"{path} feature {feature_number} is not an object")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"{path} feature {feature_number} has no geometry")
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            polygon_coordinates = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygon_coordinates = coordinates
        else:
            raise ValueError(f"{path} feature {feature_number} uses unsupported geometry {geometry_type!r}")
        if not isinstance(polygon_coordinates, list):
            raise ValueError(f"{path} feature {feature_number} has malformed polygon coordinates")
        for polygon in polygon_coordinates:
            if not isinstance(polygon, list) or not polygon:
                raise ValueError(f"{path} feature {feature_number} has an empty polygon")
            rings.append(_parse_boundary_ring(polygon[0], path, feature_number))
    return tuple(rings)


def render_world_map(
    points: Sequence[BenchmarkPoint],
    boundary_rings: Sequence[BoundaryRing],
    output_path: Path,
) -> None:
    """Render the fixed V3 coverage map without network or clock input."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    source_colors = {
        "description": "#e58a13",
        "website": "#477ff0",
        "wikipedia": "#28a67d",
    }
    source_labels = {
        "description": "Description",
        "website": "Website",
        "wikipedia": "Wikipedia",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.28, 7.72), dpi=100, facecolor="#f5f7fa")
    canvas = FigureCanvasAgg(figure)
    try:
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

        for ring in boundary_rings:
            axes.add_patch(
                Polygon(
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

        counts = Counter(point.source for point in points)
        for source in SOURCE_ORDER:
            source_points = [point for point in points if point.source == source]
            axes.scatter(
                [point.longitude for point in source_points],
                [point.latitude for point in source_points],
                s=18,
                color=source_colors[source],
                edgecolors="none",
                label=f"{source_labels[source]} ({counts[source]})",
                zorder=3,
            )

        axes.legend(loc="lower left", framealpha=0.85, fontsize=9)
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
        canvas.print_png(output_path, metadata={"Software": "Matplotlib 3.11.1"})
    finally:
        figure.clear()


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
    ring: list[tuple[float, float]] = []
    for coordinate in raw_ring:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            raise ValueError(f"{path} feature {feature_number} has an invalid boundary coordinate")
        longitude, latitude = float(coordinate[0]), float(coordinate[1])
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError(f"{path} feature {feature_number} has a non-finite boundary coordinate")
        ring.append((longitude, latitude))
    return tuple(ring)
