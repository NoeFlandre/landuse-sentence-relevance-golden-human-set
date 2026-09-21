"""Deterministic geographic coverage map generation for the V3 benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
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
    "description": "#d94801",
    "website": "#2c6fbb",
    "wikipedia": "#1b7837",
}
SOURCE_LABELS = {
    "description": "Description",
    "website": "Website",
    "wikipedia": "Wikipedia",
}

OCEAN_COLOR = "#cfe2f3"
LAND_COLOR = "#e8e0d0"
LAND_EDGE_COLOR = "#b8aa90"
FIGSIZE = (16.0, 8.0)
DPI = 100
MIN_RING_POINTS = 3

LandRing = tuple[tuple[float, float], ...]


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


def load_land_basemap(path: Path, *, expected_sha256: str | None = None) -> tuple[LandRing, ...]:
    """Load the committed Natural Earth landmass GeoJSON as exterior rings."""

    raw = path.read_bytes()
    if expected_sha256 is not None:
        actual = sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"{path} checksum {actual} does not match {expected_sha256}")
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError(f"{path} must contain a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain at least one land feature")
    rings = tuple(ring for feature in features for ring in _feature_rings(feature, path))
    if not rings:
        raise ValueError(f"{path} contains no drawable land rings")
    return rings


def render_world_map(
    points: Sequence[BenchmarkPoint],
    land: Sequence[LandRing],
    output_path: Path,
) -> None:
    """Render the fixed V3 coverage map without network or clock input."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    figure, canvas = _new_canvas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        axes = _configure_axes(figure)
        _draw_land(axes, land)
        _draw_points(axes, points)
        _add_figure_labels(figure, points)
        canvas.print_png(output_path, metadata={"Software": "Matplotlib 3.11.1"})
    finally:
        figure.clear()


def _new_canvas() -> tuple[Any, Any]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=FIGSIZE, dpi=DPI, facecolor="#ffffff")
    return figure, FigureCanvasAgg(figure)


def _configure_axes(figure: Any) -> Any:
    axes = figure.add_axes((0.045, 0.06, 0.91, 0.86), facecolor=OCEAN_COLOR)
    axes.set_xlim(-180.0, 180.0)
    axes.set_ylim(-90.0, 90.0)
    axes.set_xticks(range(-180, 181, 30))
    axes.set_yticks(range(-90, 91, 30))
    axes.grid(True, color="#ffffff", linewidth=0.3, alpha=0.4)
    axes.tick_params(colors="#666666", labelsize=7)
    axes.set_aspect("equal", adjustable="box")
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color("#b0b0b0")
        spine.set_linewidth(0.4)
    return axes


def _draw_land(axes: Any, land: Sequence[LandRing]) -> None:
    import matplotlib.patches as mpatches

    for ring in land:
        axes.add_patch(
            mpatches.Polygon(
                ring,
                closed=True,
                facecolor=LAND_COLOR,
                edgecolor=LAND_EDGE_COLOR,
                linewidth=0.2,
                zorder=1,
            )
        )


def _draw_points(axes: Any, points: Sequence[BenchmarkPoint]) -> None:
    counts = Counter(point.source for point in points)
    for source in SOURCE_ORDER:
        _draw_source_points(axes, points, source, counts[source])
    legend = axes.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        fontsize=7,
        markerscale=1.2,
        borderpad=0.6,
        labelspacing=0.5,
    )
    legend.get_frame().set_edgecolor("#b0b0b0")
    legend.get_frame().set_linewidth(0.4)


def _draw_source_points(axes: Any, points: Sequence[BenchmarkPoint], source: str, count: int) -> None:
    source_points = [point for point in points if point.source == source]
    axes.scatter(
        [point.longitude for point in source_points],
        [point.latitude for point in source_points],
        s=7,
        color=SOURCE_COLORS[source],
        edgecolors="none",
        alpha=0.85,
        label=f"{SOURCE_LABELS[source]} ({count})",
        zorder=3,
    )


def _add_figure_labels(figure: Any, points: Sequence[BenchmarkPoint]) -> None:
    figure.text(
        0.5,
        0.965,
        f"V3 benchmark coverage - {len(points):,} sentences plotted",
        color="#333333",
        fontsize=11,
        ha="center",
        va="top",
    )
    figure.text(
        0.955,
        0.012,
        "Land: Natural Earth 110m (public domain) | Equirectangular",
        color="#666666",
        fontsize=7,
        ha="right",
    )


def _feature_rings(feature: Any, path: Path) -> tuple[LandRing, ...]:
    if not isinstance(feature, dict):
        raise ValueError(f"{path} contains a non-object feature")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return ()
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon":
        polygons: Any = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        return ()
    if not isinstance(polygons, list):
        raise ValueError(f"{path} contains malformed {kind} coordinates")
    return tuple(
        ring
        for polygon in polygons
        if polygon
        for ring in (_exterior_ring(polygon[0], path),)
        if len(ring) >= MIN_RING_POINTS
    )


def _exterior_ring(ring: Any, path: Path) -> LandRing:
    if not isinstance(ring, list):
        raise ValueError(f"{path} contains a malformed exterior ring")
    return tuple((float(vertex[0]), float(vertex[1])) for vertex in ring)


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
