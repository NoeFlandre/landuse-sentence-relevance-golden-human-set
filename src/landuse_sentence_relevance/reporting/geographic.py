"""Deterministic geographic coverage map generation for the V3 benchmark."""

from __future__ import annotations

import csv
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
OSM_BASEMAP_SIZE = (1024, 1024)
OSM_MAX_LATITUDE = 85.05112878
WEB_MERCATOR_LIMIT = math.pi


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


def load_osm_basemap(path: Path, *, expected_size: tuple[int, int] = OSM_BASEMAP_SIZE) -> Any:
    """Load and validate the committed OSM raster basemap."""

    import matplotlib.image as mpimg

    image = mpimg.imread(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path} must be an RGB or RGBA PNG")
    width, height = image.shape[1], image.shape[0]
    if (width, height) != expected_size:
        raise ValueError(f"{path} must be {expected_size[0]}x{expected_size[1]}")
    return image


def web_mercator_y(latitude: float) -> float:
    """Convert latitude to the Web Mercator vertical coordinate used by OSM tiles."""

    clipped = max(-OSM_MAX_LATITUDE, min(OSM_MAX_LATITUDE, latitude))
    radians = math.radians(clipped)
    return math.log(math.tan(math.pi / 4.0 + radians / 2.0))


def render_world_map(
    points: Sequence[BenchmarkPoint],
    basemap: Any,
    output_path: Path,
) -> None:
    """Render the fixed V3 coverage map without network or clock input."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    figure, canvas = _new_canvas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        axes = _configure_axes(figure)
        axes.imshow(
            basemap,
            extent=(-180.0, 180.0, -WEB_MERCATOR_LIMIT, WEB_MERCATOR_LIMIT),
            origin="upper",
            interpolation="bilinear",
            aspect="auto",
            zorder=0,
        )
        _draw_points(axes, points)
        _add_figure_labels(figure)
        canvas.print_png(output_path, metadata={"Software": "Matplotlib 3.11.1"})
    finally:
        figure.clear()


def _new_canvas() -> tuple[Any, Any]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.28, 7.72), dpi=100, facecolor="#f5f7fa")
    return figure, FigureCanvasAgg(figure)


def _configure_axes(figure: Any) -> Any:
    axes = figure.add_axes((0.035, 0.10, 0.93, 0.80), facecolor="#a8d0df")
    axes.set_xlim(-180, 180)
    axes.set_ylim(-WEB_MERCATOR_LIMIT, WEB_MERCATOR_LIMIT)
    axes.set_xticks(range(-180, 181, 30))
    latitudes = tuple(range(-80, 81, 20))
    axes.set_yticks([web_mercator_y(latitude) for latitude in latitudes])
    axes.set_yticklabels([f"{latitude}°" for latitude in latitudes])
    axes.grid(color="white", linewidth=0.8, alpha=0.72)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color("#789eaa")
        spine.set_linewidth(0.8)
    return axes


def _draw_points(axes: Any, points: Sequence[BenchmarkPoint]) -> None:
    counts = Counter(point.source for point in points)
    for source in SOURCE_ORDER:
        _draw_source_points(axes, points, source, counts[source])
    axes.legend(loc="lower left", framealpha=0.88, fontsize=9)


def _draw_source_points(axes: Any, points: Sequence[BenchmarkPoint], source: str, count: int) -> None:
    source_points = [point for point in points if point.source == source]
    axes.scatter(
        [point.longitude for point in source_points],
        [web_mercator_y(point.latitude) for point in source_points],
        s=24,
        color=SOURCE_COLORS[source],
        edgecolors="none",
        alpha=0.82,
        label=f"{SOURCE_LABELS[source]} ({count})",
        zorder=3,
    )


def _add_figure_labels(figure: Any) -> None:
    figure.text(
        0.035,
        0.965,
        "V3 benchmark coverage - 300 sentences plotted",
        color="#20385d",
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.035,
        0.928,
        "Description, Website, and Wikipedia sources | coordinates retained across all 85 language files",
        color="#52657d",
        fontsize=10,
        va="top",
    )
    figure.text(
        0.590,
        0.022,
        "Basemap: © OpenStreetMap contributors | Web Mercator",
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
