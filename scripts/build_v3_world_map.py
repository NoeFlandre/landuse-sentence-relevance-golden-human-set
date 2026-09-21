"""Build the deterministic V3 geographic coverage map."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from landuse_sentence_relevance.reporting.geographic import (
    load_benchmark_points,
    load_land_basemap,
    render_world_map,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PROJECT_ROOT / "data/benchmark/v3/final/v3-final.csv"
DEFAULT_BASEMAP = PROJECT_ROOT / "data/benchmark/v3/assets/ne-110m-land.geojson"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/benchmark/v3/assets/v3-world-distribution.png"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--basemap", type=Path, default=DEFAULT_BASEMAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    points = load_benchmark_points(args.benchmark)
    basemap = load_land_basemap(args.basemap)
    render_world_map(points, basemap, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
