# Reproducible V3 Geographic Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add a deterministic, tested V3 map generator, regenerate the map asset, and publish the matching GitHub and Hugging Face card artifacts.

**Architecture:** A focused reporting module validates benchmark CSV rows, parses a committed geometry-only Natural Earth GeoJSON file, and renders a fixed Matplotlib figure. A thin CLI supplies repository defaults. Tests exercise validation and byte-identical regeneration; documentation records the offline data and command contract.

**Tech Stack:** Python 3.12+, standard-library CSV/JSON/dataclasses, Matplotlib 3.11.1 with the Agg backend, pytest, Ruff, ty, uv, Hugging Face Hub CLI.

---

### Task 1: Add the reproducibility test surface

**Files:**
- Create: \`tests/unit/reporting/__init__.py\`
- Create: \`tests/unit/reporting/test_geographic.py\`
- Create: \`tests/fixtures/geographic_map/benchmark.csv\`
- Create: \`tests/fixtures/geographic_map/boundaries.geojson\`

- [ ] **Step 1: Write failing tests for validated points and boundary rings**

~~~python
def test_load_points_rejects_wrong_source_counts(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text(FIXTURE_CSV.replace("website", "description"), encoding="utf-8")

    with pytest.raises(ValueError, match="source counts"):
        load_benchmark_points(path)


def test_load_boundaries_reads_polygon_and_multipolygon_exteriors():
    rings = load_boundary_rings(FIXTURE_BOUNDARIES)

    assert len(rings) == 2
    assert all(len(ring) >= 4 for ring in rings)
~~~

- [ ] **Step 2: Run the focused tests and verify the expected missing-module failure**

Run: \`uv run pytest tests/unit/reporting/test_geographic.py -q\`

Expected: collection fails because \`landuse_sentence_relevance.reporting.geographic\` does not exist.

- [ ] **Step 3: Add only fixture data and test imports/constants**

The CSV fixture must use the production nine-column header and contain rows
covering each source. The GeoJSON fixture must contain one \`Polygon\` and one
\`MultiPolygon\` feature.

- [ ] **Step 4: Commit the red test surface**

~~~bash
git add tests/unit/reporting tests/fixtures/geographic_map
git commit -m "test: specify reproducible map inputs"
~~~

### Task 2: Implement input and boundary validation

**Files:**
- Create: \`src/landuse_sentence_relevance/reporting/__init__.py\`
- Create: \`src/landuse_sentence_relevance/reporting/geographic.py\`
- Modify: \`tests/unit/reporting/test_geographic.py\`

- [ ] **Step 1: Extend tests for production validation and malformed coordinates**

~~~python
def test_load_points_uses_v3_source_quota_contract():
    points = load_benchmark_points(PRODUCTION_BENCHMARK)

    assert len(points) == 300
    assert Counter(point.source for point in points) == {
        "description": 100,
        "website": 100,
        "wikipedia": 100,
    }


def test_load_points_rejects_non_finite_coordinates(tmp_path):
    path = write_production_shape_with_coordinate(tmp_path, "nan")

    with pytest.raises(ValueError, match="latitude"):
        load_benchmark_points(path)
~~~

- [ ] **Step 2: Run tests and confirm the new assertions fail**

Run: \`uv run pytest tests/unit/reporting/test_geographic.py -q\`

Expected: failures identify the absent \`BenchmarkPoint\`, loader, and boundary parser.

- [ ] **Step 3: Implement the minimal typed loader and GeoJSON parser**

~~~python
BoundaryRing = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class BenchmarkPoint:
    latitude: float
    longitude: float
    source: str


def load_benchmark_points(
    path: Path,
    *,
    expected_counts: Mapping[str, int] = V3_SOURCE_COUNTS,
) -> tuple[BenchmarkPoint, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError("benchmark CSV has an unexpected header")
        points = tuple(_point_from_row(row, path) for row in reader)
    _validate_source_counts(points, expected_counts)
    return points


def load_boundary_rings(path: Path) -> tuple[BoundaryRing, ...]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return tuple(ring for feature in _features(document) for ring in _exterior_rings(feature))
~~~

Reject missing columns, blank rows, unsupported sources, non-numeric or
non-finite coordinates, out-of-range coordinates, and source-count mismatch
with actionable \`ValueError\` messages.

- [ ] **Step 4: Run the focused tests and confirm green**

Run: \`uv run pytest tests/unit/reporting/test_geographic.py -q\`

Expected: all input and boundary tests pass.

- [ ] **Step 5: Commit the validation module**

~~~bash
git add src/landuse_sentence_relevance/reporting tests/unit/reporting/test_geographic.py
git commit -m "feat: validate V3 map inputs"
~~~

### Task 3: Add deterministic rendering and regression tests

**Files:**
- Modify: \`src/landuse_sentence_relevance/reporting/geographic.py\`
- Modify: \`tests/unit/reporting/test_geographic.py\`
- Modify: \`pyproject.toml\`
- Modify: \`uv.lock\`

- [ ] **Step 1: Write the failing deterministic-render test**

~~~python
def test_render_map_is_byte_identical_for_identical_inputs(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    render_world_map(FIXTURE_POINTS, FIXTURE_RINGS, first)
    render_world_map(FIXTURE_POINTS, FIXTURE_RINGS, second)

    assert first.read_bytes() == second.read_bytes()
~~~

- [ ] **Step 2: Run the test and verify it fails because rendering is absent**

Run: \`uv run pytest tests/unit/reporting/test_geographic.py::test_render_map_is_byte_identical_for_identical_inputs -q\`

Expected: failure because \`render_world_map\` is not defined.

- [ ] **Step 3: Add the pinned plotting dependency**

Add \`matplotlib==3.11.1\` to the development dependency group and run
\`uv lock\` so the lockfile records the exact renderer and transitive versions.

- [ ] **Step 4: Implement fixed rendering**

~~~python
def render_world_map(
    points: Sequence[BenchmarkPoint],
    boundary_rings: Sequence[BoundaryRing],
    output_path: Path,
) -> None:
    matplotlib.use("Agg")
    # Use fixed 1428x772 output, WGS84 longitude/latitude axes, stable source
    # ordering/colors, fixed labels, fixed limits, and deterministic PNG
    # metadata. Close the figure in a finally block.
~~~

The renderer must not read the clock, random state, locale, network, or
environment-dependent style files. It must create the output parent and write
only the requested PNG. Its implementation must set the Agg backend before
importing pyplot, use a fixed `Figure` size and DPI, use explicit colors and
font settings, and close the figure in `finally`.

- [ ] **Step 5: Run focused rendering tests and confirm green**

Run: \`uv run pytest tests/unit/reporting/test_geographic.py -q\`

Expected: all tests pass and repeated PNG bytes are equal.

- [ ] **Step 6: Commit the rendering implementation**

~~~bash
git add src/landuse_sentence_relevance/reporting/geographic.py tests/unit/reporting/test_geographic.py pyproject.toml uv.lock
git commit -m "feat: render deterministic V3 world map"
~~~

### Task 4: Add the CLI and committed Natural Earth source

**Files:**
- Create: \`scripts/build_v3_world_map.py\`
- Create: \`data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson\`
- Create: \`data/benchmark/v3/assets/README.md\`
- Create: \`tests/unit/test_v3_world_map_script.py\`

- [ ] **Step 1: Write the CLI contract test**

~~~python
def test_cli_regenerates_the_repository_asset(tmp_path):
    output = tmp_path / "map.png"
    completed = subprocess.run(
        [sys.executable, "scripts/build_v3_world_map.py", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.stat().st_size > 10_000
~~~

- [ ] **Step 2: Run the CLI test and verify the expected missing-script failure**

Run: \`uv run pytest tests/unit/test_v3_world_map_script.py -q\`

Expected: failure because \`scripts/build_v3_world_map.py\` is absent.

- [ ] **Step 3: Add the compact boundary source and provenance**

Download Natural Earth 110m Admin 0 country geometry once into temporary
storage, reduce it to a geometry-only \`FeatureCollection\`, validate it, and
commit only the reduced GeoJSON. The asset README must record the upstream URL,
dataset name, 110m scale, public-domain license, and reduction method.

- [ ] **Step 4: Implement the thin CLI**

~~~python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points = load_benchmark_points(args.benchmark)
    rings = load_boundary_rings(args.boundaries)
    render_world_map(points, rings, args.output)
    return 0
~~~

- [ ] **Step 5: Run CLI and focused tests**

Run: \`uv run pytest tests/unit/test_v3_world_map_script.py tests/unit/reporting/test_geographic.py -q\`

Expected: all tests pass.

- [ ] **Step 6: Commit the CLI and source data**

~~~bash
git add scripts/build_v3_world_map.py data/benchmark/v3/assets tests/unit/test_v3_world_map_script.py
git commit -m "feat: add offline V3 map generator"
~~~

### Task 5: Regenerate and document the benchmark map

**Files:**
- Modify: \`data/benchmark/v3/assets/v3-world-distribution.png\`
- Modify: \`docs/v3-translations.md\`
- Modify: \`data/benchmark/v3/README.md\`
- Modify: \`data/benchmark/v3/hf/README.md\`

- [ ] **Step 1: Regenerate from committed inputs**

~~~bash
uv run python scripts/build_v3_world_map.py
~~~

- [ ] **Step 2: Verify reproducibility and output contract**

Run the generator twice into temporary paths and compare SHA-256 hashes.
Verify the committed PNG is 1428x772 RGBA and the source CSV still contains
300 rows with 100 records per source.

- [ ] **Step 3: Document the generator and source asset**

Document the exact command, inputs, output, pinned renderer, and offline
boundary source in project documentation and the Hugging Face card. Keep the
existing map URL/path so the card displays the regenerated asset.

- [ ] **Step 4: Commit the regenerated release artifact and documentation**

~~~bash
git add data/benchmark/v3/assets/v3-world-distribution.png docs/v3-translations.md data/benchmark/v3/README.md data/benchmark/v3/hf/README.md
git commit -m "docs: document reproducible V3 map provenance"
~~~

### Task 6: Full verification, publication, and pull request

**Files:**
- No additional source files unless verification exposes a defect.

- [ ] **Step 1: Run focused and project quality checks**

~~~bash
uv run pytest tests/unit/reporting tests/unit/test_v3_world_map_script.py -q
uv run ruff check src/landuse_sentence_relevance/reporting scripts/build_v3_world_map.py tests/unit/reporting tests/unit/test_v3_world_map_script.py
uv run ty check src tests
~~~

- [ ] **Step 2: Run the repository gauntlet**

Run: \`uv run python scripts/gauntlet.py\`

Expected: exit 0 with every required check passing.

- [ ] **Step 3: Publish only the changed Hub card artifacts**

Upload \`data/benchmark/v3/hf/README.md\` as \`README.md\` and the regenerated
PNG as \`assets/v3-world-distribution.png\` to the existing dataset repo. Verify
live hashes and image URL; do not delete the 85 CSV files, manifest, or other
valid dataset assets.

- [ ] **Step 4: Push and open the PR**

~~~bash
git push -u origin codex/reproducible-v3-map
gh pr create --base main --head codex/reproducible-v3-map --title "feat: make V3 geographic map reproducible" --body "Closes #37. Adds a tested offline generator and regenerates the dataset-card map from committed benchmark and Natural Earth inputs."
~~~

- [ ] **Step 5: Attach, review, and verify the PR**

Attach the PR to the task, request code review, fix all important findings,
wait for CI, and verify the final PR diff contains the generator, tests,
source-data provenance, regenerated PNG, and card documentation.
