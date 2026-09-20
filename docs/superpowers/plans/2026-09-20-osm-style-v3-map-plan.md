# OSM-Style V3 Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V3 map's country-outline graphic with a deterministic OpenStreetMap zoom-2 physical basemap and source-colored benchmark points.

**Architecture:** A maintenance-only CLI downloads and verifies the fixed 4x4 OpenStreetMap z2 tile set, stitches it into a committed 1024x1024 PNG, and records per-tile and mosaic checksums in a manifest. The normal map CLI remains offline: it validates the English benchmark, loads the committed PNG, projects points to the Web Mercator coordinates used by the tiles, and renders the final card asset with the existing Matplotlib version.

**Tech Stack:** Python 3.12+, Matplotlib 3.11.1 Agg, Pillow 12.3.0, NumPy, standard-library urllib/hashlib/json, pytest, Hypothesis, Ruff, ty, mutmut, MkDocs.

---

## File map

- Modify `src/landuse_sentence_relevance/reporting/geographic.py` to replace GeoJSON country-boundary loading with validated OSM PNG loading, Web Mercator conversion, and the physical-map renderer.
- Modify `scripts/build_v3_world_map.py` to accept `--basemap` and call the offline renderer.
- Create `scripts/prepare_v3_osm_basemap.py` as the only network-aware asset maintenance command. It downloads exactly 16 z2 tiles, checks their hashes, stitches them, and writes the manifest.
- Modify `pyproject.toml` and `uv.lock` to pin Pillow for deterministic tile assembly.
- Delete `data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`.
- Create `data/benchmark/v3/assets/osm-world-z2.png` and `data/benchmark/v3/assets/osm-world-z2-manifest.json`.
- Modify `data/benchmark/v3/assets/README.md`, `data/benchmark/v3/README.md`, `docs/v3-translations.md`, and `data/benchmark/v3/hf/README.md`.
- Modify `tests/unit/reporting/test_geographic.py` and `tests/unit/test_v3_world_map_script.py`; delete the obsolete GeoJSON fixture; create `tests/unit/test_prepare_v3_osm_basemap.py`.

### Task 1: Add failing basemap and projection tests

**Files:**
- Modify: `tests/unit/reporting/test_geographic.py`
- Create: `tests/unit/test_prepare_v3_osm_basemap.py`

- [ ] **Step 1: Replace the GeoJSON fixture tests with PNG-contract tests**

Use an in-test PNG created with Pillow so tests do not need a large binary fixture. The first test must call the public loader with an explicit expected size; the production default remains `(1024, 1024)`.

~~~
from PIL import Image

from landuse_sentence_relevance.reporting.geographic import (
    load_osm_basemap,
    web_mercator_y,
)


def write_basemap(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    Image.new("RGBA", size, (190, 215, 230, 255)).save(path)


def test_load_osm_basemap_validates_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "basemap.png"
    write_basemap(path)

    image = load_osm_basemap(path, expected_size=(8, 8))

    assert image.shape == (8, 8, 4)


def test_load_osm_basemap_rejects_wrong_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "basemap.png"
    write_basemap(path, size=(8, 4))

    with pytest.raises(ValueError, match="1024"):
        load_osm_basemap(path)


def test_web_mercator_y_is_monotonic_and_symmetric() -> None:
    assert web_mercator_y(0.0) == pytest.approx(0.0)
    assert web_mercator_y(60.0) == pytest.approx(-web_mercator_y(-60.0))
    assert web_mercator_y(90.0) == pytest.approx(web_mercator_y(85.05112878))
~~~

- [ ] **Step 2: Add failing tile-preparation unit tests**

Test pure functions only; no test may call the network. The tests must cover the exact 4x4 coordinate set, hash verification, and deterministic stitching.

~~~
def test_tile_coordinates_for_zoom_two_are_four_by_four() -> None:
    assert list(tile_coordinates(2)) == [(x, y) for y in range(4) for x in range(4)]


def test_stitch_tiles_places_tiles_in_row_major_order() -> None:
    tiles = {
        (0, 0): solid_tile((255, 0, 0, 255)),
        (1, 0): solid_tile((0, 255, 0, 255)),
        (0, 1): solid_tile((0, 0, 255, 255)),
        (1, 1): solid_tile((255, 255, 0, 255)),
    }

    mosaic = stitch_tiles(tiles, columns=2, rows=2, tile_size=2)

    assert mosaic.getpixel((0, 0)) == (255, 0, 0, 255)
    assert mosaic.getpixel((2, 0)) == (0, 255, 0, 255)
    assert mosaic.getpixel((0, 2)) == (0, 0, 255, 255)
    assert mosaic.getpixel((2, 2)) == (255, 255, 0, 255)


def test_verify_tile_hash_rejects_changed_bytes() -> None:
    with pytest.raises(ValueError, match="checksum"):
        verify_tile_hash(b"changed", "0" * 64, "2/0/0")
~~~

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

~~~
uv run pytest tests/unit/reporting/test_geographic.py tests/unit/test_prepare_v3_osm_basemap.py -q
~~~

Expected: collection or import failures because `load_osm_basemap`, `web_mercator_y`, `tile_coordinates`, `stitch_tiles`, and `verify_tile_hash` do not yet exist.

- [ ] **Step 4: Commit the RED tests**

~~~
git add tests/unit/reporting/test_geographic.py tests/unit/test_prepare_v3_osm_basemap.py
git commit -m "test: define OSM basemap contracts"
~~~

### Task 2: Implement validated OSM loading and deterministic rendering

**Files:**
- Modify: `src/landuse_sentence_relevance/reporting/geographic.py`
- Modify: `tests/unit/reporting/test_geographic.py`

- [ ] **Step 1: Implement the PNG loader and Web Mercator conversion**

Keep the existing CSV validation and source constants unchanged. Replace the GeoJSON types and loader with:

~~~
OSM_BASEMAP_SIZE = (1024, 1024)
OSM_MAX_LATITUDE = 85.05112878
WEB_MERCATOR_LIMIT = math.pi


def load_osm_basemap(path: Path, *, expected_size: tuple[int, int] = OSM_BASEMAP_SIZE) -> Any:
    import matplotlib.image as mpimg

    image = mpimg.imread(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path} must be an RGB or RGBA PNG")
    width, height = image.shape[1], image.shape[0]
    if (width, height) != expected_size:
        raise ValueError(f"{path} must be {expected_size[0]}x{expected_size[1]}")
    return image


def web_mercator_y(latitude: float) -> float:
    clipped = max(-OSM_MAX_LATITUDE, min(OSM_MAX_LATITUDE, latitude))
    radians = math.radians(clipped)
    return math.log(math.tan(math.pi / 4.0 + radians / 2.0))
~~~

- [ ] **Step 2: Implement the physical basemap renderer**

Change the public renderer signature to `render_world_map(points, basemap, output_path)`. Configure the axes with `xlim=(-180, 180)` and `ylim=(-math.pi, math.pi)`, draw the loaded image with `origin="upper"` and extent `(-180, 180, -math.pi, math.pi)`, and put latitude ticks at `web_mercator_y(latitude)` while retaining degree labels. Use the existing source palette and legend counts; each source layer must plot `web_mercator_y(point.latitude)` rather than raw latitude.

Use these exact source colors and labels:

~~~
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
~~~

Add the attribution directly to the output footer: `Basemap: © OpenStreetMap contributors | Web Mercator`. Keep the fixed 1428x772 RGBA PNG output and `Matplotlib 3.11.1` metadata.

- [ ] **Step 3: Extend the rendering tests**

Render two files with the same fixture image and points and assert identical bytes. Assert that `web_mercator_y(60)` is used by checking the helper directly; the deterministic image test then protects the complete render contract.

- [ ] **Step 4: Run focused tests and verify GREEN**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run pytest tests/unit/reporting/test_geographic.py -q
~~~

Expected: all geographic unit tests pass.

- [ ] **Step 5: Commit the renderer**

~~~
git add src/landuse_sentence_relevance/reporting/geographic.py tests/unit/reporting/test_geographic.py
git commit -m "feat: render V3 map over OSM basemap"
~~~

### Task 3: Implement the pinned OSM tile preparation CLI

**Files:**
- Create: `scripts/prepare_v3_osm_basemap.py`
- Modify: `tests/unit/test_prepare_v3_osm_basemap.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add the pinned Pillow dependency**

Add `"pillow==12.3.0"` to the dev dependency group and run:

~~~
uv lock
~~~

Expected: `uv.lock` records Pillow 12.3.0 without unrelated dependency changes.

- [ ] **Step 2: Implement pure tile functions**

The module must expose these testable functions:

~~~
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ZOOM = 2
TILE_SIZE = 256


def tile_coordinates(zoom: int) -> tuple[tuple[int, int], ...]:
    width = 2**zoom
    return tuple((x, y) for y in range(width) for x in range(width))


def verify_tile_hash(data: bytes, expected: str, tile_name: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{tile_name} checksum {actual} does not match {expected}")


def stitch_tiles(
    tiles: Mapping[tuple[int, int], Image.Image],
    *,
    columns: int,
    rows: int,
    tile_size: int,
) -> Image.Image:
    mosaic = Image.new("RGBA", (columns * tile_size, rows * tile_size))
    for (x, y), tile in sorted(tiles.items(), key=lambda item: (item[0][1], item[0][0])):
        mosaic.paste(tile.convert("RGBA"), (x * tile_size, y * tile_size))
    return mosaic
~~~

- [ ] **Step 3: Implement manifest-driven downloading**

The CLI must accept `--manifest`, `--output`, and `--record-checksums`. In normal mode it reads the manifest, downloads each of the 16 URLs with a descriptive User-Agent, verifies every recorded tile SHA-256, stitches the tiles, verifies `mosaic_sha256`, and writes the output. In recording mode it downloads the same coordinates, writes tile hashes and the mosaic hash, and uses the fixed retrieval date supplied by `--retrieved-at YYYY-MM-DD`; the command must reject a missing date in recording mode.

The manifest fields are exactly:

~~~
{
  "source_url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  "zoom": 2,
  "tile_size": 256,
  "retrieved_at": "2026-09-20",
  "attribution": "© OpenStreetMap contributors",
  "license": "Open Database License (ODbL) 1.0",
  "tiles": [{"x": 0, "y": 0, "sha256": "..."}],
  "mosaic_sha256": "..."
}
~~~

- [ ] **Step 4: Run preparation tests and commit**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run pytest tests/unit/test_prepare_v3_osm_basemap.py -q
git add scripts/prepare_v3_osm_basemap.py tests/unit/test_prepare_v3_osm_basemap.py pyproject.toml uv.lock
git commit -m "feat: add pinned OSM basemap preparation"
~~~

Expected: all preparation tests pass and the commit contains no downloaded tiles or temporary files.

### Task 4: Build the checked-in asset and wire the CLI

**Files:**
- Modify: `scripts/build_v3_world_map.py`
- Create: `data/benchmark/v3/assets/osm-world-z2.png`
- Create: `data/benchmark/v3/assets/osm-world-z2-manifest.json`
- Delete: `data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`
- Modify: `data/benchmark/v3/assets/README.md`
- Modify: `tests/unit/test_v3_world_map_script.py`

- [ ] **Step 1: Generate the pinned OSM mosaic in task-scoped temporary storage**

Run the maintenance command with the verified source and fixed date:

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/prepare_v3_osm_basemap.py \
  --manifest data/benchmark/v3/assets/osm-world-z2-manifest.json \
  --output data/benchmark/v3/assets/osm-world-z2.png \
  --record-checksums \
  --retrieved-at 2026-09-20
~~~

Expected: one 1024x1024 RGBA PNG and one JSON manifest; no tile cache remains in the repository.

- [ ] **Step 2: Update the map CLI defaults**

Replace `--boundaries` with `--basemap`, defaulting to `data/benchmark/v3/assets/osm-world-z2.png`. The CLI must load points, load the basemap, and call `render_world_map(points, basemap, output)`.

- [ ] **Step 3: Regenerate and test the final map**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/build_v3_world_map.py
uv run pytest tests/unit/test_v3_world_map_script.py -q
file data/benchmark/v3/assets/v3-world-distribution.png
~~~

Expected: `PNG image data, 1428 x 772, 8-bit/color RGBA` and a passing CLI test. Re-running the CLI twice must produce the same SHA-256.

- [ ] **Step 4: Document the asset provenance**

Replace Natural Earth text in `data/benchmark/v3/assets/README.md` with the OSM tile URL, z2/4x4 layout, manifest checksum contract, attribution link, ODbL link, and the explicit statement that normal map generation performs no network access.

- [ ] **Step 5: Commit the asset and CLI wiring**

~~~
git add scripts/build_v3_world_map.py data/benchmark/v3/assets tests/unit/test_v3_world_map_script.py
git rm data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson
git commit -m "build: replace V3 map background with OSM mosaic"
~~~

### Task 5: Update documentation and the Hugging Face card source

**Files:**
- Modify: `data/benchmark/v3/README.md`
- Modify: `docs/v3-translations.md`
- Modify: `data/benchmark/v3/hf/README.md`

- [ ] **Step 1: Replace all Natural Earth and country-outline claims**

Describe the map as an OSM standard z2 physical basemap with Description, Website, and Wikipedia source colors. State that coordinates are benchmark metadata, that all languages retain the same coordinates, and that the map is not a population or source-density estimate.

- [ ] **Step 2: Add reproducibility and attribution instructions**

Document `uv run python scripts/build_v3_world_map.py` as the offline command; link to `scripts/prepare_v3_osm_basemap.py` for asset maintenance; link to the OSM copyright and ODbL pages; and link to the committed manifest for exact source checksums.

- [ ] **Step 3: Validate documentation and commit**

~~~
rg -n "Natural Earth|country outlines|country-boundary" data/benchmark/v3 docs --glob '!superpowers/**'
uv run mkdocs build --strict --clean --site-dir state/osm-style-docs
git diff --check
git add data/benchmark/v3/README.md docs/v3-translations.md data/benchmark/v3/hf/README.md
git commit -m "docs: document OSM V3 map provenance"
~~~

Expected: the search returns no obsolete map description, MkDocs succeeds, and the diff has no whitespace errors.

### Task 6: Run the full quality gate and prepare the release

**Files:**
- Modify: none beyond the generated asset if regeneration changes bytes.

- [ ] **Step 1: Run focused tests, lint, types, and architecture checks**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run pytest tests/unit/reporting tests/unit/test_prepare_v3_osm_basemap.py tests/unit/test_v3_world_map_script.py -q
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run ty check src tests scripts
uv run python scripts/check_architecture.py
~~~

Expected: all focused tests pass and all static checks exit zero.

- [ ] **Step 2: Verify byte-level determinism and asset contracts**

Generate two temporary output files with the CLI, compare SHA-256 values, and assert the committed OSM mosaic and final map dimensions. The two generated hashes must be identical to each other; the final committed PNG must be the same hash after the second generation.

- [ ] **Step 3: Run the full project gauntlet**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/gauntlet.py --skip-docker
~~~

Expected: `GAUNTLET PASSED`. The local Docker step may be separately reported as unavailable if the daemon is not running; GitHub CI must run the complete workflow.

- [ ] **Step 4: Review, commit status, and publish**

Run `git diff --check origin/main...HEAD`, inspect the complete diff, push `codex/osm-style-v3-map`, open a PR, attach it to the Codex task, and wait for GitHub CI. Merge only after CI passes.

- [ ] **Step 5: Update and verify Hugging Face**

Upload only the regenerated `README.md` and `assets/v3-world-distribution.png` to `NoeFlandre/landuse-sentence-relevance-golden-human-set`. Download both live files, compare the PNG SHA-256 to the local committed asset, verify the card contains the OSM attribution and generator link, and verify the repository still has exactly 85 language CSVs plus its existing manifest/card/map files.

- [ ] **Step 6: Clean task-owned temporary storage**

Remove only the task-scoped OSM tile staging directory, UV/Matplotlib caches, Hugging Face verification directory, and managed worktree after the PR is merged and the primary checkout is fast-forwarded. Confirm the primary checkout is clean and no task-specific temporary path remains.
