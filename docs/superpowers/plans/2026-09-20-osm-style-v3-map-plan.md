# OSM-Style V3 Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. The steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the country-outline graphic of the V3 map. Use a deterministic OpenStreetMap zoom-2 physical basemap and benchmark points in the colors of the sources.

**Architecture:** A CLI for maintenance only refreshes or verifies the fixed 4x4 OpenStreetMap z2 tile snapshot. It stitches the snapshot into a committed PNG of 1024x1024. It records the checksums of each tile and of the mosaic in a manifest. The repository contains the exact 16 tile PNGs. The normal map CLI stays offline. It validates the English benchmark and loads the committed PNG. It projects the points to the Web Mercator coordinates that the tiles use. It renders the final card asset with the existing Matplotlib version.

**Tech Stack:** Python 3.12+, Matplotlib 3.11.1 Agg, Pillow 12.3.0, NumPy, standard-library urllib/hashlib/json, pytest, Hypothesis, Ruff, ty, mutmut, MkDocs.

---

## File map

- Modify `src/landuse_sentence_relevance/reporting/geographic.py`. Replace the loading of GeoJSON country boundaries with the loading of a validated OSM PNG, a Web Mercator conversion, and the physical-map renderer.
- Modify `scripts/build_v3_world_map.py`. Make it accept `--basemap` and call the offline renderer.
- Create `scripts/prepare_v3_osm_basemap.py`. It is the only command for asset maintenance that uses the network. By default, it verifies exactly 16 vendored z2 tiles. The explicit `--download --record-checksums` option refreshes those bytes, checks their hashes, stitches them, and writes the manifest.
- Modify `pyproject.toml` and `uv.lock`. Pin Pillow for the deterministic tile assembly.
- Delete `data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`.
- Create `data/benchmark/v3/assets/osm-world-z2.png` and `data/benchmark/v3/assets/osm-world-z2-manifest.json`.
- Create the 16 source snapshots in `data/benchmark/v3/assets/osm-tiles/z2/<x>/<y>.png`.
- Modify `data/benchmark/v3/assets/README.md`, `data/benchmark/v3/README.md`, `docs/v3-translations.md`, and `data/benchmark/v3/hf/README.md`.
- Modify `tests/unit/reporting/test_geographic.py` and `tests/unit/test_v3_world_map_script.py`. Delete the obsolete GeoJSON fixture. Create `tests/unit/test_prepare_v3_osm_basemap.py`.

### Task 1: Add failing basemap and projection tests

**Files:**
- Modify: `tests/unit/reporting/test_geographic.py`
- Create: `tests/unit/test_prepare_v3_osm_basemap.py`

- [ ] **Step 1: Replace the GeoJSON fixture tests with PNG-contract tests**

Use a PNG that the test creates with Pillow. The tests then do not need a large binary fixture. The first test must call the public loader with an explicit expected size. The production default stays `(1024, 1024)`.

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

Test only pure functions. No test can call the network. The tests must cover the exact 4x4 coordinate set, the hash verification, and the deterministic stitching.

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

Expected: collection or import failures. `load_osm_basemap`, `web_mercator_y`, `tile_coordinates`, `stitch_tiles`, and `verify_tile_hash` do not exist yet.

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

Do not change the existing CSV validation and the source constants. Replace the GeoJSON types and the loader with this code:

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

Change the public renderer signature to `render_world_map(points, basemap, output_path)`. Configure the axes with `xlim=(-180, 180)` and `ylim=(-math.pi, math.pi)`. Draw the loaded image with `origin="upper"` and the extent `(-180, 180, -math.pi, math.pi)`. Put the latitude ticks at `web_mercator_y(latitude)` and keep the degree labels. Use the existing source palette and the legend counts. Each source layer must plot `web_mercator_y(point.latitude)`. It must not plot the raw latitude.

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

Add the attribution directly to the footer of the output: `Basemap: © OpenStreetMap contributors | Web Mercator`. Keep the fixed RGBA PNG output of 1428x772 and the `Matplotlib 3.11.1` metadata.

- [ ] **Step 3: Extend the rendering tests**

Render two files with the same fixture image and the same points. Assert that the bytes are identical. Check the helper directly to assert that the renderer uses `web_mercator_y(60)`. The deterministic image test then protects the complete render contract.

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

Add `"pillow==12.3.0"` to the dev dependency group. Then run:

~~~
uv lock
~~~

Expected: `uv.lock` records Pillow 12.3.0 and no unrelated dependency changes.

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

The CLI must accept `--manifest`, `--output`, and `--record-checksums`. In normal mode, it does these steps:

1. It reads the manifest.
2. It downloads each of the 16 URLs with a descriptive User-Agent.
3. It verifies every recorded tile SHA-256.
4. It stitches the tiles.
5. It verifies `mosaic_sha256`.
6. It writes the output.

In recording mode, it downloads the same coordinates. It writes the tile hashes and the mosaic hash. It uses the fixed retrieval date that `--retrieved-at YYYY-MM-DD` supplies. The command must reject a missing date in recording mode.

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

Expected: all preparation tests pass. The commit contains no downloaded tiles and no temporary files.

### Task 4: Build the checked-in asset and wire the CLI

**Files:**
- Modify: `scripts/build_v3_world_map.py`
- Create: `data/benchmark/v3/assets/osm-world-z2.png`
- Create: `data/benchmark/v3/assets/osm-world-z2-manifest.json`
- Delete: `data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`
- Modify: `data/benchmark/v3/assets/README.md`
- Modify: `tests/unit/test_v3_world_map_script.py`

- [ ] **Step 1: Generate the pinned OSM mosaic in task-scoped temporary storage**

Run the maintenance command with the verified source and the fixed date:

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/prepare_v3_osm_basemap.py \
  --manifest data/benchmark/v3/assets/osm-world-z2-manifest.json \
  --output data/benchmark/v3/assets/osm-world-z2.png \
  --tiles-dir data/benchmark/v3/assets/osm-tiles/z2 \
  --download \
  --record-checksums \
  --retrieved-at 2026-09-20
~~~

Expected: one RGBA PNG of 1024x1024, one JSON manifest, and 16 vendored tile PNG files. A later run without `--download` must rebuild and verify the same mosaic without network access.

- [ ] **Step 2: Update the map CLI defaults**

Replace `--boundaries` with `--basemap`. Its default is `data/benchmark/v3/assets/osm-world-z2.png`. The CLI must load the points, load the basemap, and call `render_world_map(points, basemap, output)`.

- [ ] **Step 3: Regenerate and test the final map**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/build_v3_world_map.py
uv run pytest tests/unit/test_v3_world_map_script.py -q
file data/benchmark/v3/assets/v3-world-distribution.png
~~~

Expected: `PNG image data, 1428 x 772, 8-bit/color RGBA` and a passing CLI test. Two runs of the CLI must produce the same SHA-256.

- [ ] **Step 4: Document the asset provenance**

In `data/benchmark/v3/assets/README.md`, replace the Natural Earth text. Add these items: the OSM tile URL, the z2/4x4 layout, the manifest checksum contract, the attribution link, the ODbL link, and the explicit statement that the normal map generation does not use the network.

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

Describe the map as an OSM standard z2 physical basemap with the source colors Description, Website, and Wikipedia. State that the coordinates are benchmark metadata. State that all languages keep the same coordinates. State that the map is not an estimate of the population or of the source density.

- [ ] **Step 2: Add reproducibility and attribution instructions**

Document `uv run python scripts/build_v3_world_map.py` as the offline command. Link to `scripts/prepare_v3_osm_basemap.py` for the asset maintenance. Link to the OSM copyright page and the ODbL page. Link to the committed manifest for the exact source checksums.

- [ ] **Step 3: Validate documentation and commit**

~~~
rg -n "Natural Earth|country outlines|country-boundary" data/benchmark/v3 docs --glob '!docs/superpowers/**'
uv run mkdocs build --strict --clean --site-dir state/osm-style-docs
git diff --check
git add data/benchmark/v3/README.md docs/v3-translations.md data/benchmark/v3/hf/README.md
git commit -m "docs: document OSM V3 map provenance"
~~~

Expected: the search returns no obsolete map description. MkDocs succeeds. The diff has no whitespace errors.

### Task 6: Run the full quality gate and prepare the release

**Files:**
- Modify: none, except the generated asset if the regeneration changes bytes.

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

Expected: all focused tests pass. All static checks exit with zero.

- [ ] **Step 2: Verify byte-level determinism and asset contracts**

Generate two temporary output files with the CLI. Compare their SHA-256 values. Assert the dimensions of the committed OSM mosaic and of the final map. The two generated hashes must be identical to each other. After the second generation, the final committed PNG must have the same hash.

- [ ] **Step 3: Run the full project gauntlet**

~~~
MPLCONFIGDIR=/private/tmp/landuse-osm-style-mpl \
UV_CACHE_DIR=/private/tmp/landuse-osm-style-uv \
uv run python scripts/gauntlet.py --skip-docker
~~~

Expected: `GAUNTLET PASSED`. If the Docker daemon does not run, report the local Docker step separately as unavailable. GitHub CI must run the complete workflow.

- [ ] **Step 4: Review, commit status, and publish**

Run `git diff --check origin/main...HEAD`. Inspect the complete diff. Push `codex/osm-style-v3-map`. Open a PR. Attach it to the Codex task. Wait for GitHub CI. Merge only after CI passes.

- [ ] **Step 5: Update and verify Hugging Face**

Upload only the regenerated `README.md` and `assets/v3-world-distribution.png` to `NoeFlandre/landuse-sentence-relevance-golden-human-set`. Download both live files. Compare the PNG SHA-256 with the local committed asset. Verify that the card contains the OSM attribution and the generator link. Verify that the repository still has exactly 85 language CSV files and its existing manifest, card, and map files.

- [ ] **Step 6: Clean task-owned temporary storage**

After the PR is merged and the primary checkout is fast-forwarded, remove only these items: the task-scoped OSM tile staging directory, the UV and Matplotlib caches, the Hugging Face verification directory, and the managed worktree. Confirm that the primary checkout is clean and that no task-specific temporary path remains.
