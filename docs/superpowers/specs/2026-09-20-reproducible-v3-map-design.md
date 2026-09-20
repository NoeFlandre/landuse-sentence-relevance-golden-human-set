# Reproducible V3 Geographic Map Design

## Goal

Replace the opaque V3 world-distribution PNG with a deterministic, tested
generator that reads the final English benchmark and a versioned Natural Earth
110m boundary source, then regenerates the map used by the GitHub and Hugging
Face dataset cards.

## Requirements

- The generator must work from a clean checkout without a network request.
- The benchmark input must be validated before rendering: exact schema, 300
  records, finite WGS84 coordinates, and 100 records for each of the three
  sources (`description`, `website`, and `wikipedia`).
- The boundary source must be committed in a compact, geometry-only form with
  its Natural Earth provenance and public-domain attribution documented.
- Rendering must use a fixed non-interactive Matplotlib backend, pinned
  dependency version, fixed dimensions, colors, typography, axes, and metadata.
- Running the generator twice with the same inputs must produce byte-identical
  PNG files.
- Tests must cover valid loading, malformed rows, source quotas, GeoJSON
  geometry handling, and deterministic rendering.
- Documentation must identify the source CSV, boundary asset, command, and
  output path. The Hugging Face card must continue to display the regenerated
  asset and identify its reproducible provenance.

## Architecture

`src/landuse_sentence_relevance/reporting/geographic.py` will contain the
small, reusable implementation. It will expose typed point records, CSV
validation, GeoJSON boundary parsing, and rendering functions. Parsing and
validation will be independent from Matplotlib so malformed input and quota
errors can be tested without creating figures. Rendering will consume only the
validated points and parsed polygon rings.

`scripts/build_v3_world_map.py` will be a thin command-line adapter with
repository defaults for `data/benchmark/v3/final/v3-final.csv`,
`data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`, and
`data/benchmark/v3/assets/v3-world-distribution.png`. It will also accept
explicit paths for clean-checkout automation and test fixtures.

The Natural Earth boundary file will contain only the geometry needed for
country outlines. It will be stored beside the generated PNG, while a short
asset README will record the upstream dataset, release scale, retrieval date,
license, and reduction procedure. The generator will not download or discover
boundaries at runtime.

The existing visual contract will be retained: a WGS84 longitude/latitude
Plate Carree view, 1428x772 output, Natural Earth 110m outlines, points colored
by source, source-count legend, title/subtitle, and coverage disclaimer.

## Testing and release flow

1. Add failing unit tests for the loader, quota validation, boundary parser,
   and deterministic output.
2. Implement the smallest code that satisfies those tests.
3. Run focused tests, Ruff, type checks, and the repository quality gauntlet.
4. Regenerate the PNG from the committed inputs and verify its SHA-256 twice.
5. Update GitHub documentation and the Hugging Face card.
6. Upload only the regenerated map and card changes to the existing dataset,
   then verify the live file hashes and card image link.
7. Open a pull request linked to issue #37, attach it to the task, and wait
   for CI before reporting completion.
