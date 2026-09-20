# OSM-Style V3 Geographic Map Design

## Goal

Replace the V3 benchmark card map's country-outline visualization with a
physical world map that matches the project's coverage-map reference: pale
land, blue water, light graticule lines, and benchmark locations plotted as
source-colored points.

## Requirements

- Use a low-zoom OpenStreetMap physical basemap, not country-boundary
  polygons.
- Keep the map global and in WGS84 longitude/latitude so the existing benchmark
  coordinates remain directly comparable.
- Plot all 300 English benchmark records and preserve the exact source colors:
  description orange, website blue, and wikipedia green.
- Keep a source-count legend and a concise title/subtitle.
- Make normal rendering completely offline and deterministic. The runtime
  generator must read committed inputs and must not download tiles or data.
- Keep the asset small enough for the repository and dataset card while
  retaining a clean world-scale coastline silhouette. The selected basemap is
  the 4x4 z2 tile mosaic, which is only a small set of 256px OSM tiles rather
  than the approximately 926 MB full OSM land-polygon archive.
- Record the OSM tile URL template, zoom, retrieval date, per-tile checksums,
  mosaic checksum, attribution, and ODbL terms in the asset documentation and
  card.
- Test validation, source grouping, rendering determinism, CLI behavior, and
  the final PNG contract.

## Approaches considered

### OSM raster tile mosaic

This would look closest to the live OpenStreetMap website, but requires a
zoom-specific tile mosaic, has a much larger and less stable artifact, and
would make global labels and roads difficult to read at this scale.

### OSM z2 raster mosaic (selected)

OpenStreetMap's standard physical tiles at zoom 2 provide the requested
appearance without country-border emphasis. A deterministic preparation step
will fetch the fixed 4x4 tile set, verify pinned tile checksums, and stitch it
into one committed PNG. The normal map generator will consume that PNG
offline. This is substantially smaller than the full coastline archive and
matches the reference image more closely.

### Restyled Natural Earth country geometry

This would be compact and easy to render, but it would retain the wrong data
lineage and country-border semantics, so it is rejected.

## Architecture

The existing reporting module remains responsible for benchmark validation and
rendering. Its basemap loader will validate the committed OSM PNG dimensions
and channels. The renderer will draw that image in Web Mercator, then three
source-specific scatter layers and a legend. The CLI remains a thin adapter
with explicit benchmark, basemap, and output paths.

A separate preparation script will download the pinned z2 OSM tiles, verify
their checksums, stitch them deterministically, and write the committed PNG
plus manifest. This script is for asset maintenance only; normal card-map
generation will never access the network.

## Output and documentation

The generated asset remains
`data/benchmark/v3/assets/v3-world-distribution.png`. The Natural Earth asset
and country-boundary language will be removed. The repository README,
translation documentation, and Hugging Face dataset card will describe the OSM
z2 basemap, source colors, offline command, attribution, and the fact that the
map shows benchmark coverage rather than population or source density.
