# OSM-Style V3 Geographic Map Design

## Goal

Replace the V3 benchmark card map's country-outline visualization with a
physical world map that matches the project's coverage-map reference: pale
land, blue water, light graticule lines, and benchmark locations plotted as
source-colored points.

## Requirements

- Use land geometry derived from OpenStreetMap coastline data, not country
  boundaries.
- Keep the map global and in WGS84 longitude/latitude so the existing benchmark
  coordinates remain directly comparable.
- Plot all 300 English benchmark records and preserve the exact source colors:
  description orange, website blue, and wikipedia green.
- Keep a source-count legend and a concise title/subtitle.
- Make normal rendering completely offline and deterministic. The runtime
  generator must read committed inputs and must not download tiles or data.
- Keep the asset small enough for the repository and dataset card while
  retaining a clean world-scale coastline silhouette.
- Record the OSM-derived source URL, retrieval date, checksum, reduction
  method, attribution, and ODbL terms in the asset documentation and card.
- Test validation, source grouping, rendering determinism, CLI behavior, and
  the final PNG contract.

## Approaches considered

### OSM raster tile mosaic

This would look closest to the live OpenStreetMap website, but requires a
zoom-specific tile mosaic, has a much larger and less stable artifact, and
would make global labels and roads difficult to read at this scale.

### OSM-derived land polygons (selected)

OpenStreetMap coastline-derived land polygons provide the requested physical
map appearance without country-border emphasis. A deterministic preparation
step will reduce the upstream geometry to a committed GeoJSON asset; the
normal map generator will consume that asset offline. This is the smallest
faithful OSM-based solution and preserves reproducibility.

### Restyled Natural Earth country geometry

This would be compact and easy to render, but it would retain the wrong data
lineage and country-border semantics, so it is rejected.

## Architecture

The existing reporting module remains responsible for benchmark validation and
rendering. Its boundary loader will accept the reduced OSM land GeoJSON and
return only exterior land rings. The renderer will draw one filled land layer,
then three source-specific scatter layers and a legend. The CLI remains a thin
adapter with explicit benchmark, land-source, and output paths.

A separate preparation script will download the pinned OSM-derived source,
verify its upstream checksum, reduce and simplify its geometry deterministically,
and write the committed GeoJSON plus provenance metadata. This script is for
asset maintenance only; normal card-map generation will never access the
network.

## Output and documentation

The generated asset remains
`data/benchmark/v3/assets/v3-world-distribution.png`. The Natural Earth asset
and country-boundary language will be removed. The repository README,
translation documentation, and Hugging Face dataset card will describe the OSM
coastline-derived land layer, source colors, offline command, attribution, and
the fact that the map shows benchmark coverage rather than population or
source density.

