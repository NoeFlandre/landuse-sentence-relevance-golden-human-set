# OSM-Style V3 Geographic Map Design

## Goal

Replace the map of the V3 benchmark card. The old map shows country outlines. The new map is a physical world map. It matches the reference coverage map: pale land, blue water, light graticule lines, and benchmark locations as points. The color of each point shows the source.

## Requirements

- Use a low-zoom OpenStreetMap physical basemap. Do not use country-boundary polygons.
- Keep the map global and in WGS84 longitude/latitude. The existing benchmark coordinates then stay directly comparable.
- Plot all 300 English benchmark records. Keep the exact source colors: description orange, website blue, and wikipedia green.
- Keep a source-count legend and a concise title and subtitle.
- Make the normal rendering completely offline and deterministic. The runtime generator must read committed inputs. It must not download tiles or data.
- Keep the asset small enough for the repository and the dataset card. Keep a clean coastline silhouette at world scale. The selected basemap is the 4x4 z2 tile mosaic. It is a small set of 256px OSM tiles. It is not the full OSM land-polygon archive of approximately 926 MB.
- Record these items in the asset documentation and in the card: the OSM tile URL template, the zoom, the retrieval date, the checksum of each tile, the checksum of the mosaic, the attribution, and the ODbL terms.
- Test the validation, the source grouping, the rendering determinism, the CLI behavior, and the contract of the final PNG.

## Approaches considered

### OSM raster tile mosaic

This approach looks closest to the live OpenStreetMap website. It needs a mosaic of tiles for a specific zoom. Its artifact is much larger and less stable. At this scale, the global labels and roads are difficult to read.

### OSM z2 raster mosaic (selected)

The standard physical tiles of OpenStreetMap at zoom 2 give the requested appearance. They do not emphasize country borders. A deterministic preparation step gets the fixed 4x4 tile set one time. It stores the exact 16 PNG bytes in the repository. It verifies their pinned checksums. It stitches them into one committed PNG. The normal map generator and the normal snapshot verification use only these local bytes. This approach is much smaller than the full coastline archive. It matches the reference image more closely.

### Restyled Natural Earth country geometry

This approach is compact and easy to render. It keeps the wrong data lineage and the semantics of country borders. The project rejects it.

## Architecture

The existing reporting module stays responsible for the benchmark validation and the rendering. Its basemap loader will validate the dimensions and channels of the committed OSM PNG. The renderer will draw that image in Web Mercator. Then it will draw three source-specific scatter layers and a legend. The CLI stays a thin adapter with explicit paths for the benchmark, the basemap, and the output.

A separate preparation script will verify the vendored z2 OSM tiles and stitch them deterministically. It will write the committed PNG and the manifest. The explicit `--download --record-checksums` mode refreshes the local snapshot. The network is never implicit. This script is for asset maintenance only. The normal generation of the card map will never use the network.

## Output and documentation

The generated asset stays `data/benchmark/v3/assets/v3-world-distribution.png`. The project will remove the Natural Earth asset and the country-boundary language. The repository README, the translation documentation, and the Hugging Face dataset card will describe these items: the OSM z2 basemap, the source colors, the offline command, the attribution, and the fact that the map shows benchmark coverage. It does not show population or source density.
