# V3 geographic map assets

`osm-world-z2.png` is a 1024x1024, 4x4 mosaic of OpenStreetMap standard
tiles at zoom level 2. It provides the pale physical world basemap shown
behind the 300 benchmark coordinates. The source tile URL template is:

<https://tile.openstreetmap.org/{z}/{x}/{y}.png>

The exact tile checksums, retrieval date, mosaic checksum, attribution, and
license record are in `osm-world-z2-manifest.json`. The 16 exact source tile
PNGs are vendored under `osm-tiles/z2/<x>/<y>.png`. The current mosaic SHA-256
is `cca8296b91a42f92f7c85a20273c0454e2207e85530f2b5f776d1f5d91847be5`.

The maintenance command verifies the vendored snapshot and rebuilds the
mosaic without network access:

```bash
uv run python scripts/prepare_v3_osm_basemap.py \
  --manifest data/benchmark/v3/assets/osm-world-z2-manifest.json \
  --output data/benchmark/v3/assets/osm-world-z2.png \
  --tiles-dir data/benchmark/v3/assets/osm-tiles/z2
```

An intentional snapshot refresh uses `--download --record-checksums
--retrieved-at YYYY-MM-DD`; network access is never implicit. The normal map
command only reads committed inputs and performs no network access:

```bash
uv run python scripts/build_v3_world_map.py
```

Map data attribution: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
OpenStreetMap data is available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/);
the tile service is also subject to the
[OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
