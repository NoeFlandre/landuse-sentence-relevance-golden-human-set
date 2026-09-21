# V3 geographic map assets

`ne-110m-land.geojson` is the Natural Earth 1:110m physical land vector
layer. It provides the pale landmass outlines drawn behind the 300
benchmark coordinates. The source is:

<https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_land.geojson>

The pinned upstream ref, retrieval date, checksum, attribution, and license
record are in `ne-110m-land-manifest.json`. The current snapshot SHA-256 is
`9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9`.

The maintenance command verifies the vendored snapshot without network
access:

```bash
uv run python scripts/prepare_v3_land_basemap.py \
  --manifest data/benchmark/v3/assets/ne-110m-land-manifest.json \
  --output data/benchmark/v3/assets/ne-110m-land.geojson
```

An intentional snapshot refresh uses `--download --record-checksums
--retrieved-at YYYY-MM-DD`; network access is never implicit. The normal map
command only reads committed inputs and performs no network access:

```bash
uv run python scripts/build_v3_world_map.py
```

Natural Earth is in the public domain; the requested credit is
"Made with Natural Earth" (<https://www.naturalearthdata.com/about/terms-of-use/>).
