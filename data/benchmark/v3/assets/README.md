# V3 geographic map assets

`ne-110m-land.geojson` is the Natural Earth 1:110m physical land vector
layer. It supplies the pale landmass outlines that the map draws behind the 300
benchmark coordinates. The source is:

<https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_land.geojson>

`ne-110m-land-manifest.json` has the pinned upstream ref, the retrieval date,
the checksum, the attribution, and the license record. The SHA-256 of the
current snapshot is
`9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9`.

To verify the vendored snapshot, run the maintenance command. It does not use
the network.

```bash
uv run python scripts/prepare_v3_land_basemap.py \
  --manifest data/benchmark/v3/assets/ne-110m-land-manifest.json \
  --output data/benchmark/v3/assets/ne-110m-land.geojson
```

To refresh the snapshot on purpose, use `--download --record-checksums
--retrieved-at YYYY-MM-DD`. The network is never used unless you ask for it.
The normal map command only reads committed inputs. It does not use the
network.

```bash
uv run python scripts/build_v3_world_map.py
```

Natural Earth is in the public domain. The requested credit is
"Made with Natural Earth" (<https://www.naturalearthdata.com/about/terms-of-use/>).
