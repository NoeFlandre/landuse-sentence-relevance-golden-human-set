# V3 geographic map assets

`osm-world-z2.png` is a 1024x1024, 4x4 mosaic of OpenStreetMap standard
tiles at zoom level 2. It provides the pale physical world basemap shown
behind the 300 benchmark coordinates. The source tile URL template is:

<https://tile.openstreetmap.org/{z}/{x}/{y}.png>

The exact tile checksums, retrieval date, mosaic checksum, attribution, and
license record are in `osm-world-z2-manifest.json`. The current mosaic SHA-256
is `36d8cdc3d8442e6a71660eac7d81c3d1d89af347808c7a2c923cac9c382fdf7a`.

The maintenance command
`scripts/prepare_v3_osm_basemap.py` can re-fetch and verify the pinned tiles.
The normal map command only reads the committed PNG and performs no network
access:

```bash
uv run python scripts/build_v3_world_map.py
```

Map data attribution: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
OpenStreetMap data is available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/);
the tile service is also subject to the
[OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
