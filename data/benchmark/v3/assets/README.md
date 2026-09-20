# V3 geographic map assets

`natural-earth-110m-admin-0.geojson` is a geometry-only reduction of Natural
Earth's 1:110m Admin 0 Countries dataset. It was retrieved on 2026-09-20 from:

<https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip>

The upstream archive SHA-256 is
`0f243aeac8ac6cf26f0417285b0bd33ac47f1b5bdb719fd3e0df37d03ea37110`. The
reduction keeps the original Polygon and MultiPolygon geometries, removes all
properties, and writes a sorted compact GeoJSON FeatureCollection. This keeps
map generation offline and avoids depending on a local GIS cache.

Natural Earth data is public domain. See <https://www.naturalearthdata.com/about/terms-of-use/>
for the upstream terms and attribution guidance.

The asset is consumed by `scripts/build_v3_world_map.py` and is not downloaded
or modified during normal generation.
