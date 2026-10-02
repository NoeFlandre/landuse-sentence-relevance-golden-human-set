# V3 benchmark artifacts

This directory contains the complete V3 benchmark trail in lifecycle order:

- `reference/`: the original completed human benchmark.
- `independent-review/`: the blinded input, the GPT response, the generated disagreement queue, and the resolutions that a human reviewed.
- `final/`: the final benchmark. It comes from the reference and the reviewed resolutions.
- `assets/`: the pinned OSM physical basemap, the provenance manifest, and the generated world-distribution map. The multilingual release uses them.
- `translations/`: one parallel CSV for each `sat-3l-sm` language code that the project provides.

Use [`final/v3-final.csv`](final/v3-final.csv) as the final V3 benchmark for this annotation round. The [V3 final benchmark documentation](../../../docs/v3-final-benchmark.md) has the full method, the counts, and the integrity record.

Do not overwrite the reference, the GPT response, the disagreement queue, or the resolved-review file when you make a later benchmark revision. Add a new versioned artifact. Document how it changes the data.

The translation tree contains 85 files. One of them is the exact English copy at
`translations/en/v3-final-en.csv`. Only `sentence` changes between language
files. The labels and the geographic and source metadata stay aligned with the
final English benchmark. The translation was produced by `gpt-5.6-luna-max`
(`gpt-5.6-luna`, `max` reasoning) through the Codex harness. The
[V3 multilingual benchmark](../../../docs/v3-translations.md) page documents the
full provenance, the language list, the validation contract, the map, and the
Hugging Face layout.

To reproduce the map from this checkout, run:

```bash
uv run python scripts/build_v3_world_map.py
```

The command validates `final/v3-final.csv`. It reads the committed Natural Earth
110m land vectors at `assets/ne-110m-land.geojson`. It writes
`assets/v3-world-distribution.png` with the pinned Matplotlib 3.11.1
renderer. It does not use the network. The `assets/` directory keeps the exact
snapshot bytes and their checksum. The checksum is also in
`assets/ne-110m-land-manifest.json`.
