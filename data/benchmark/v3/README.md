# V3 benchmark artifacts

This directory contains the complete V3 benchmark trail in lifecycle order:

- `reference/` — the original completed human benchmark.
- `independent-review/` — the blinded input, GPT response, generated disagreement queue, and human-reviewed resolutions.
- `final/` — the final benchmark produced from the reference plus the reviewed resolutions.
- `assets/` — the world-distribution map used by the multilingual release.
- `translations/` — one parallel CSV per project-provided `sat-3l-sm` language code.

Use [`final/v3-final.csv`](final/v3-final.csv) as the final V3 benchmark for this annotation round. The full method, counts, and integrity record are in the [V3 final benchmark documentation](../../../docs/v3-final-benchmark.md).

Do not overwrite the reference, GPT response, disagreement queue, or resolved-review file when making a later benchmark revision. Add a new versioned artifact and document its transformation.

The translation tree contains 85 files, including the exact English copy at
`translations/en/v3-final-en.csv`. Only `sentence` changes between language
files; labels and geographic/source metadata remain aligned with the final
English benchmark. The translation was produced by `gpt-5.6-luna-max`
(`gpt-5.6-luna`, `max` reasoning) through the Codex harness. The full
provenance, language inventory, validation contract, map, and Hugging Face
layout are documented in [V3 multilingual benchmark](../../../docs/v3-translations.md).
