# V3 benchmark artifacts

This directory contains the complete V3 benchmark trail in lifecycle order:

- `reference/` — the original completed human benchmark.
- `independent-review/` — the blinded input, GPT response, generated disagreement queue, and human-reviewed resolutions.
- `final/` — the final benchmark produced from the reference plus the reviewed resolutions.

Use [`final/v3-final.csv`](final/v3-final.csv) as the final V3 benchmark for this annotation round. The full method, counts, and integrity record are in the [V3 final benchmark documentation](../../../docs/v3-final-benchmark.md).

Do not overwrite the reference, GPT response, disagreement queue, or resolved-review file when making a later benchmark revision. Add a new versioned artifact and document its transformation.
