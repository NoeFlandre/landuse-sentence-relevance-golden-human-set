# Data catalogue

Every data file in the project, what it contains, and where it comes from.

There are three data trees:

- **`data/`** is committed to Git. It holds the small, final, hand-reviewable tables that must be readable without the Seagate drive.
- **`data/provenance/`** is the committed exception for small, immutable release snapshots needed to reproduce published evaluations.
- **`results/`** lives only on the Seagate project drive and is ignored by Git. It holds the raw working artifacts: session logs, candidate pools, per-rater exports, and generated reports.

Source rows from Hugging Face are streamed and are never stored locally.

## The benchmark

| Path | Rows | Contents |
| --- | --- | --- |
| `data/benchmark/v2-adjudicated.csv` | 154 | **The benchmark to use.** The human V2 export with every three-rater disagreement resolved by hand and the flagged rows removed |

It carries the same nine columns as the human export it was built from — `sentence`, `label`, `polygon_name`, `h3_cell`, `latitude`, `longitude`, `source`, `region`, `source_url` — in the same column and row order. Only two things differ from that export: 4 sentences flagged `remove` during adjudication are gone, and 9 `label` values were replaced by the adjudicated verdict. Labels are 80 `yes` and 74 `no`.

It is regenerated, never hand-edited: `scripts/interrater_agreement.py` rebuilds it from the three rater exports plus `data/interrater/adjudication.csv`. See [Interrater agreement](interrater-agreement.md) for the method.

This is the only final-benchmark copy; generated analysis outputs do not duplicate it.

## Interrater review tables

Committed, one row per sentence the three raters did not agree on.

| Path | Rows | Contents |
| --- | --- | --- |
| `data/interrater/disagreements.csv` | 31 | Generated. The non-unanimous sentences with each rater's label, the outvoted rater, and provenance |
| `data/interrater/adjudication.csv` | 31 | The same rows plus a hand-assigned `final_label` of `yes`, `no`, or `remove`. **The only hand-edited file in `data/`**, and the input that produces the benchmark above |

## Rater inputs

Round 1's three labelled copies of the same 158 sentences. The release keeps a
tracked provenance snapshot; future rounds use the numbered drive-only
structure in [LLM evaluation rounds](llm-evaluation-rounds.md).

| Path | Rater | Label column |
| --- | --- | --- |
| `data/provenance/round-01/human.csv` | human annotator | `label` |
| `data/provenance/round-01/outputs/gpt.csv` | GPT 5.6 Extra High | `llm_label` |
| `data/provenance/round-01/outputs/claude.csv` | Claude Opus 5 Extra | `llm_label` |

The two machine runs and their shared prompt are recorded in [LLM labeling](llm-labeling.md); the repeatable workflow is in [LLM evaluation rounds](llm-evaluation-rounds.md).

## Working artifacts

Drive-only. Historical or intermediate; none of these is a benchmark.

| Path | Contents |
| --- | --- |
| `data/provenance/round-01/` | Committed, immutable Round 1 release snapshot: prompt, 158-row input and rater files, manifest, canonical 154-row agreement report, and historical 158-row report |
| `results/evaluations/round-01/` | Drive-only working mirror of the tracked Round 1 snapshot |
| `results/annotations/benchmark/v1-human-annotated.csv` | The 100-row V1 human set, superseded by the V2 adjudicated benchmark |
| `results/annotations/benchmark/v2-wikipedia.csv` | The 100 Wikipedia rows before they were combined |
| `results/annotations/benchmark/v2-website-balanced-58.csv` | The 58 website rows before they were combined, balanced 29 Yes / 29 No |
| `results/evaluations/round-01/` | Complete historical Round 1 snapshot: prompt, input, benchmark, model outputs, manifest, and analysis |
| `results/annotations/sessions/` | Resumable JSONL annotation sessions for the V1 and V2 runs |
| `results/candidates/v1/pool.json` | Historical V1 candidate bank |
| `results/candidates/v2/pool.json` | 512 reusable V2 candidates, 256 per source |
| `results/candidates/v2/website-only-pool.json` | Website-only V2 candidate bank |
| `results/candidates/v2/progress.json` | Resumable candidate-building checkpoint |
| `results/candidates/v3/pool.json` | Finalized V3 oversized reservoir: at least 400 fresh globally unique H3 resolution-3 cells per source, plus compact metadata |
| `results/candidates/v3/progress.json` | Atomic resumable V3 candidate-building checkpoint, containing compact candidate metadata only |
| `results/annotations/seeds/v3.json` | Atomic V3 annotation seed: 153 frozen V2 rows, 147 deterministic unlabeled slots, quota and hash metadata |
| `results/candidates/v2/archive/failures/` | Preserved checkpoints from the failed builds described in [Data sources](data-sources.md) |
| `results/maps/{v1,v2}/` | Static world/H3 distribution plots |

## Runtime state

`state/` is drive-only and holds reusable local state, never data:

- `state/huggingface-auth/` — the persistent Hugging Face login.
- `state/runtime-cache/` — resumable model and data cache files.
- `state/uv-environment/`, `state/uv-python/` — the UV environment and interpreter.
- `state/uv-cache/`, `state/python-cache/`, `state/xdg-cache/` — tool caches.
- `state/tmp/` — disposable process scratch space.

Use `scripts/uv-seagate` for every local UV command. It creates and uses these paths only after confirming the Seagate project directory is mounted.
