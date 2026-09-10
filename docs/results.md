# Data catalogue

Every data file in the project, what it contains, and where it comes from.

There are two data trees:

- **`data/`** is committed to Git. It holds the small, final, hand-reviewable tables that must be readable without the Seagate drive.
- **`results/`** lives only on the Seagate project drive and is ignored by Git. It holds the raw working artifacts: session logs, candidate pools, per-rater exports, and generated reports.

Source rows from Hugging Face are streamed and are never stored locally.

## The benchmark

| Path | Rows | Contents |
| --- | --- | --- |
| `data/benchmark/v2-adjudicated.csv` | 154 | **The benchmark to use.** The human V2 export with every three-rater disagreement resolved by hand and the flagged rows removed |

It carries the same nine columns as the human export it was built from — `sentence`, `label`, `polygon_name`, `h3_cell`, `latitude`, `longitude`, `source`, `region`, `source_url` — in the same column and row order. Only two things differ from that export: 4 sentences flagged `remove` during adjudication are gone, and 9 `label` values were replaced by the adjudicated verdict. Labels are 80 `yes` and 74 `no`.

It is regenerated, never hand-edited: `scripts/interrater_agreement.py` rebuilds it from the three rater exports plus `data/interrater/adjudication.csv`. See [Interrater agreement](interrater-agreement.md) for the method.

## Interrater review tables

Committed, one row per sentence the three raters did not agree on.

| Path | Rows | Contents |
| --- | --- | --- |
| `data/interrater/disagreements.csv` | 31 | Generated. The non-unanimous sentences with each rater's label, the outvoted rater, and provenance |
| `data/interrater/adjudication.csv` | 31 | The same rows plus a hand-assigned `final_label` of `yes`, `no`, or `remove`. **The only hand-edited file in `data/`**, and the input that produces the benchmark above |

## Rater inputs

The three labelled copies of the same 158 sentences that the agreement analysis compares. Drive-only.

| Path | Rater | Label column |
| --- | --- | --- |
| `results/annotations/benchmark/v2-wikipedia-website-combined.csv` | human annotator | `label` |
| `results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv` | GPT 5.6 Extra High | `llm_label` |
| `results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv` | Claude Opus 5 Extra | `llm_label` |

The two machine runs and their shared prompt are recorded in [LLM labeling](llm-labeling.md).

## Working artifacts

Drive-only. Historical or intermediate; none of these is a benchmark.

| Path | Contents |
| --- | --- |
| `results/analysis/interrater/agreement.json` | Generated. The full agreement report: matched rows, label counts, pairwise metrics with confusion matrices, three-rater agreement, every disagreement, the adjudication summary, and each input's path, row count, and SHA-256 |
| `results/annotations/benchmark/v1-human-annotated.csv` | The 100-row V1 human set, superseded by the V2 adjudicated benchmark |
| `results/annotations/benchmark/v2-wikipedia.csv` | The 100 Wikipedia rows before they were combined |
| `results/annotations/benchmark/v2-website-balanced-58.csv` | The 58 website rows before they were combined, balanced 29 Yes / 29 No |
| `results/annotations/unlabeled/` | The same exports before labelling; the input given to the two models |
| `results/annotations/sessions/` | Resumable JSONL annotation sessions for the V1 and V2 runs |
| `results/candidates/v1/pool.json` | Historical V1 candidate bank |
| `results/candidates/v2/pool.json` | 512 reusable V2 candidates, 256 per source |
| `results/candidates/v2/website-only-pool.json` | Website-only V2 candidate bank |
| `results/candidates/v2/progress.json` | Resumable candidate-building checkpoint |
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
