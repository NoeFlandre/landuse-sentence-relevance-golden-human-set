# Data catalogue

This page lists every data file in the project, what it contains, and where it comes from.

The project has three data trees:

- **`data/`** is committed to Git. It holds the small, final tables that a person can review by hand. A reader must be able to read them without the Seagate drive.
- **`data/provenance/`** is the committed exception. It holds the small, immutable release snapshots that a reader needs to reproduce the published evaluations.
- **`results/`** is only on the Seagate project drive. Git ignores it. It holds the raw working artifacts: session logs, candidate pools, per-rater exports, and generated reports.

The project streams the source rows from Hugging Face. It never stores them locally.

## The benchmark

| Path | Rows | Contents |
| --- | --- | --- |
| `data/benchmark/v2-adjudicated.csv` | 154 | The V2 runtime and release benchmark. It is the human V2 export. Every three-rater disagreement is resolved by hand. The flagged rows are removed. |
| `data/benchmark/v3/final/v3-final.csv` | 300 | **The final V3 benchmark for the completed review.** It keeps the V3 reference rows. It applies the reviewed decisions from the independent-review stage. |

The V2 file has the same nine columns as the human export that the project used to build it. The columns are `sentence`, `label`, `polygon_name`, `h3_cell`, `latitude`, `longitude`, `source`, `region`, and `source_url`. The column order and the row order are the same. The adjudication removed four sentences that it flagged `remove`. It replaced 9 `label` values with the adjudicated verdict. The labels are 80 `yes` and 74 `no`.

The project regenerates the file. Nobody edits it by hand. `scripts/interrater_agreement.py` rebuilds it from the three rater exports and `data/interrater/adjudication.csv`. Refer to [Interrater agreement](interrater-agreement.md) for the method.

The V3 final file has the same schema of nine columns and the same reference order. It is a separate immutable benchmark. It does not replace the V2 release benchmark. It has 100 rows for each source. Its final labels are 160 `yes` and 140 `no`. The `yes`/`no` split is Wikipedia 54/46, Website 49/51, and Description 57/43.

These are the only committed copies of the final benchmarks. The generated analysis outputs do not duplicate them.

## V3 review and finalization

| Path | Rows | Contents |
| --- | --- | --- |
| `data/benchmark/v3/reference/v3-human-completed.csv` | 300 | The original completed human V3 reference, before the reassessment of the independent review decisions. |
| `data/benchmark/v3/independent-review/v3-independent-review-input.csv` | 300 | The blinded input that the project sent for the independent model review. It keeps the metadata. It does not have `label`. |
| `data/benchmark/v3/independent-review/v3-gpt-sol-5.6-extra-high.csv` | 300 | The canonical UTF-8 copy of the GPT response that the project received. It has the `label` of GPT. |
| `data/benchmark/v3/independent-review/v3-human-gpt-disagreements.csv` | 42 | The generated disagreement queue with the human decisions and the GPT decisions. Its `final_human_label` column was blank at first. |
| `data/benchmark/v3/independent-review/v3-resolved.csv` | 42 | The human-reviewed resolution of every disagreement. It has the completed `final_human_label` values. |
| `data/benchmark/v3/final/v3-final.csv` | 300 | The final benchmark. The project made it when it applied `v3-resolved.csv` to the reference in reference order. |

The multilingual release is in [V3 multilingual benchmark](v3-translations.md). It has 85 parallel CSV files in `data/benchmark/v3/translations/`: one English file and 84 translations. All files have the same 300 rows, the same final labels, and the same geographic and source metadata. The source-only map is at `data/benchmark/v3/assets/v3-world-distribution.png`.

[V3 final benchmark](v3-final-benchmark.md) describes the complete workflow of comparison, reassessment, and finalization. The independent model output is evidence for the review. It does not replace the human label automatically.

## Interrater review tables

These tables are committed. Each has one row for each sentence where the three raters did not agree.

| Path | Rows | Contents |
| --- | --- | --- |
| `data/interrater/disagreements.csv` | 31 | Generated. It has the non-unanimous sentences with the label of each rater, the outvoted rater, and the provenance. |
| `data/interrater/adjudication.csv` | 31 | The same rows and a hand-assigned `final_label` of `yes`, `no`, or `remove`. **It is the only hand-edited file in `data/`.** It is the input that produces the benchmark above. |

## Rater inputs

Round 1 has three labeled copies of the same 158 sentences. The release keeps a tracked provenance snapshot. Future rounds use the numbered structure on the drive only. Refer to [LLM evaluation rounds](llm-evaluation-rounds.md).

| Path | Rater | Label column |
| --- | --- | --- |
| `data/provenance/round-01/human.csv` | human annotator | `label` |
| `data/provenance/round-01/outputs/gpt.csv` | GPT 5.6 Extra High | `llm_label` |
| `data/provenance/round-01/outputs/claude.csv` | Claude Opus 5 Extra | `llm_label` |

[LLM labeling](llm-labeling.md) records the two machine runs and their shared prompt. [LLM evaluation rounds](llm-evaluation-rounds.md) describes the repeatable workflow.

## Working artifacts

These artifacts are on the drive only. They are historical or intermediate. None of them is a benchmark.

| Path | Contents |
| --- | --- |
| `data/provenance/round-01/` | The committed, immutable Round 1 release snapshot: prompt, input of 158 rows, rater files, manifest, canonical agreement report of 154 rows, and historical report of 158 rows. |
| `results/evaluations/round-01/` | The working mirror of the tracked Round 1 snapshot, on the drive only. |
| `results/annotations/benchmark/v1-human-annotated.csv` | The V1 human set of 100 rows. The V2 adjudicated benchmark replaces it. |
| `results/annotations/benchmark/v2-wikipedia.csv` | The 100 Wikipedia rows before the project combined them. |
| `results/annotations/benchmark/v2-website-balanced-58.csv` | The 58 website rows before the project combined them. They are balanced: 29 Yes and 29 No. |
| `results/evaluations/round-01/` | The complete historical Round 1 snapshot: prompt, input, benchmark, model outputs, manifest, and analysis. |
| `results/annotations/sessions/` | The resumable JSONL annotation sessions for the V1 and V2 runs. |
| `results/candidates/v1/pool.json` | The historical V1 candidate bank. |
| `results/candidates/v2/pool.json` | 512 reusable V2 candidates, 256 for each source. |
| `results/candidates/v2/website-only-pool.json` | The website-only V2 candidate bank. |
| `results/candidates/v2/progress.json` | The resumable checkpoint of the candidate build. |
| `results/candidates/v3/pool.json` | The finalized oversized V3 reservoir. It has at least 400 fresh, globally unique H3 resolution-3 cells for each source, and compact metadata. |
| `results/candidates/v3/progress.json` | The atomic, resumable checkpoint of the V3 candidate build. It has only compact candidate metadata. |
| `results/annotations/seeds/v3.json` | The atomic V3 annotation seed: 153 frozen V2 rows, 147 deterministic unlabeled slots, and quota and hash metadata. |
| `results/candidates/v2/archive/failures/` | The checkpoints from the failed builds. [Data sources](data-sources.md) describes the builds. |
| `results/maps/{v1,v2}/` | The static plots of the world and H3 distribution. |

## Runtime state

`state/` is on the drive only. It holds reusable local state. It never holds data:

- `state/huggingface-auth/` - the persistent Hugging Face login.
- `state/runtime-cache/` - the resumable model and data cache files.
- `state/uv-environment/`, `state/uv-python/` - the UV environment and the interpreter.
- `state/uv-cache/`, `state/python-cache/`, `state/xdg-cache/` - the tool caches.
- `state/tmp/` - the disposable scratch space of the processes.

Use `scripts/uv-seagate` for every local UV command. It creates and uses these paths only after it confirms that the Seagate project directory is mounted.
