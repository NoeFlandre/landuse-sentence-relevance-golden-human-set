# Results and storage layout

All generated artifacts stay on the Seagate project drive and are ignored by Git. Source rows from Hugging Face are streamed and are never stored locally.

## Results

| Purpose | Path | Contents |
| --- | --- | --- |
| V1 benchmark | `results/annotations/benchmark/v1-human-annotated.csv` | 100 human-annotated rows; the benchmark CSV |
| V2 Wikipedia benchmark | `results/annotations/benchmark/v2-wikipedia.csv` | 100 context-aware Wikipedia rows |
| V2 website benchmark | `results/annotations/benchmark/v2-website-balanced-58.csv` | 58 context-aware website rows, balanced 29 Yes / 29 No |
| V2 combined export | `results/annotations/benchmark/v2-wikipedia-website-combined.csv` | Combined V2 Wikipedia and website export |
| V2 LLM outputs | `results/annotations/llm/` | Two 158-row machine-labeled copies; not the human benchmark |
| Annotation sessions | `results/annotations/sessions/` | Resumable JSONL sessions, including historical V1 and V2 runs |
| V1 candidate pool | `results/candidates/v1/pool.json` | Historical V1 candidate bank |
| V2 candidate pool | `results/candidates/v2/pool.json` | 512 reusable candidates, 256 per source |
| V2 website pool | `results/candidates/v2/website-only-pool.json` | Website-only V2 candidate bank |
| V2 progress | `results/candidates/v2/progress.json` | Resumable candidate-building checkpoint |
| Failed checkpoints | `results/candidates/v2/archive/failures/` | Preserved historical failure evidence |
| Static maps | `results/maps/{v1,v2}/` | Non-interactive world/H3 distribution plots |

The benchmark path above is the human-annotated CSV to use for benchmark evaluation. The two machine-labeling runs and their exact shared prompt are documented in [LLM labeling](llm-labeling.md). Session logs and candidate pools are working artifacts, not alternate benchmark files. Before publishing a new benchmark, validate its source, label, uniqueness, and H3 quotas with the project quality gate.

## Runtime state

`state/` contains reusable local state and is also kept only on the Seagate drive:

- `state/huggingface-auth/` stores the persistent Hugging Face login.
- `state/runtime-cache/` stores resumable model/data cache files.
- `state/uv-environment/` and `state/uv-python/` store the UV environment and interpreter.
- `state/uv-cache/`, `state/python-cache/`, and `state/xdg-cache/` store tool caches.
- `state/tmp/` is disposable process scratch space.
- `state/legacy/` holds pre-migration scratch and quality-cache trees preserved for later deletion; new runs do not use them.

Use `scripts/uv-seagate` for every local UV command. It creates and uses these paths only after confirming that the Seagate project directory is mounted.
