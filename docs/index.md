# Land-use sentence relevance golden human set

This project builds a 100-sentence human golden set through a small local UI.

The V2 app streams two public Hugging Face datasets, keeps English sentences from inside source text blocks when available, and presents unique-cell candidates selected across the world with deterministic H3 maximin spacing. One human annotator assigns Yes or No relevance labels. The benchmark to use is `data/benchmark/v2-adjudicated.csv`, the human V2 set with its three-rater disagreements adjudicated. Every data file is listed in the [Data catalogue](results.md).

## Run locally

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

Open <http://127.0.0.1:8000>. The source datasets are streamed. The V2 candidate bank and annotation logs live under `results/`; auth files, the virtual environment, UV cache, temporary files, and model cache live under `state/`. Both trees are on `/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set` by default.

Use `scripts/uv-seagate` for local UV commands. It refuses to fall back to the Mac's internal storage when the Seagate drive is not mounted.

Authenticate once if needed; the login is stored separately from disposable model/data files:

```bash
./scripts/uv-seagate run hf auth login
```

The terminal shows startup stages, sparse stream checkpoints, annotation counts, and final upload/cleanup without printing sentence text or raw rows. An existing login in the old application cache is migrated automatically on the next start.

After the public final upload succeeds, that exact cache is deleted automatically.

The final upload is automatic once all quotas are satisfied. See [annotation](annotation.md) for the contract and [QA](qa.md) for the deterministic checks.

The follow-up 300-row set is paused; its sources, quotas, and storage are described in [V3 workflow](v3.md).
