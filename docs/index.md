# Land-use sentence relevance golden human set

This project builds a 100-sentence human golden set through a small local UI.

The app streams two public Hugging Face datasets, keeps English sentences only, and presents unique-cell candidates selected across the world with deterministic H3 maximin spacing. One human annotator assigns Yes or No relevance labels. The authoritative benchmark is `/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set/annotations.csv`, the human-annotated file used for benchmark evaluations.

## Run locally

```bash
uv sync --extra models
uv run landuse-annotate
```

Open <http://127.0.0.1:8000>. The source datasets are streamed. The candidate bank, annotation log, auth files, and disposable model/data cache live under `/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set` by default.

Authenticate once if needed; the login is stored separately from disposable model/data files:

```bash
HF_HOME="/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set/huggingface-auth" \
HF_TOKEN_PATH="/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set/huggingface-auth/token" \
uv run hf auth login
```

The terminal shows startup stages, sparse stream checkpoints, annotation counts, and final upload/cleanup without printing sentence text or raw rows. An existing login in the old application cache is migrated automatically on the next start.

After the public final upload succeeds, that exact cache is deleted automatically.

The final upload is automatic once all quotas are satisfied. See [annotation](annotation.md) for the contract and [QA](qa.md) for the deterministic checks.
