# Land-use sentence relevance golden human set

This project builds a golden set of 100 sentences. A human annotator labels the sentences in a small local UI.

The V2 app streams two public Hugging Face datasets. When possible, it keeps the English sentences from inside the source text blocks. It shows candidates from unique cells across the world. It selects the candidates with deterministic H3 maximin spacing. One human annotator assigns the relevance label Yes or No.

The V2 runtime benchmark is `data/benchmark/v2-adjudicated.csv`. It is the human V2 set after the adjudication of its three-rater disagreements. The finalized V3 annotation benchmark is `data/benchmark/v3/final/v3-final.csv`. A separate page describes its review and resolution trail. The [Data catalogue](results.md) lists every data file.

The project also releases the V3 benchmark as 85 language-aligned CSV files. Refer to the [V3 multilingual benchmark](v3-translations.md). It gives the translation method, the file layout, the integrity contract, and the geographic coverage map.

For the project terms, refer to the [Glossary](glossary.md).

## Run locally

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

Open <http://127.0.0.1:8000>. The app streams the source datasets. The V2 candidate bank and the annotation logs are in `results/`. The auth files, the virtual environment, the UV cache, the temporary files, and the model cache are in `state/`. By default, both trees are on `/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set`.

Use `scripts/uv-seagate` for local UV commands. It does not use the internal storage of the Mac. It stops when the Seagate drive is not mounted.

If necessary, log in one time. The login is stored separately from the disposable model and data files.

```bash
./scripts/uv-seagate run hf auth login
```

The terminal shows the startup stages, the sparse stream checkpoints, the annotation counts, and the final upload and cleanup. It does not print sentence text or raw rows. On the next start, the app moves an existing login from the old application cache automatically.

After the public final upload succeeds, the app deletes that exact cache automatically.

The final upload starts automatically when all quotas are complete. Refer to [annotation](annotation.md) for the contract. Refer to [QA](qa.md) for the deterministic checks.

The project also has a design for a follow-up set of 300 rows. The oversized candidate pool, the geographic preflight, and the deterministic annotation seed are available. The annotation is paused. The sources, quotas, storage, and runbook are in [V3 workflow](v3.md).
