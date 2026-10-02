# Land-use sentence relevance golden human set

This project is a small local UI. It builds a human relevance set of 100 sentences from two streamed Hugging Face datasets.

V2 uses English-only sentences from inside the source text blocks when possible. It uses H3 resolution 3 for geographic stratification. It selects 100 distinct cells that are spread across the world. It takes 50 records from each source and 50 Yes / 50 No annotations. The app streams the source rows. The Seagate project drive stores the reusable V2 candidate bank of 512 records (256 per source) and the labeled records.

The V2 runtime benchmark is `data/benchmark/v2-adjudicated.csv`. It has 154 rows. It is the human V2 set after the manual adjudication of its three-rater disagreements. The separate finalized V3 annotation benchmark is `data/benchmark/v3/final/v3-final.csv`. Its review trail is in the [Data catalogue](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/results/).

The committed tables are in `data/`. The reusable authentication, model, and tool state is in `state/`. The raw and intermediate annotation artifacts are in `results/`. Only `data/` is in Git.

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

To run the local, unpublished V3 annotation session, use its separate candidate, seed, and session paths:

```bash
PROJECT_DATA_ROOT="/Volumes/Seagate M3/projects/landuse-v3-annotation" \
  ANNOTATION_VERSION=v3 ./scripts/uv-seagate run landuse-annotate
```

Use `scripts/uv-seagate` for local UV commands. It does not run without the Seagate project drive. It keeps these items in the Seagate `state/` directory, outside the internal storage of the Mac: the virtual environment, the UV cache, the temporary files, the Python bytecode, and the Hugging Face login.

If no Hugging Face login is available, log in one time. Use the persistent project location:

```bash
./scripts/uv-seagate run hf auth login
```

The terminal shows these events: model loading, streamed-source checkpoints, candidate-pool readiness, saved-label progress, and upload and cleanup. It never logs sentence text or raw rows. The project keeps the login separately. Normal restarts and later sessions do not need a new login. The model and data caches are disposable.

The project prepares, hashes, and evaluates the prompt-tuning rounds for GPT and Claude independently. Refer to [LLM evaluation rounds](docs/llm-evaluation-rounds.md).

The app streams the public source rows. It keeps the candidate bank and the annotations on the Seagate project drive. The [project documentation](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/) gives the source revisions, the annotation contract, the licenses, and the QA gauntlet. For the project terms, refer to the [Glossary](docs/glossary.md).

## Versioning

A release is an immutable git tag. Nobody edits a tagged benchmark file. The current release is **v2.0.0**. Refer to [CHANGELOG.md](CHANGELOG.md) and the [versioning policy](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/versioning/).

## License

The project code and documentation are Apache-2.0. Upstream OSM and Wikipedia material keeps its original license. The dataset card gives the licenses of the components.
