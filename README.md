# Land-use sentence relevance golden human set

A small local UI for building a 100-sentence human relevance set from two streamed Hugging Face datasets.

V2 uses English-only sentences from inside source text blocks when available, H3 resolution 3 geographic stratification, 100 distinct globally spread cells, 50 records from each source, and 50 Yes / 50 No annotations. Source rows are streamed; the reusable 512-record V2 candidate bank (256 per source) and labeled records are stored on the Seagate project drive. The benchmark to use is `data/benchmark/v2-adjudicated.csv`: 154 rows, the human V2 set with its three-rater disagreements adjudicated by hand. Every data file is listed in the [Data catalogue](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/results/). Committed tables live under `data/`; reusable authentication, model, and tool state is kept under `state/`, and raw or intermediate annotation artifacts under `results/`. Only `data/` is in Git.

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

To run the local, unpublished V3 annotation session explicitly, use its
separate candidate, seed, and session paths:

```bash
PROJECT_DATA_ROOT="/Volumes/Seagate M3/projects/landuse-v3-annotation" \
  ANNOTATION_VERSION=v3 ./scripts/uv-seagate run landuse-annotate
```

Use `scripts/uv-seagate` for local UV commands. It refuses to run without the Seagate project drive and keeps the virtual environment, UV cache, temporary files, Python bytecode, and Hugging Face login under the Seagate `state/` directory, outside the Mac's internal storage.

If no Hugging Face login is available, authenticate once with the persistent project location:

```bash
./scripts/uv-seagate run hf auth login
```

The terminal reports model loading, streamed-source checkpoints, candidate-pool readiness, saved-label progress, and upload/cleanup. It never logs sentence text or raw rows. The login is kept separately so normal restarts and later sessions do not require another login; model and data caches remain disposable.

Prompt-tuning rounds for GPT and Claude are prepared, hashed, and evaluated independently; see [LLM evaluation rounds](docs/llm-evaluation-rounds.md).

The app streams public source rows and keeps the candidate bank and annotations on the Seagate project drive. See the [project documentation](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/) for the source revisions, annotation contract, licenses, and QA gauntlet.

## Versioning

Releases are immutable git tags; a tagged benchmark file is never edited. The current release is **v2.0.0**. See [CHANGELOG.md](CHANGELOG.md) and the [versioning policy](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/versioning/).

## License

Project code and documentation are Apache-2.0. Upstream OSM and Wikipedia material keeps its original license; the dataset card documents the component licenses.
