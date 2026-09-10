# Land-use sentence relevance golden human set

A small local UI for building a 100-sentence human relevance set from two streamed Hugging Face datasets.

V2 uses English-only sentences from inside source text blocks when available, H3 resolution 3 geographic stratification, 100 distinct globally spread cells, 50 records from each source, and 50 Yes / 50 No annotations. Source rows are streamed; the reusable 512-record V2 candidate bank (256 per source) and labeled records are stored on the Seagate project drive. The authoritative human-annotated benchmark is `results/annotations/benchmark/v1-human-annotated.csv`. The organized V2 artifacts and their status are documented in [Results](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/results/). Reusable authentication, model, and tool state is kept under `state/`; generated annotation artifacts are kept under `results/`.

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

Use `scripts/uv-seagate` for local UV commands. It refuses to run without the Seagate project drive and keeps the virtual environment, UV cache, temporary files, Python bytecode, and Hugging Face login under the Seagate `state/` directory, outside the Mac's internal storage.

If no Hugging Face login is available, authenticate once with the persistent project location:

```bash
./scripts/uv-seagate run hf auth login
```

The terminal reports model loading, streamed-source checkpoints, candidate-pool readiness, saved-label progress, and upload/cleanup. It never logs sentence text or raw rows. The login is kept separately so normal restarts and later sessions do not require another login; model and data caches remain disposable.

The app streams public source rows and keeps the candidate bank and annotations on the Seagate project drive. See the [project documentation](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/) for the source revisions, annotation contract, licenses, and QA gauntlet.

## License

Project code and documentation are Apache-2.0. Upstream OSM and Wikipedia material keeps its original license; the dataset card documents the component licenses.
