# Land-use sentence relevance golden human set

A small local UI for building a 100-sentence human relevance set from two streamed Hugging Face datasets.

The workflow uses English-only sentences, H3 resolution 3 geographic stratification, 100 distinct globally spread cells, 50 records from each source, and 50 Yes / 50 No annotations. Source rows are streamed; the reusable 256-record candidate bank and labeled records are stored on the Seagate project drive. The disposable model/data cache is retained for resume and removed after a successful upload.

```bash
uv sync --extra models
uv run landuse-annotate
```

If no Hugging Face login is available, authenticate once with the persistent project location:

```bash
HF_HOME="/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set/huggingface-auth" \
HF_TOKEN_PATH="/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set/huggingface-auth/token" \
uv run hf auth login
```

The terminal reports model loading, streamed-source checkpoints, candidate-pool readiness, saved-label progress, and upload/cleanup. It never logs sentence text or raw rows. The login is kept separately so normal restarts and later sessions do not require another login; model and data caches remain disposable.

The app streams public source rows and keeps the candidate bank and annotations on the Seagate project drive. See the [project documentation](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/) for the source revisions, annotation contract, licenses, and QA gauntlet.

## License

Project code and documentation are Apache-2.0. Upstream OSM and Wikipedia material keeps its original license; the dataset card documents the component licenses.
