# Land-use sentence relevance golden human set

A small local UI for building a 100-sentence human relevance set from two streamed Hugging Face datasets.

The workflow uses English-only sentences, H3 resolution 3 geographic stratification, 50 records from each source, and 50 Yes / 50 No annotations. Source rows are streamed; only labeled records and an application-owned runtime cache are kept locally until the final public dataset is uploaded. The cache is retained for resume and removed after a successful upload.

```bash
uv sync --extra models
uv run landuse-annotate
```

The app streams public source rows and keeps only labeled annotations in the ignored `state/` directory. See the [project documentation](https://noeflandre.github.io/landuse-sentence-relevance-golden-human-set/) for the source revisions, annotation contract, licenses, and QA gauntlet.

## License

Project code and documentation are Apache-2.0. Upstream OSM and Wikipedia material keeps its original license; the dataset card documents the component licenses.
