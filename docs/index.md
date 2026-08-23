# Land-use sentence relevance golden human set

This project builds a 100-sentence human golden set through a small local UI.

The app streams two public Hugging Face datasets, keeps English sentences only, spreads candidates over shared H3 resolution-3 cells, and asks one human annotator for a Yes or No relevance label.

## Run locally

```bash
uv sync --extra models
uv run landuse-annotate
```

Open <http://127.0.0.1:8000>. The source datasets are streamed. Hugging Face metadata, streamed artifacts, and model weights use an application-owned cache that remains available for resume and upload retries.

After the public final upload succeeds, that exact cache is deleted automatically.

The final upload is automatic once all quotas are satisfied. See [annotation](annotation.md) for the contract and [QA](qa.md) for the deterministic checks.
