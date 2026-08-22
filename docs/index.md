# Land-use sentence relevance golden human set

This project builds a 100-sentence human golden set through a small local UI.

The app streams two public Hugging Face datasets, keeps English sentences only, spreads candidates over shared H3 resolution-3 cells, and asks one human annotator for a Yes or No relevance label.

## Run locally

```bash
uv sync --extra models
uv run landuse-annotate
```

Open <http://127.0.0.1:8000>. The source datasets are never downloaded as local files. Only the labeled JSONL session is kept locally until the final public dataset is ready.

The final upload is automatic once all quotas are satisfied. See [annotation](annotation.md) for the contract and [QA](qa.md) for the deterministic checks.
