# Data sources and models

All source rows are read with Hugging Face streaming. The source revisions are pinned by commit SHA.

## Upstream datasets

- [OSM polygon Wikidata and Wikipedia](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia), revision `7c2a123ba2d4b27db415af6153deab9b98b75ec1`.
  The pipeline joins `polygons`, `polygon_document_links`, and `wikipedia_sections`. It accepts only `project=wikipedia`, `language=en`, and English sections. Wikivoyage rows are not used.
- [OSM polygon website tag](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag), revision `2c68154460f0bee314b887217ba54b23e3a2e181`.
  The pipeline uses both `website_text` and `contact_website_text`, with their matching URL fields. CommonLingua filters the resulting sentences to English.

Source rows are streamed. V2 selects up to 512 candidate cells per source and requires at least 320 candidates per source before saving 512 records, 256 per source, at `results/candidates/v2/pool.json`; labeled rows are saved at `results/annotations/sessions/v2-wikipedia.jsonl`. It lists 32 aligned, pinned Parquet files per source dataset, keeps website blocks to 400 characters, requires paragraph-like website text, and runs SaT in bounded four-text batches. CommonLingua checks language candidates in staged batches and stops after the first accepted sentence; `LANGUAGE_MODEL_DEVICE=auto` uses MPS on supported Macs and CPU otherwise, while `LANGUAGE_MODEL_DEVICE=cpu` is the reproducible CPU reference. For paragraph-mode selection, bounded discovery evaluates candidate eligibility before choosing the globally spread H3 cells, so non-English or title-only cells do not consume the geographic quota. A partial checkpoint resumes with only the remaining candidate-cell quota; expanding the remote file sample reuses the existing checkpoint because the source revision remains pinned. Only compact candidates and model caches are kept on the Seagate project drive. The benchmark to use is `data/benchmark/v2-adjudicated.csv`; see the [Data catalogue](results.md). V2 will publish to the `v2` split of the same public Hugging Face dataset after annotation. The disposable Hugging Face/model cache is kept for resume and upload retries, then deleted after a successful final upload.

## Models

- Sentence splitting: [SaT-12L-sm](https://huggingface.co/segment-any-text/sat-12l-sm), revision `d70c72a9331b2d5a9e82baad00c64964a23a09bb`.
- SaT tokenizer: [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base), revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- English detection: [CommonLingua](https://huggingface.co/PleIAs/CommonLingua), revision `43fe88d75e94b11283b66daccbbe4a73e7bc1361`. Website sentences need a confidence of at least `0.90` and are rejected when they exceed the model's supported byte limit.

The pinned identifiers and revisions live in `src/landuse_sentence_relevance/config.py`.

## V2 sentence selection

V2 sends only website blocks with at least two likely sentence boundaries to SaT, then prefers accepted sentences after the first one. This removes title-only rows while keeping selection deterministic.

## Failed V2 build: 2026-08-29

The first V2 build scanned 227,521 website rows but found only 213 valid English candidate cells, so it stopped before the required 256 website candidates and failed with `need enough disjoint source cells`. One malformed remote Parquet shard was also skipped because its Arrow batch size was zero. The root cause was an early-stop rule that treated the per-cell row cap as a completed candidate-cell quota. Its checkpoint is preserved as `results/candidates/v2/archive/failures/20260829.json`. The recovery path now counts valid candidates, prioritizes paragraph-like rows, reuses up to 16 compact discovery rows per cell, and checkpoints candidates at `results/candidates/v2/progress.json` on the Seagate. A restart can reuse completed source candidates without repeating their model work; raw streamed rows are still never persisted.

A second retry reached Wikipedia section splitting but produced no checkpoint after 19 minutes; the first batch was still inside SaT inference. Its zero-candidate checkpoint is preserved as `results/candidates/v2/archive/failures/20260829-section-bound.json`. The current path bounds Wikipedia sections to 800 characters before inference and uses bounded SaT batches. Remote catalog requests also force IPv4, retry once, and fail after 20 seconds instead of hanging.

The next retry reached 159 website candidates before it was intentionally stopped in a slow CommonLingua fallback batch. The compact checkpoint was preserved, and the run now resumes from it with the staged detector, a 400-character website bound, MPS acceleration when available, and the remaining-cell quota rather than rebuilding either source.

A subsequent restart failed during Hugging Face dataset metadata resolution with an IPv4 TLS handshake timeout. No source rows were stored and the 479-candidate checkpoint remained valid. The bounded Hugging Face client now retries one transport failure, keeps a finite timeout, and logs the failure instead of hanging indefinitely.

The 2026-08-29 resume then discovered 90 remaining website cells, but incorrectly rescanned 227,058 rows and yielded no candidates before failing the disjoint-cell constraint. Sparse cells had already been fully consumed during discovery, but the code tracked only their per-cell row cap. The checkpoint remained at 548 candidates. Discovery now records whether its bounded streams ended naturally, so a completed discovery pass cannot trigger that redundant rescan.
