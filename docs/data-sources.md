# Data sources and models

The project reads all source rows with Hugging Face streaming. The commit SHA pins each source revision.

## Upstream datasets

- [OSM polygon Wikidata and Wikipedia](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia), revision `7c2a123ba2d4b27db415af6153deab9b98b75ec1`.
  The pipeline joins `polygons`, `polygon_document_links`, and `wikipedia_sections`. It accepts only `project=wikipedia`, `language=en`, and English sections. It does not use Wikivoyage rows.
- [OSM polygon website tag](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag), revision `2c68154460f0bee314b887217ba54b23e3a2e181`.
  The pipeline uses `website_text` and `contact_website_text`, with their matching URL fields. CommonLingua filters the resulting sentences to English.

## Streaming and candidate selection

The pipeline streams the source rows. V2 selects up to 512 candidate cells for each source. It needs at least 320 candidates for each source before it saves 512 records, 256 for each source, at `results/candidates/v2/pool.json`. It saves the labeled rows at `results/annotations/sessions/v2-wikipedia.jsonl`.

The pipeline lists 32 aligned, pinned Parquet files for each source dataset. It keeps the website blocks to 400 characters. It needs paragraph-like website text. It runs SaT in bounded batches of four texts.

CommonLingua checks the language candidates in staged batches. It stops after the first accepted sentence. `LANGUAGE_MODEL_DEVICE=auto` uses MPS on supported Macs and CPU on other machines. `LANGUAGE_MODEL_DEVICE=cpu` is the reproducible CPU reference.

In paragraph mode, the bounded discovery evaluates the eligibility of the candidates before it chooses the globally spread H3 cells. Non-English cells and title-only cells therefore do not use the geographic quota.

A partial checkpoint resumes with only the remaining candidate-cell quota. When the pipeline expands the remote file sample, it reuses the existing checkpoint. The source revision stays pinned.

The Seagate project drive keeps only the compact candidates and the model caches. The V2 runtime benchmark is `data/benchmark/v2-adjudicated.csv`. The finalized V3 annotation benchmark is `data/benchmark/v3/final/v3-final.csv`. Refer to the [Data catalogue](results.md) for their separate roles.

After the annotation, V2 will publish to the `v2` split of the same public Hugging Face dataset. The app keeps the disposable Hugging Face and model cache for resume and upload retries. It deletes the cache after a successful final upload.

## Models

- Sentence splitting: [SaT-12L-sm](https://huggingface.co/segment-any-text/sat-12l-sm), revision `d70c72a9331b2d5a9e82baad00c64964a23a09bb`.
- SaT tokenizer: [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base), revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- English detection: [CommonLingua](https://huggingface.co/PleIAs/CommonLingua), revision `43fe88d75e94b11283b66daccbbe4a73e7bc1361`. A website sentence needs a confidence of at least `0.90`. The model rejects a sentence that is longer than its supported byte limit.

The pinned identifiers and revisions are in `src/landuse_sentence_relevance/config.py`.

## V2 sentence selection

V2 sends to SaT only the website blocks that have at least two likely sentence boundaries. Then it prefers the accepted sentences after the first sentence. This removes the title-only rows. The selection stays deterministic.

## Failed V2 build: 2026-08-29

The first V2 build scanned 227,521 website rows. It found only 213 valid English candidate cells. It stopped before the required 256 website candidates and failed with `need enough disjoint source cells`. The build also skipped one malformed remote Parquet shard, because its Arrow batch size was zero.

The root cause was an early-stop rule. The rule treated the per-cell row cap as a completed candidate-cell quota. The project keeps the checkpoint as `results/candidates/v2/archive/failures/20260829.json`.

The recovery path now does these things:

- It counts the valid candidates.
- It gives priority to paragraph-like rows.
- It reuses up to 16 compact discovery rows for each cell.
- It saves the candidate checkpoints at `results/candidates/v2/progress.json` on the Seagate drive.

A restart can reuse the completed source candidates. It does not repeat their model work. The pipeline still never persists raw streamed rows.

A second retry reached the Wikipedia section splitting. It produced no checkpoint after 19 minutes. The first batch was still in SaT inference. The project keeps its zero-candidate checkpoint as `results/candidates/v2/archive/failures/20260829-section-bound.json`. The current path limits the Wikipedia sections to 800 characters before inference. It uses bounded SaT batches. The remote catalog requests also force IPv4 and retry one time. They fail after 20 seconds. They do not hang.

The next retry reached 159 website candidates. The operator stopped it on purpose in a slow CommonLingua fallback batch. The project kept the compact checkpoint. The run now resumes from the checkpoint. It uses the staged detector, a website bound of 400 characters, MPS acceleration when available, and the remaining-cell quota. It does not rebuild either source.

A later restart failed during the Hugging Face dataset metadata resolution. An IPv4 TLS handshake timeout caused the failure. The run stored no source rows. The checkpoint of 479 candidates stayed valid. The bounded Hugging Face client now retries one transport failure. It keeps a finite timeout. It logs the failure. It does not hang.

The resume of 2026-08-29 then discovered 90 remaining website cells. It wrongly scanned 227,058 rows again and yielded no candidates. Then it failed the disjoint-cell constraint. The discovery had already fully consumed the sparse cells, but the code tracked only their per-cell row cap. The checkpoint stayed at 548 candidates. Now the discovery records if its bounded streams ended naturally. A completed discovery pass cannot start the redundant rescan.
