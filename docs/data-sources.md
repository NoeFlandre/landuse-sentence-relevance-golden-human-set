# Data sources and models

All source rows are read with Hugging Face streaming. The source revisions are pinned by commit SHA.

## Upstream datasets

- [OSM polygon Wikidata and Wikipedia](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia), revision `7c2a123ba2d4b27db415af6153deab9b98b75ec1`.
  The pipeline joins `polygons`, `polygon_document_links`, and `wikipedia_sections`. It accepts only `project=wikipedia`, `language=en`, and English sections. Wikivoyage rows are not used.
- [OSM polygon website tag](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag), revision `2c68154460f0bee314b887217ba54b23e3a2e181`.
  The pipeline uses both `website_text` and `contact_website_text`, with their matching URL fields. CommonLingua filters the resulting sentences to English.

The source rows are not cached locally. A model cache is allowed because sentence splitting and language detection require model weights; it contains no raw dataset rows.

## Models

- Sentence splitting: [SaT-12L-sm](https://huggingface.co/segment-any-text/sat-12l-sm), revision `d70c72a9331b2d5a9e82baad00c64964a23a09bb`.
- SaT tokenizer: [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base), revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- English detection: [CommonLingua](https://huggingface.co/PleIAs/CommonLingua), revision `43fe88d75e94b11283b66daccbbe4a73e7bc1361`. Website sentences need a confidence of at least `0.90` and are rejected when they exceed the model's supported byte limit.

The pinned identifiers and revisions live in `src/landuse_sentence_relevance/config.py`.
