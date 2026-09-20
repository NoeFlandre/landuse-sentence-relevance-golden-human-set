# V3 multilingual benchmark

The V3 multilingual release is a translation of the immutable English
benchmark at `data/benchmark/v3/final/v3-final.csv`. It contains one CSV per
language under `data/benchmark/v3/translations/<iso>/` and the corresponding
files are published under `data/translations/<iso>/` in the Hugging Face
dataset.

## What is translated

Only the `sentence` field is translated. The following fields are copied from
the English final benchmark without semantic changes and remain in the same
row order:

```text
label, polygon_name, h3_cell, latitude, longitude, source, region, source_url
```

`label` is the final human-reviewed English benchmark decision. It is not
recomputed from a translated sentence. This makes every language file a
parallel text view of the same 300 benchmark records rather than a new
annotation round. Proper names, URLs, coordinates, source metadata, and the
three source quotas are retained as supplied.

## English benchmark provenance

The English file was assembled in the following controlled sequence:

1. The completed 300-row human reference was retained as the starting point.
2. A label-free copy was sent for an independent GPT assessment using the
   validated land-use/land-cover remote-sensing prompt.
3. The 42 human/GPT disagreements were exported into a minimal review queue.
4. A human reassessed every disagreement; the original reference and GPT
   response were not overwritten.
5. The final benchmark was rebuilt in reference order, applying only the
   reviewed `final_human_label` values to disagreement rows.
6. Row count, schema, identity, labels, source quotas, and SHA-256 integrity
   were checked before release.

The complete review trail is documented in [V3 final benchmark](v3-final-benchmark.md)
and [V3 independent GPT review](v3-gpt-independent-review.md).

## Translation process

The translations were produced by **gpt-5.6-luna-max**, meaning the
`gpt-5.6-luna` model at `max` reasoning, executed through the **Codex
harness**. Each language was handled as a bounded CSV transformation: the
worker received the English rows and the target language, translated only
`sentence`, preserved all other columns, and validated 300 output rows before
the file was accepted. The English `en` file is an exact copy of the final
English benchmark and is included so the Hub release has one uniform file per
language.

The language set is the project-provided set supported by `sat-3l-sm`:

| ISO | Language | ISO | Language | ISO | Language |
| --- | --- | --- | --- | --- | --- |
| af | Afrikaans | am | Amharic | ar | Arabic |
| az | Azerbaijani | be | Belarusian | bg | Bulgarian |
| bn | Bengali | ca | Catalan | ceb | Cebuano |
| cs | Czech | cy | Welsh | da | Danish |
| de | German | el | Greek | en | English |
| eo | Esperanto | es | Spanish | et | Estonian |
| eu | Basque | fa | Persian | fi | Finnish |
| fr | French | fy | Western Frisian | ga | Irish |
| gd | Scottish Gaelic | gl | Galician | gu | Gujarati |
| ha | Hausa | he | Hebrew | hi | Hindi |
| hu | Hungarian | hy | Armenian | id | Indonesian |
| ig | Igbo | is | Icelandic | it | Italian |
| ja | Japanese | jv | Javanese | ka | Georgian |
| kk | Kazakh | km | Central Khmer | kn | Kannada |
| ko | Korean | ku | Kurdish | ky | Kirghiz |
| la | Latin | lt | Lithuanian | lv | Latvian |
| mg | Malagasy | mk | Macedonian | ml | Malayalam |
| mn | Mongolian | mr | Marathi | ms | Malay |
| mt | Maltese | my | Burmese | ne | Nepali |
| nl | Dutch | no | Norwegian | pa | Panjabi |
| pl | Polish | ps | Pushto | pt | Portuguese |
| ro | Romanian | ru | Russian | si | Sinhala |
| sk | Slovak | sl | Slovenian | sq | Albanian |
| sr | Serbian | sv | Swedish | ta | Tamil |
| te | Telugu | tg | Tajik | th | Thai |
| tr | Turkish | uk | Ukrainian | ur | Urdu |
| uz | Uzbek | vi | Vietnamese | xh | Xhosa |
| yi | Yiddish | yo | Yoruba | zh | Chinese |
| zu | Zulu |  |  |  |  |

There are **85 language files in total**: English plus 84 non-English
translations.

## Geographic distribution

![World map of V3 benchmark sentence locations](https://raw.githubusercontent.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/main/data/benchmark/v3/assets/v3-world-distribution.png)

The map shows the latitude/longitude metadata of the 300 English benchmark
records, colored by source. All language files retain these same coordinates,
so the geographic distribution is shared across the multilingual release. The
map uses Natural Earth 110m country outlines and a WGS84 longitude/latitude
display; it is a visualization of benchmark coverage, not a population or
source-density estimate.

The map is generated deterministically with:

```bash
uv run python scripts/build_v3_world_map.py
```

The generator validates `data/benchmark/v3/final/v3-final.csv`, reads the
geometry-only Natural Earth source at
`data/benchmark/v3/assets/natural-earth-110m-admin-0.geojson`, and writes the
PNG at `data/benchmark/v3/assets/v3-world-distribution.png`. It uses the
pinned Matplotlib 3.11.1 Agg renderer and performs no network access. The
boundary source provenance and upstream checksum are recorded in
`data/benchmark/v3/assets/README.md`.

## Release layout and checks

```text
data/benchmark/v3/
├── assets/
│   └── v3-world-distribution.png
├── final/
│   └── v3-final.csv
├── independent-review/
├── reference/
└── translations/
    ├── af/v3-final-af.csv
    ├── ...
    ├── en/v3-final-en.csv
    └── zu/v3-final-zu.csv
```

The translation manifest records the model/harness attribution, source hash,
language inventory, per-file row counts, and per-file hashes. Release checks
require exactly one file for each listed ISO code, 300 rows per file, the
English schema in the same order, no blank sentences, and byte-for-byte
agreement for every non-sentence field against the English benchmark.

The public release is the [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/landuse-sentence-relevance-golden-human-set).
Its old V2 files were removed before the multilingual files and dataset card
were uploaded. The card repeats the provenance, language inventory, file
layout, license notes, and geographic map so the public artifact is
self-describing.
