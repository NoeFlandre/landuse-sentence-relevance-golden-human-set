# V3 multilingual benchmark

The V3 multilingual release is a translation of the immutable English benchmark `data/benchmark/v3/final/v3-final.csv`. It has one CSV file for each language in `data/benchmark/v3/translations/<iso>/`. The project publishes the corresponding files in `data/translations/<iso>/` in the Hugging Face dataset.

## What is translated

The project translates only the `sentence` field. It copies these fields from the English final benchmark without semantic changes. They stay in the same row order:

```text
label, polygon_name, h3_cell, latitude, longitude, source, region, source_url
```

`label` is the final human-reviewed decision of the English benchmark. The project does not compute it again from a translated sentence. Each language file is therefore a parallel text view of the same 300 benchmark records. It is not a new annotation round. The project keeps the proper names, the URLs, the coordinates, the source metadata, and the three source quotas as supplied.

## English benchmark provenance

The project assembled the English file in this controlled sequence:

1. The project kept the completed human reference of 300 rows as the starting point.
2. The project sent a copy without labels for an independent GPT assessment. It used the validated land-use/land-cover remote-sensing prompt.
3. The project exported the 42 human/GPT disagreements to a minimal review queue.
4. A human reassessed every disagreement. The project did not overwrite the original reference or the GPT response.
5. The project rebuilt the final benchmark in the order of the reference. It applied only the reviewed `final_human_label` values to the disagreement rows.
6. The project checked the row count, the schema, the identity, the labels, the source quotas, and the SHA-256 integrity before the release.

The complete review trail is in [V3 final benchmark](v3-final-benchmark.md) and [V3 independent GPT review](v3-gpt-independent-review.md).

## Translation process

**gpt-5.6-luna-max** produced the translations. This is the `gpt-5.6-luna` model at `max` reasoning, which ran through the **Codex harness**. The project handled each language as a bounded CSV transformation. The worker received the English rows and the target language. It translated only `sentence`. It kept all other columns. It validated 300 output rows before the project accepted the file. The English `en` file is an exact copy of the final English benchmark. The project includes it so that the Hub release has one uniform file for each language.

The language set is the set that the project supplied and that `sat-3l-sm` supports:

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

There are **85 language files in total**: English and 84 non-English translations.

## Geographic distribution

![World map of V3 benchmark sentence locations](https://raw.githubusercontent.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/main/data/benchmark/v3/assets/v3-world-distribution.png)

The map shows the latitude and longitude metadata of the 300 English benchmark records. The colors show the source. All language files keep these same coordinates. The geographic distribution is therefore the same for the complete multilingual release. The map draws Natural Earth 110m land vectors in an equirectangular projection. It shows the coverage of the benchmark. It is not an estimate of the population or of the source density. The legend distinguishes the Description, Website, and Wikipedia points.

To generate the map deterministically, run:

```bash
uv run python scripts/build_v3_world_map.py
```

The generator validates `data/benchmark/v3/final/v3-final.csv`. It reads the committed Natural Earth land vectors at `data/benchmark/v3/assets/ne-110m-land.geojson`. It writes the PNG at `data/benchmark/v3/assets/v3-world-distribution.png`. It uses the pinned Matplotlib 3.11.1 Agg renderer. It does not use the network. The repository contains the exact bytes of the snapshot. The provenance and the checksum are in `data/benchmark/v3/assets/ne-110m-land-manifest.json` and in its README.

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

The translation manifest records the model and harness attribution, the source hash, the language inventory, the row count of each file, and the hash of each file. The release checks need these properties:

- Exactly one file exists for each listed ISO code.
- Each file has 300 rows.
- Each file has the English schema in the same order.
- No sentence is blank.
- Every non-sentence field agrees byte for byte with the English benchmark.

The public release is the [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/landuse-sentence-relevance-golden-human-set). The project removed its old V2 files before it uploaded the multilingual files and the dataset card. The card repeats the provenance, the language inventory, the file layout, the license notes, and the geographic map. The public artifact therefore describes itself.
