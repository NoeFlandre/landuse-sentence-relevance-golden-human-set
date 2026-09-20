---
pretty_name: Land-use sentence relevance golden human set (V3 multilingual)
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- geospatial
- multilingual
size_categories:
- n<1K
license: other
language:
- af
- am
- ar
- az
- be
- bg
- bn
- ca
- ceb
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hu
- hy
- id
- ig
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- mt
- my
- ne
- nl
- no
- pa
- pl
- ps
- pt
- ro
- ru
- si
- sk
- sl
- sq
- sr
- sv
- ta
- te
- tg
- th
- tr
- uk
- ur
- uz
- vi
- xh
- yi
- yo
- zh
- zu
---

# Land-use sentence relevance golden human set

This release contains the final 300-row V3 benchmark in English plus one
parallel CSV for each of the 84 non-English project-provided `sat-3l-sm`
language codes. There are 85 language files in total.

## Files

Every file is at
`data/translations/<iso>/v3-final-<iso>.csv`. The nine columns are:

`sentence`, `label`, `polygon_name`, `h3_cell`, `latitude`, `longitude`,
`source`, `region`, `source_url`.

Only `sentence` is translated. Labels are the final human-reviewed English
decisions and all geographic/source metadata is retained in the same row order
in every language file. `data/translations/en/v3-final-en.csv` is an exact copy
of the final English benchmark.

## Geographical distribution

![World map showing the geographical distribution of benchmark sentences](assets/v3-world-distribution.png)

The map shows the coordinates of all 300 benchmark records, colored by source
(100 Description, 100 Website, 100 Wikipedia). The same coordinates are
retained across all language files. Country outlines use Natural Earth 110m
data; the map is a coverage visualization, not a population-density estimate.

## English benchmark provenance

The English benchmark was built from a completed human reference. An
independent GPT assessment was used to identify 42 disagreements, those cases
were reassessed by a human, and the final file was rebuilt in reference order.
Only 16 labels changed after review (13 `no` to `yes`, 3 `yes` to `no`). The
final distribution is 160 `yes` and 140 `no`, with 100 rows per source.

The repository documents the complete chain in
[`docs/v3-final-benchmark.md`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/docs/v3-final-benchmark.md)
and the multilingual release method in
[`docs/v3-translations.md`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/docs/v3-translations.md).

## Translation provenance

Translation was performed by **gpt-5.6-luna-max**: the available
`gpt-5.6-luna` model at `max` reasoning, executed through the **Codex harness**.
The English label is deliberately not reclassified after translation. The
translation manifest records the source hash, model/harness attribution, row
counts, language inventory, and SHA-256 for every file.

## Licensing and attribution

The project code and original benchmark curation are Apache-2.0. The included
upstream-derived material retains its own terms: OpenStreetMap-derived fields
are subject to the [ODbL](https://opendatacommons.org/licenses/odbl/), and
Wikipedia-derived text is subject to [CC BY-SA
4.0](https://creativecommons.org/licenses/by-sa/4.0/). See the repository
[`NOTICE`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/NOTICE)
and [licensing documentation](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/docs/licensing.md).
