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
configs:
- config_name: v3-multilingual
  data_files:
  - split: af
    path: data/translations/af/v3-final-af.csv
  - split: am
    path: data/translations/am/v3-final-am.csv
  - split: ar
    path: data/translations/ar/v3-final-ar.csv
  - split: az
    path: data/translations/az/v3-final-az.csv
  - split: be
    path: data/translations/be/v3-final-be.csv
  - split: bg
    path: data/translations/bg/v3-final-bg.csv
  - split: bn
    path: data/translations/bn/v3-final-bn.csv
  - split: ca
    path: data/translations/ca/v3-final-ca.csv
  - split: ceb
    path: data/translations/ceb/v3-final-ceb.csv
  - split: cs
    path: data/translations/cs/v3-final-cs.csv
  - split: cy
    path: data/translations/cy/v3-final-cy.csv
  - split: da
    path: data/translations/da/v3-final-da.csv
  - split: de
    path: data/translations/de/v3-final-de.csv
  - split: el
    path: data/translations/el/v3-final-el.csv
  - split: en
    path: data/translations/en/v3-final-en.csv
  - split: eo
    path: data/translations/eo/v3-final-eo.csv
  - split: es
    path: data/translations/es/v3-final-es.csv
  - split: et
    path: data/translations/et/v3-final-et.csv
  - split: eu
    path: data/translations/eu/v3-final-eu.csv
  - split: fa
    path: data/translations/fa/v3-final-fa.csv
  - split: fi
    path: data/translations/fi/v3-final-fi.csv
  - split: fr
    path: data/translations/fr/v3-final-fr.csv
  - split: fy
    path: data/translations/fy/v3-final-fy.csv
  - split: ga
    path: data/translations/ga/v3-final-ga.csv
  - split: gd
    path: data/translations/gd/v3-final-gd.csv
  - split: gl
    path: data/translations/gl/v3-final-gl.csv
  - split: gu
    path: data/translations/gu/v3-final-gu.csv
  - split: ha
    path: data/translations/ha/v3-final-ha.csv
  - split: he
    path: data/translations/he/v3-final-he.csv
  - split: hi
    path: data/translations/hi/v3-final-hi.csv
  - split: hu
    path: data/translations/hu/v3-final-hu.csv
  - split: hy
    path: data/translations/hy/v3-final-hy.csv
  - split: id
    path: data/translations/id/v3-final-id.csv
  - split: ig
    path: data/translations/ig/v3-final-ig.csv
  - split: is
    path: data/translations/is/v3-final-is.csv
  - split: it
    path: data/translations/it/v3-final-it.csv
  - split: ja
    path: data/translations/ja/v3-final-ja.csv
  - split: jv
    path: data/translations/jv/v3-final-jv.csv
  - split: ka
    path: data/translations/ka/v3-final-ka.csv
  - split: kk
    path: data/translations/kk/v3-final-kk.csv
  - split: km
    path: data/translations/km/v3-final-km.csv
  - split: kn
    path: data/translations/kn/v3-final-kn.csv
  - split: ko
    path: data/translations/ko/v3-final-ko.csv
  - split: ku
    path: data/translations/ku/v3-final-ku.csv
  - split: ky
    path: data/translations/ky/v3-final-ky.csv
  - split: la
    path: data/translations/la/v3-final-la.csv
  - split: lt
    path: data/translations/lt/v3-final-lt.csv
  - split: lv
    path: data/translations/lv/v3-final-lv.csv
  - split: mg
    path: data/translations/mg/v3-final-mg.csv
  - split: mk
    path: data/translations/mk/v3-final-mk.csv
  - split: ml
    path: data/translations/ml/v3-final-ml.csv
  - split: mn
    path: data/translations/mn/v3-final-mn.csv
  - split: mr
    path: data/translations/mr/v3-final-mr.csv
  - split: ms
    path: data/translations/ms/v3-final-ms.csv
  - split: mt
    path: data/translations/mt/v3-final-mt.csv
  - split: my
    path: data/translations/my/v3-final-my.csv
  - split: ne
    path: data/translations/ne/v3-final-ne.csv
  - split: nl
    path: data/translations/nl/v3-final-nl.csv
  - split: no
    path: data/translations/no/v3-final-no.csv
  - split: pa
    path: data/translations/pa/v3-final-pa.csv
  - split: pl
    path: data/translations/pl/v3-final-pl.csv
  - split: ps
    path: data/translations/ps/v3-final-ps.csv
  - split: pt
    path: data/translations/pt/v3-final-pt.csv
  - split: ro
    path: data/translations/ro/v3-final-ro.csv
  - split: ru
    path: data/translations/ru/v3-final-ru.csv
  - split: si
    path: data/translations/si/v3-final-si.csv
  - split: sk
    path: data/translations/sk/v3-final-sk.csv
  - split: sl
    path: data/translations/sl/v3-final-sl.csv
  - split: sq
    path: data/translations/sq/v3-final-sq.csv
  - split: sr
    path: data/translations/sr/v3-final-sr.csv
  - split: sv
    path: data/translations/sv/v3-final-sv.csv
  - split: ta
    path: data/translations/ta/v3-final-ta.csv
  - split: te
    path: data/translations/te/v3-final-te.csv
  - split: tg
    path: data/translations/tg/v3-final-tg.csv
  - split: th
    path: data/translations/th/v3-final-th.csv
  - split: tr
    path: data/translations/tr/v3-final-tr.csv
  - split: uk
    path: data/translations/uk/v3-final-uk.csv
  - split: ur
    path: data/translations/ur/v3-final-ur.csv
  - split: uz
    path: data/translations/uz/v3-final-uz.csv
  - split: vi
    path: data/translations/vi/v3-final-vi.csv
  - split: xh
    path: data/translations/xh/v3-final-xh.csv
  - split: yi
    path: data/translations/yi/v3-final-yi.csv
  - split: yo
    path: data/translations/yo/v3-final-yo.csv
  - split: zh
    path: data/translations/zh/v3-final-zh.csv
  - split: zu
    path: data/translations/zu/v3-final-zu.csv
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

The Dataset Viewer exposes these files through the `v3-multilingual`
configuration as 85 named splits, one per ISO language code.

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
