# V3 final benchmark

`data/benchmark/v3/final/v3-final.csv` is the final V3 benchmark of 300 rows for this annotation round. It comes from the completed human reference and from the human-reviewed resolution of the independent GPT disagreements. The GPT assessment informed the review. The project never applied it automatically.

## Artifact layout

The files are in groups by role. They are not in one flat directory:

```text
data/benchmark/v3/
├── reference/
│   └── v3-human-completed.csv
├── independent-review/
│   ├── v3-independent-review-input.csv
│   ├── v3-gpt-sol-5.6-extra-high.csv
│   ├── v3-human-gpt-disagreements.csv
│   └── v3-resolved.csv
└── final/
    └── v3-final.csv
```

| Artifact | Purpose |
| --- | --- |
| `reference/v3-human-completed.csv` | The original human benchmark of 300 rows, with the initial `label` values. |
| `independent-review/v3-independent-review-input.csv` | A copy without labels. The project sent it for the independent GPT assessment. |
| `independent-review/v3-gpt-sol-5.6-extra-high.csv` | The supplied GPT response, with one GPT `label` for each sentence. |
| `independent-review/v3-human-gpt-disagreements.csv` | The generated queue of 42 rows. It contains only the human/GPT disagreements. Its `final_human_label` column was blank before the reassessment. |
| `independent-review/v3-resolved.csv` | The human-reviewed copy of the disagreement queue. Every `final_human_label` is complete. |
| `final/v3-final.csv` | The final benchmark. It keeps the schema and the order of the reference. It applies the reviewed decisions. |

The independent-review prompt and the original comparison are in [V3 independent GPT review](v3-gpt-independent-review.md).

## Finalization process

1. The project kept the human reference of 300 rows unchanged as the starting point.
2. The project compared the independent GPT response with the reference by `(sentence, h3_cell)`. All 300 identities and all non-label metadata fields matched.
3. The project exported the 42 disagreements to the minimal review queue. No GPT label changed the reference automatically.
4. The human reviewer completed `final_human_label` for all 42 queue rows. The project saved the reviewed file as `v3-resolved.csv`. It kept the original blank queue.
5. The project generated the final CSV in the order of the reference. For the 42 disagreement sentences, `label` comes from `final_human_label`. For the other 258 rows, the project keeps the original human `label`.
6. The project validated the result. The checks were: row count, schema, unique sentence identity, allowed labels, source quotas, and no blank final labels.

The project stored the submitted resolved CSV and the GPT CSV in canonical UTF-8 CSV form. Where necessary, it normalized only the BOM and the line endings. It did not change the parsed fields or the decisions.

## Final contents

The final benchmark has 300 rows, 100 from each source. It has 160 `yes` labels and 140 `no` labels:

| Source | Yes | No | Total |
| --- | ---: | ---: | ---: |
| Wikipedia | 54 | 46 | 100 |
| Website | 49 | 51 | 100 |
| Description | 57 | 43 | 100 |
| **Total** | **160** | **140** | **300** |

Compared with the reference, 16 labels changed: 13 from `no` to `yes` and 3 from `yes` to `no`. The other 284 rows keep their reference label. This includes 26 of the 42 originally disputed rows that the reviewer confirmed.

## Integrity record

- Final CSV SHA-256: `cabbae3823d012aa61139d5a0c37c7f00f9d8873d32d8874dba89cba05a37575`
- Resolved-review CSV SHA-256: `6b55ff326a223d7b3153d6824174160cdf1080897ab4a9f18355fcdeffeb64da`
- Final schema: `sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url`
- Final labels: lowercase `yes` or `no`. No label is blank.

The final artifact is immutable for this round. Save a later correction as another versioned artifact with its own resolution record. Do not edit the artifact in place.
