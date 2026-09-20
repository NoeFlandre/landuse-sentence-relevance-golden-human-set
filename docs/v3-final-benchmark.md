# V3 final benchmark

`data/benchmark/v3/final/v3-final.csv` is the final 300-row V3 benchmark for this annotation round. It is derived from the completed human reference and the human-reviewed resolution of the independent GPT disagreements. The GPT assessment informed the review but was never applied automatically.

## Artifact layout

The files are grouped by their role rather than kept in one flat directory:

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
| `reference/v3-human-completed.csv` | Original 300-row human benchmark, with the initial `label` values. |
| `independent-review/v3-independent-review-input.csv` | Label-free copy sent for independent GPT assessment. |
| `independent-review/v3-gpt-sol-5.6-extra-high.csv` | Supplied GPT response, with one GPT `label` per sentence. |
| `independent-review/v3-human-gpt-disagreements.csv` | Generated 42-row queue containing only human/GPT disagreements. Its `final_human_label` column was blank before reassessment. |
| `independent-review/v3-resolved.csv` | Human-reviewed copy of the disagreement queue, with every `final_human_label` completed. |
| `final/v3-final.csv` | Final benchmark. It keeps the reference schema and order and applies the reviewed decisions. |

The independent-review prompt and the original comparison are documented in [V3 independent GPT review](v3-gpt-independent-review.md).

## Finalization process

1. The 300-row human reference was kept unchanged as the starting point.
2. The independent GPT response was compared against it by `(sentence, h3_cell)`. All 300 identities and non-label metadata fields matched.
3. The 42 disagreements were exported to the minimal review queue. No GPT label changed the reference automatically.
4. The human reviewer completed `final_human_label` for all 42 queue rows. The reviewed file was saved as `v3-resolved.csv`; the original blank queue was retained.
5. The final CSV was generated in reference order. For the 42 disagreement sentences, `label` comes from `final_human_label`. For the other 258 rows, the original human `label` is retained.
6. The result was validated for row count, schema, unique sentence identity, allowed labels, source quotas, and absence of blank final labels.

The submitted resolved CSV and GPT CSV were stored in canonical UTF-8 CSV form. Only BOM/line-ending representation was normalized where needed; parsed fields and decisions were not changed.

## Final contents

The final benchmark has 300 rows, 100 from each source, and 160 `yes` / 140 `no` labels:

| Source | Yes | No | Total |
| --- | ---: | ---: | ---: |
| Wikipedia | 54 | 46 | 100 |
| Website | 49 | 51 | 100 |
| Description | 57 | 43 | 100 |
| **Total** | **160** | **140** | **300** |

Relative to the reference, 16 labels changed: 13 `no` to `yes` and 3 `yes` to `no`. The other 284 rows retain their reference label, including 26 of the 42 originally disputed rows that the reviewer confirmed.

## Integrity record

- Final CSV SHA-256: `cabbae3823d012aa61139d5a0c37c7f00f9d8873d32d8874dba89cba05a37575`
- Resolved-review CSV SHA-256: `6b55ff326a223d7b3153d6824174160cdf1080897ab4a9f18355fcdeffeb64da`
- Final schema: `sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url`
- Final labels: lowercase `yes` or `no`; no blank labels

The final artifact is immutable for this round. A later correction should be saved as another versioned artifact with its own resolution record rather than edited in place.
