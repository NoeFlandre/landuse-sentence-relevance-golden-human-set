# V3 independent GPT review

This records the independent model check performed against the completed V3 reference benchmark. It does not replace the reference benchmark and it does not automatically change any human decision. The resulting human reassessment and final benchmark are documented in [V3 final benchmark](v3-final-benchmark.md).

## Artifact roles

| File | Role |
| --- | --- |
| `data/benchmark/v3/reference/v3-human-completed.csv` | The 300-row reference benchmark. It contains the completed labels: 100 Wikipedia, 100 Website, and 100 Description rows, with 50 `yes` and 50 `no` labels per source. |
| `data/benchmark/v3/independent-review/v3-independent-review-input.csv` | The blinded input sent for independent review. It contains the same 300 sentences, in the same order, with the metadata preserved and the `label` column removed. |
| `data/benchmark/v3/independent-review/v3-gpt-sol-5.6-extra-high.csv` | The GPT response supplied for this review. It contains the same 300 rows and a GPT `label`; its parsed content is preserved in canonical UTF-8 CSV form. |
| `data/benchmark/v3/independent-review/v3-human-gpt-disagreements.csv` | The human-review queue. It contains only disagreements and the columns `sentence`, `human_label`, `gpt_label`, and empty `final_human_label`. |

The GPT filename identifies the run as `GPT Sol 5.6 Extra High`. No additional API run identifier or generation metadata was supplied, so the response file and its Git history are the authoritative record of that assessment.

## Comparison method

1. Read the human reference and canonical GPT CSV as CSV. The submitted GPT file's BOM and CRLF line endings were normalized for repository storage; parsed fields and labels were unchanged.
2. Match rows by the stable `(sentence, h3_cell)` identity.
3. Verify that all 300 identities and all non-label metadata fields match exactly.
4. Compare only the human `label` and GPT `label` values.
5. Export rows where those labels differ, in the reference benchmark's order. No label is automatically resolved and no source file is overwritten.

The comparison found 258 agreements and 42 disagreements, for 86% raw agreement. GPT returned 180 `yes` and 120 `no` labels; the reference contains 150 `yes` and 150 `no` labels. The disagreement split is:

| Human | GPT | Rows |
| --- | --- | ---: |
| `no` | `yes` | 36 |
| `yes` | `no` | 6 |

## Classification prompt

The per-sentence classification rule was:

> Classify whether the TARGET SENTENCE contains information about the target place that could help characterize its land use, land cover, or geographic environment from remote sensing, either directly or through observable proxies.
>
> Return exactly one token: yes or no.
>
> Answer **yes** for information about vegetation, agriculture, forests, water, soil or surface, terrain, buildings, settlements, infrastructure, transport networks, mining, managed land, or other human or natural features with a spatial or remotely detectable signature.
>
> Answer **no** for information only about history, administration, people, events, demographics, economy, navigation, or activities with no meaningful land-use, land-cover, or remotely detectable implication.
>
> Output only the lowercase token yes or no.
>
> TARGET SENTENCE: {}

For the CSV exchange, the surrounding instruction was to classify every row's `sentence`, preserve the original rows and columns, append a final `label` column, and return only the completed CSV.

## Human reassessment

The review queue at `data/benchmark/v3/independent-review/v3-human-gpt-disagreements.csv` was filled by a human reviewer using lowercase `yes` or `no` in `final_human_label`. The completed decisions are stored in `data/benchmark/v3/independent-review/v3-resolved.csv`; the original queue remains unchanged as the pre-resolution record. `sentence`, `human_label`, and `gpt_label` were kept unchanged. The final column is the adjudicator's decision, not a third model vote.

The final benchmark was then generated as `data/benchmark/v3/final/v3-final.csv` by copying the reference rows in order and replacing only the 42 disagreement labels with their resolved values. Neither the reference, GPT response, nor review queue was overwritten.

## Storage and provenance

The V3 artifacts are small committed CSVs under `data/benchmark/v3/`; no candidate pool, session log, model cache, or duplicate dataset is copied into the repository. The GPT response, disagreement queue, resolved decisions, and final benchmark remain separate so the provenance is auditable.
