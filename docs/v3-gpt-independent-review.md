# V3 independent GPT review

This page records the independent model check against the completed V3 reference benchmark. The check does not replace the reference benchmark. It does not change a human decision automatically. The human reassessment and the final benchmark are in [V3 final benchmark](v3-final-benchmark.md).

## Artifact roles

| File | Role |
| --- | --- |
| `data/benchmark/v3/reference/v3-human-completed.csv` | The reference benchmark of 300 rows. It has the completed labels: 100 Wikipedia, 100 Website, and 100 Description rows, with 50 `yes` and 50 `no` labels for each source. |
| `data/benchmark/v3/independent-review/v3-independent-review-input.csv` | The blinded input that the project sent for the independent review. It has the same 300 sentences in the same order. It keeps the metadata. It does not have the `label` column. |
| `data/benchmark/v3/independent-review/v3-gpt-sol-5.6-extra-high.csv` | The GPT response that the project received for this review. It has the same 300 rows and a GPT `label`. The project keeps its parsed content in canonical UTF-8 CSV form. |
| `data/benchmark/v3/independent-review/v3-human-gpt-disagreements.csv` | The human-review queue. It has only the disagreements. Its columns are `sentence`, `human_label`, `gpt_label`, and an empty `final_human_label`. |

The GPT filename identifies the run as `GPT Sol 5.6 Extra High`. The project received no other API run identifier and no generation metadata. The response file and its Git history are the authoritative record of the assessment.

## Comparison method

1. Read the human reference and the canonical GPT CSV as CSV. The project normalized the BOM and the CRLF line endings of the submitted GPT file for repository storage. It did not change the parsed fields and labels.
2. Match the rows by the stable `(sentence, h3_cell)` identity.
3. Verify that all 300 identities and all non-label metadata fields match exactly.
4. Compare only the human `label` values and the GPT `label` values.
5. Export the rows where these labels differ, in the order of the reference benchmark. Do not resolve a label automatically. Do not overwrite a source file.

The comparison found 258 agreements and 42 disagreements. The raw agreement is 86%. GPT returned 180 `yes` labels and 120 `no` labels. The reference has 150 `yes` labels and 150 `no` labels. The disagreements split in this way:

| Human | GPT | Rows |
| --- | --- | ---: |
| `no` | `yes` | 36 |
| `yes` | `no` | 6 |

## Classification prompt

This was the classification rule for each sentence:

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

For the CSV exchange, the instruction around the prompt told the model to do these things: classify the `sentence` of every row, keep the original rows and columns, add a final `label` column, and return only the completed CSV.

## Human reassessment

A human reviewer filled the review queue at `data/benchmark/v3/independent-review/v3-human-gpt-disagreements.csv`. The reviewer used lowercase `yes` or `no` in `final_human_label`. The completed decisions are in `data/benchmark/v3/independent-review/v3-resolved.csv`. The original queue stays unchanged as the record before the resolution. The project kept `sentence`, `human_label`, and `gpt_label` unchanged. The final column is the decision of the adjudicator. It is not a third model vote.

The project then generated the final benchmark as `data/benchmark/v3/final/v3-final.csv`. It copied the reference rows in order. It replaced only the 42 disagreement labels with their resolved values. The project did not overwrite the reference, the GPT response, or the review queue.

## Storage and provenance

The V3 artifacts are small committed CSV files in `data/benchmark/v3/`. The repository has no copy of a candidate pool, a session log, a model cache, or a duplicate dataset. The GPT response, the disagreement queue, the resolved decisions, and the final benchmark stay separate. A reviewer can then audit the provenance.
