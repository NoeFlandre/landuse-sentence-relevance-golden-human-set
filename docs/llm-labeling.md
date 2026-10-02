# LLM labeling

This page records two machine-labeled copies of the V2 combined results. They are separate from the human benchmark. They do not replace the human annotations.

This page is the historical record of Round 1. Its complete release snapshot is in `data/provenance/round-01/`. For each new prompt iteration, use [LLM evaluation rounds](llm-evaluation-rounds.md).

## Runs

Both runs used the shared prompt below, the same input of 158 rows, and no web search.

| Run | Model | Time | Output | Labels | Output SHA-256 |
| --- | --- | --- | --- | --- | --- |
| GPT | GPT 5.6 Extra High | 2026-09-10 10:09 Europe/Paris | `data/provenance/round-01/outputs/gpt.csv` | 96 `yes`, 62 `no` | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| Claude | Claude Opus 5 Extra | 2026-09-10 10:14 Europe/Paris | `data/provenance/round-01/outputs/claude.csv` | 96 `yes`, 62 `no` | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The input is `data/provenance/round-01/input.csv`. It has 158 rows: 100 Wikipedia and 58 website. Its SHA-256 is `a35e8a9aaf95a097d7b0de25778b6aa9a5b5b9ebf665d9b940a0701bab53950d`. Both outputs keep the rows, columns, values, and order of the input. They add only `llm_label` with the lowercase value `yes` or `no`.

On the original input of 158 rows, the two outputs agree on 146 rows. They disagree on 12 rows. Against the final human benchmark of 154 rows, Claude agrees on 140/154 (Cohen's kappa 0.8168). GPT agrees on 138/154 (Cohen's kappa 0.7906). The canonical report is in [Interrater agreement](interrater-agreement.md).

The project records these results for review. It does not select either output as ground truth. Neither machine file is a benchmark. The benchmark is `data/benchmark/v2-adjudicated.csv`. The project built it when it adjudicated the original disagreements between these two runs and the human annotator.

## Shared prompt

```text
Classify every row in the attached CSV. Use only the `sentence` column as the TARGET SENTENCE. Apply the following prompt independently to each row, replacing `{}` with that row's sentence.
Return a downloadable CSV with exactly the same 158 rows, in the same order, preserving every original column and value exactly. Add exactly one column named `llm_label`. Each value must be exactly lowercase `yes` or `no`. Do not add explanations, markdown, code fences, or other columns. Do not overwrite or infer any human annotation. Do not use web search.
Prompt:
Classify whether the TARGET SENTENCE contains information about the target place that could help characterize its land use, land cover, or geographic environment from remote sensing, either directly or through observable proxies.
Return exactly one token: yes or no.
Answer yes for information about vegetation, agriculture, forests, water, soil or surface, terrain, buildings, settlements, infrastructure, transport networks, mining, managed land, or other human or natural features with a spatial or remotely detectable signature.
Answer no for information only about history, administration, people, events, demographics, economy, navigation, or activities with no meaningful land-use, land-cover, or remotely detectable implication.
Output only the lowercase token yes or no.
TARGET SENTENCE: {}
```

Round 1 is immutable. Each later run must use a new numbered round. Record the model, the prompt, the input hash, the output hash, and the row-level validation separately. Follow [LLM evaluation rounds](llm-evaluation-rounds.md).
