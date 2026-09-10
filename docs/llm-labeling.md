# LLM labeling

This page records two machine-labeled copies of the V2 combined results. They are separate from the human benchmark and do not replace human annotations.

## Runs

Both runs used the shared prompt below, the same 158-row input, and web search disabled.

| Run | Model | Time | Output | Labels | Output SHA-256 |
| --- | --- | --- | --- | --- | --- |
| GPT | GPT 5.6 Extra High | 2026-09-10 10:09 Europe/Paris | `results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv` | 96 `yes`, 62 `no` | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| Claude | Claude Opus 5 Extra | 2026-09-10 10:14 Europe/Paris | `results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv` | 96 `yes`, 62 `no` | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The input is `results/annotations/unlabeled/v2-wikipedia-website-combined.csv` with 158 rows: 100 Wikipedia and 58 website. Its SHA-256 is `a35e8a9aaf95a097d7b0de25778b6aa9a5b5b9ebf665d9b940a0701bab53950d`. Both outputs preserve the input rows, columns, values, and order, and add only `llm_label` with lowercase `yes` or `no`.

The two outputs agree on 146 of 158 rows and disagree on 12. Agreement with the human benchmark is quantified in [Interrater agreement](interrater-agreement.md). This is recorded for review; neither output is silently selected as ground truth. The human benchmark remains `results/annotations/benchmark/v1-human-annotated.csv`.

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

Any later run must use a new dated artifact and record its model, prompt, input hash, output hash, and row-level validation separately.
