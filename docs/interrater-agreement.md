# Interrater agreement and adjudication

Three raters labelled the same 158-sentence V2 combined export: the human annotator, GPT, and Claude. This page reports how much they agreed, and how the disagreements were resolved into a final benchmark.

## At a glance

| Pair | Agreement | Cohen's kappa |
| --- | --- | --- |
| GPT vs Claude | 92.4% (146/158) | 0.84 |
| GPT vs human | 84.2% (133/158) | 0.68 |
| Claude vs human | 84.2% (133/158) | 0.68 |

All three agree on 127/158 (80.4%), Fleiss' kappa 0.73. The 31 remaining sentences were adjudicated by hand, which removed 4 and left a **154-row final benchmark**.

## Raters

| Rater | Source | Label column |
| --- | --- | --- |
| `human` | `results/evaluations/round-01/human.csv` | `label` |
| `gpt` | `results/evaluations/round-01/outputs/gpt.csv` | `llm_label` |
| `claude` | `results/evaluations/round-01/outputs/claude.csv` | `llm_label` |

The analysis reads all three files and never writes to them.

## Method

### Matching

Rows are matched by **exact sentence identity**, not by row position. Each file is indexed as a mapping from its `sentence` value to its label, so row order is irrelevant and two files that happen to share an order are still matched by content.

The analysis fails loudly, with no partial result, when a file lacks the `sentence` column or the rater's label column, a row is truncated, a label is not `yes` or `no` after trimming and lowercasing, a rater has no rows, a sentence repeats within one file, or a rater's sentence set differs from the reference rater's in either direction. Only when every rater covers exactly the same sentence set, once each, is any metric computed. Matched sentences are then processed in sorted order, so every output is deterministic.

### Metrics

- **Observed agreement** — the share of matched sentences two raters labelled the same.
- **Expected agreement** — chance agreement from the two raters' own label marginals: the sum over labels of the product of each rater's share of that label.
- **Cohen's kappa** — `(observed - expected) / (1 - expected)`. When expected agreement is exactly 1, both raters used one label everywhere, observed agreement is also 1, and kappa is reported as 1.
- **Confusion matrix** — counts with the first rater's labels as rows and the second rater's as columns.
- **Three-rater agreement** — sentences where all three chose the same label, as a count and a share.
- **Fleiss' kappa** — chance correction across all three raters at once: per sentence it measures the share of agreeing ordered rater pairs, averages that over sentences, and corrects it by the agreement expected from the pooled marginals. A corpus of one single label is reported as 1.

Pairs are enumerated over rater names in sorted order: `claude`–`gpt`, `claude`–`human`, `gpt`–`human`.

### Adjudication

Each of the 31 non-unanimous sentences carries one hand-assigned verdict in `final_label`: `yes`, `no`, or `remove`. The final benchmark takes the unanimous label where the raters agreed, the verdict where they did not, and drops every `remove`.

Adjudication is validated as strictly as the agreement itself: every disagreement must carry a verdict, no unanimous sentence may carry one, and no verdict may be anything but the three tokens above. Any breach aborts the run.

## Results

Run on 2026-09-10 over the 158-row V2 combined export. All three files matched one-to-one: **158 matched rows**, no missing, duplicate, or invalid labels.

Input SHA-256, as recorded in `agreement.json`:

| Rater | Rows | SHA-256 |
| --- | --- | --- |
| human | 158 | `42d1cc2a447c15ee559437ede01ef76fef1d694e16749482abf0a7a3ef4713e0` |
| gpt | 158 | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| claude | 158 | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The two model hashes match the ones recorded in [LLM labeling](llm-labeling.md), confirming the analysis read the published outputs unchanged.

### Label counts

| Rater | `yes` | `no` |
| --- | --- | --- |
| human | 79 | 79 |
| gpt | 96 | 62 |
| claude | 96 | 62 |

### Pairwise

| Pair | Agreements | Observed | Expected | Cohen's kappa |
| --- | --- | --- | --- | --- |
| claude vs gpt | 146 / 158 | 0.9241 | 0.5232 | 0.8407 |
| claude vs human | 133 / 158 | 0.8418 | 0.5000 | 0.6835 |
| gpt vs human | 133 / 158 | 0.8418 | 0.5000 | 0.6835 |

### Confusion matrices

Rows are the first rater's label, columns the second rater's.

| claude \ gpt | no | yes |
| --- | --- | --- |
| **no** | 56 | 6 |
| **yes** | 6 | 90 |

| claude \ human | no | yes |
| --- | --- | --- |
| **no** | 58 | 4 |
| **yes** | 21 | 75 |

| gpt \ human | no | yes |
| --- | --- | --- |
| **no** | 58 | 4 |
| **yes** | 21 | 75 |

The two model-versus-human matrices are numerically identical by coincidence: the models share the same 96/62 marginal and happen to agree with the human on the same number of rows in each cell. Their label vectors are not identical — they disagree with each other on 12 rows.

### Three raters

| Metric | Value |
| --- | --- |
| Matched rows | 158 |
| Unanimous rows | 127 |
| Unanimous agreement | 0.8038 |
| Fleiss' kappa | 0.7329 |

Both models label `yes` far more often than the human annotator: each assigns `yes` to 21 sentences the human called `no`, against only 4 the other way. The disagreement is a systematic recall-versus-precision offset, not scattered noise.

### Adjudication outcome

| Verdict | Sentences |
| --- | --- |
| `no` | 20 |
| `yes` | 7 |
| `remove` | 4 |

Of the 27 verdicts that kept a sentence, the human's original label was upheld 18 times, Claude's 13, and GPT's 11 — consistent with the models' `yes` bias, since two thirds of the verdicts went to `no`.

The result is `data/benchmark/v2-adjudicated.csv`: the human export with 4 rows removed and 9 labels replaced, keeping its own columns and row order.

| Final benchmark | Value |
| --- | --- |
| Rows | 154 |
| `yes` | 80 |
| `no` | 74 |
| Labels changed from the human original | 9 |
| Rows removed | 4 |

## Files

| Path | Committed | Contents |
| --- | --- | --- |
| [`data/interrater/disagreements.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/interrater/disagreements.csv) | yes | One row per non-unanimous sentence, generated |
| [`data/interrater/adjudication.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/interrater/adjudication.csv) | yes | The same rows plus the hand-assigned `final_label`; the only hand-edited file here |
| `results/evaluations/round-01/analysis/agreement.json` | no | Full machine-readable report: matched rows, label counts, pairwise metrics with confusion matrices, three-rater agreement, every disagreement, the adjudication summary, and a `sources` block with each input path, label column, row count, and SHA-256 |
| [`data/benchmark/v2-adjudicated.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/benchmark/v2-adjudicated.csv) | yes | The 154-row final benchmark: the human export's own nine columns, with removed rows dropped and adjudicated labels applied |

Committed tables live under `data/`; raw and intermediate artifacts stay under `results/` on the Seagate drive. See the [Data catalogue](results.md) for every file in the project.

### Review table columns

Both CSVs under `data/interrater/` share the same shape:

| Column | Meaning |
| --- | --- |
| `minority_rater` | The rater the other two outvoted. Empty if no label holds a strict majority |
| `minority_label` | The label that outvoted rater chose |
| `human`, `gpt`, `claude` | The three labels for that sentence, in that order |
| `final_label` | The adjudicated verdict — `yes`, `no`, or `remove`. Only in `adjudication.csv` |
| `sentence` | The sentence itself, immediately after the labels |
| `source`, `region`, `polygon_name`, `source_url` | Where the sentence came from |

Rows are sorted by `minority_rater`, then `minority_label`, then the sentence, so each systematic pattern reads as one block:

| Outvoted rater | Their label | Rows |
| --- | --- | --- |
| human | `no` | 17 |
| human | `yes` | 2 |
| gpt | `yes` | 4 |
| gpt | `no` | 2 |
| claude | `yes` | 4 |
| claude | `no` | 2 |

## Running it

```bash
./scripts/uv-seagate run python scripts/interrater_agreement.py
```

One command produces every output above. `--human`, `--gpt`, `--claude`, `--adjudication`, `--review-csv`, and `--output-directory` override the defaults. Outputs are byte-identical across repeated runs on unchanged inputs. A validation failure prints to standard error, returns exit status 1, and writes nothing.

The code lives in `src/landuse_sentence_relevance/analysis/`, is covered by unit tests under `tests/unit/analysis/` and `tests/unit/test_interrater_script.py`, and is part of the mutation gate described in [QA](qa.md).
