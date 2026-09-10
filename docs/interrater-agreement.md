# Interrater agreement

A one-screen version of the numbers is on [Interrater agreement at a glance](interrater-agreement-summary.md).

This page records how agreement is computed between the human benchmark and the two machine-labeled copies of the V2 combined export, and what the current run reports.

## Raters

| Rater | Source | Label column |
| --- | --- | --- |
| `human` | `results/annotations/benchmark/v2-wikipedia-website-combined.csv` | `label` |
| `gpt` | `results/annotations/llm/v2-wikipedia-website-combined-gpt-5.6-extra-high-2026-09-10.csv` | `llm_label` |
| `claude` | `results/annotations/llm/v2-wikipedia-website-combined-claude-opus-5-extra-2026-09-10.csv` | `llm_label` |

The human column is the ground truth. The analysis reads all three files, never writes to them, and never edits a human label.

## Methodology

### Matching

Rows are matched by **exact sentence identity**, not by row position. Each file is indexed as a mapping from its `sentence` value to its label. Row order in a source file is irrelevant to the result, and two files that happen to share an order are still matched by content.

The analysis fails loudly, with no partial result, when any of the following holds:

- a file lacks the `sentence` column or the rater's label column;
- a row is truncated, so the sentence or the label is absent;
- a label is not `yes` or `no` after trimming surrounding whitespace and lowercasing;
- a rater has no rows;
- a sentence appears more than once within one file; or
- a rater's sentence set differs from the reference rater's set, in either direction. The error names the missing and the unexpected sentences.

Only when every rater covers exactly the same sentence set, once each, is any metric computed. The matched sentences are then processed in sorted order, so every output is deterministic.

### Metrics

Let `N` be the number of matched sentences and let the label set be `{no, yes}`.

- **Observed agreement** for a pair is the share of matched sentences given the same label by both raters.
- **Expected agreement** for a pair is the chance agreement implied by the two raters' own label marginals: the sum, over labels, of the product of each rater's share of that label.
- **Cohen's kappa** is `(observed - expected) / (1 - expected)`. When expected agreement is exactly 1, both raters used one single label everywhere, observed agreement is also 1, and kappa is reported as 1.
- **Confusion matrix** for a pair cross-tabulates counts with the first rater's labels as rows and the second rater's labels as columns.
- **Three-rater agreement** counts the sentences on which all three raters chose the same label, as a count and as a share of `N`.
- **Fleiss' kappa** extends chance correction to all three raters at once. Per sentence it measures the share of agreeing ordered rater pairs, averages that over sentences, and corrects it by the agreement expected from the pooled label marginals. A corpus in which every rating is the same single label is reported as 1.
- **Label counts** are each rater's `yes`/`no` totals over the matched sentences.
- **Disagreements** list every sentence that was not unanimous, with each rater's label.

Pairs are enumerated over rater names in sorted order, so the reported pairs are `claude`–`gpt`, `claude`–`human`, and `gpt`–`human`.

## Running it

```bash
./scripts/uv-seagate run python scripts/interrater_agreement.py
```

The command reads the three default paths above and writes two files into `results/analysis/interrater/`:

| File | Contents |
| --- | --- |
| `results/analysis/interrater/agreement.json` | The full machine-readable report: matched-row count, label counts, pairwise metrics with confusion matrices, three-rater agreement, every disagreement, and a `sources` block with each input path, label column, row count, and SHA-256 |
| `docs/data/interrater-disagreements.csv` | The reviewable disagreement table described below |

Both outputs are byte-identical across repeated runs on unchanged inputs. `--human`, `--gpt`, `--claude`, `--output-directory`, and `--review-csv` override the defaults. A validation failure prints to standard error, returns exit status 1, and writes nothing.

`results/` stays on the Seagate project drive and is not committed; the review CSV is committed with the documentation so it can be read without the drive.

## Reviewing the disagreements

[`interrater-disagreements.csv`](data/interrater-disagreements.csv) holds one row per non-unanimous sentence — 31 of the 158 — with the columns a reviewer needs and nothing else:

| Column | Meaning |
| --- | --- |
| `minority_rater` | The rater the other two outvoted. Empty if no label holds a strict majority |
| `minority_label` | The label that outvoted rater chose |
| `human`, `gpt`, `claude` | The three labels for that sentence, in that order |
| `sentence` | The sentence itself, immediately after the labels |
| `source`, `region`, `polygon_name`, `source_url` | Where the sentence came from |

Rows are sorted by `minority_rater`, then `minority_label`, then the sentence, so each systematic pattern reads as one block. For the current run:

| Outvoted rater | Their label | Rows |
| --- | --- | --- |
| human | `no` | 17 |
| human | `yes` | 2 |
| gpt | `yes` | 4 |
| gpt | `no` | 2 |
| claude | `yes` | 4 |
| claude | `no` | 2 |

The largest block is the 17 sentences where both models said `yes` and the human annotator said `no`.

## Current results

Run on 2026-09-10 over the 158-row V2 combined export. All three files matched one-to-one: **158 matched rows**, no missing, duplicate, or invalid labels.

Input SHA-256, as recorded in `agreement.json`:

| Rater | Rows | SHA-256 |
| --- | --- | --- |
| human | 158 | `42d1cc2a447c15ee559437ede01ef76fef1d694e16749482abf0a7a3ef4713e0` |
| gpt | 158 | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| claude | 158 | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The two model hashes are identical to the ones recorded in [LLM labeling](llm-labeling.md), confirming that the analysis read the published outputs unchanged.

### Label counts

| Rater | `yes` | `no` |
| --- | --- | --- |
| human | 79 | 79 |
| gpt | 96 | 62 |
| claude | 96 | 62 |

### Pairwise

| Pair | Agreements | Observed agreement | Expected agreement | Cohen's kappa |
| --- | --- | --- | --- | --- |
| claude vs gpt | 146 / 158 | 0.9241 | 0.5232 | 0.8407 |
| claude vs human | 133 / 158 | 0.8418 | 0.5000 | 0.6835 |
| gpt vs human | 133 / 158 | 0.8418 | 0.5000 | 0.6835 |

### Confusion matrices

Rows are the first rater's label, columns the second rater's label.

claude vs gpt:

| claude \ gpt | no | yes |
| --- | --- | --- |
| **no** | 56 | 6 |
| **yes** | 6 | 90 |

claude vs human:

| claude \ human | no | yes |
| --- | --- | --- |
| **no** | 58 | 4 |
| **yes** | 21 | 75 |

gpt vs human:

| gpt \ human | no | yes |
| --- | --- | --- |
| **no** | 58 | 4 |
| **yes** | 21 | 75 |

The two model-versus-human matrices are numerically identical by coincidence: the two models share the same 96/62 marginal and happen to agree with the human on the same number of rows in each cell. Their label vectors are not identical, and they disagree with each other on 12 rows.

### Three raters

| Metric | Value |
| --- | --- |
| Matched rows | 158 |
| Unanimous rows | 127 |
| Unanimous agreement | 0.8038 |
| Fleiss' kappa | 0.7329 |

31 sentences were not unanimous; they are listed in `results/analysis/interrater/disagreements.csv`. Both models label `yes` far more often than the human annotator: each assigns `yes` to 21 sentences the human called `no`, against only 4 in the other direction. The disagreement is therefore mostly a systematic recall-versus-precision offset, not scattered noise.

These numbers describe agreement only. They do not make either machine-labeled file a benchmark; the human file remains the ground truth.

## Quality gate

The analysis lives in `src/landuse_sentence_relevance/analysis/`, is covered by unit tests under `tests/unit/analysis/` and `tests/unit/test_interrater_script.py`, and is part of the mutation-testing gate described in [QA](qa.md), which requires zero surviving mutants and a CRAP score below 6.
