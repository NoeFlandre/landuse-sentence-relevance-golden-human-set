# Interrater agreement and adjudication

The human annotator, GPT, and Claude originally labelled the same 158-sentence V2 combined export. The final adjudicated human benchmark contains 154 rows. The metrics below are the canonical Round 1 metrics against those 154 benchmark rows; the original 158-row report is preserved under `data/provenance/round-01/analysis/historical-158/`.

## At a glance

| Pair | Agreement | Cohen's kappa |
| --- | --- | --- |
| GPT vs Claude | 92.2% (142/154) | 0.84 |
| GPT vs human | 89.6% (138/154) | 0.79 |
| Claude vs human | 90.9% (140/154) | 0.82 |

All three agree on 133/154 (86.4%), Fleiss' kappa 0.81. The 31 disagreements in the original 158-row run were adjudicated by hand; 4 were removed, leaving the **154-row final benchmark** used here.

## Raters

| Rater | Source | Label column |
| --- | --- | --- |
| `human` | `data/provenance/round-01/benchmark.csv` | `label` |
| `gpt` | `data/provenance/round-01/outputs/gpt.csv` | `llm_label` |
| `claude` | `data/provenance/round-01/outputs/claude.csv` | `llm_label` |

The canonical report uses the final 154-row benchmark as its reference. The GPT and Claude files are the historical 158-row outputs; their four rows removed during human adjudication are excluded from the canonical comparison. The original 158-row human labels remain at `data/provenance/round-01/human.csv`.

## Method

### Matching

Rows are matched by **exact sentence identity**, not by row position. Each file is indexed as a mapping from its `sentence` value to its label, so row order is irrelevant and two files that happen to share an order are still matched by content. For the canonical Round 1 report, the 154 sentences in `benchmark.csv` define the reference set; the four historical model rows absent from that set are excluded before the one-to-one validation.

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

The canonical report was regenerated over the final benchmark: **154 matched rows**, with no missing, duplicate, or invalid labels in the reference scope. The model source files each contain 158 rows; 4 rows are excluded because they were removed from the final human benchmark. The original 158-row report is retained under `analysis/historical-158/`.

Input SHA-256, as recorded in `agreement.json`:

| Rater | Source rows | Rows used | SHA-256 |
| --- | --- | --- | --- |
| human benchmark | 154 | 154 | `ac185e51835eb626c932744009873382aa80027d26e93615dbde28d23af1d5fd` |
| gpt | 158 | 154 | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| claude | 158 | 154 | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The two model hashes match the ones recorded in [LLM labeling](llm-labeling.md), confirming the analysis read the published outputs unchanged.

### Label counts

| Rater | `yes` | `no` |
| --- | --- | --- |
| human benchmark | 80 | 74 |
| gpt | 92 | 62 |
| claude | 92 | 62 |

### Pairwise

| Pair | Agreements | Observed | Expected | Cohen's kappa |
| --- | --- | --- | --- | --- |
| claude vs gpt | 142 / 154 | 0.9221 | 0.5190 | 0.8380 |
| claude vs human benchmark | 140 / 154 | 0.9091 | 0.5038 | 0.8168 |
| gpt vs human benchmark | 138 / 154 | 0.8961 | 0.5038 | 0.7906 |

### Confusion matrices

Rows are the first rater's label, columns the second rater's.

| claude \ gpt | no | yes |
| --- | --- | --- |
| **no** | 56 | 6 |
| **yes** | 6 | 86 |

| claude \ human benchmark | no | yes |
| --- | --- | --- |
| **no** | 61 | 1 |
| **yes** | 13 | 79 |

| gpt \ human benchmark | no | yes |
| --- | --- | --- |
| **no** | 60 | 2 |
| **yes** | 14 | 78 |

The two models still skew toward `yes` relative to the final human benchmark. Claude assigns `yes` to 13 sentences labelled `no` by the benchmark and `no` to 1 benchmark `yes`; GPT assigns `yes` to 14 benchmark `no` sentences and `no` to 2 benchmark `yes` sentences. The models disagree with each other on 12 of the original 158 rows, 6 of which remain in the 154-row reference scope.

### Three raters

| Metric | Value |
| --- | --- |
| Matched rows | 154 |
| Unanimous rows | 133 |
| Unanimous agreement | 0.8636 |
| Fleiss' kappa | 0.8144 |

Both models label `yes` more often than the final human benchmark: Claude is 13-to-1 and GPT is 14-to-2 on the two error directions. The remaining disagreement is a systematic recall-versus-precision offset, not scattered noise.

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
| `data/provenance/round-01/analysis/agreement.json` | yes | Canonical machine-readable Round 1 report against the 154-row benchmark, including label counts, pairwise metrics, confusion matrices, three-rater agreement, disagreements, scope, and hashes |
| `data/provenance/round-01/analysis/historical-158/` | yes | Preserved original 158-row agreement report and disagreement table |
| `data/provenance/round-01/` | yes | Complete immutable Round 1 snapshot: prompt, input, benchmark, model outputs, manifest, and analysis |
| [`data/benchmark/v2-adjudicated.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/benchmark/v2-adjudicated.csv) | yes | The 154-row final benchmark: the human export's own nine columns, with removed rows dropped and adjudicated labels applied |

Committed tables and immutable release provenance live under `data/`; raw and intermediate artifacts stay under `results/` on the Seagate drive. See the [Data catalogue](results.md) for every file in the project.

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

## Rebuilding the original adjudication

```bash
./scripts/uv-seagate run python scripts/interrater_agreement.py
```

This legacy command rebuilds the original 158-row adjudication inputs and committed review tables. The canonical 154-row Round 1 agreement report is the preserved generated artifact at `data/provenance/round-01/analysis/agreement.json`; do not overwrite it with the legacy command. Future prompt rounds use `scripts/evaluate_llm_round.py` and their own numbered directory. Outputs are byte-identical across repeated runs on unchanged inputs. A validation failure prints to standard error, returns exit status 1, and writes nothing.

The code lives in `src/landuse_sentence_relevance/analysis/`, is covered by unit tests under `tests/unit/analysis/` and `tests/unit/test_interrater_script.py`, and is part of the mutation gate described in [QA](qa.md).
