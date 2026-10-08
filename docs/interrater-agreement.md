# Interrater agreement and adjudication

The human annotator, GPT, and Claude first labeled the same V2 combined export of 158 sentences. The final adjudicated human benchmark has 154 rows. The metrics below are the canonical Round 1 metrics against those 154 benchmark rows. The original report of 158 rows is in `data/provenance/round-01/analysis/historical-158/`.

## At a glance

| Pair | Agreement | Cohen's kappa |
| --- | --- | --- |
| GPT vs Claude | 92.2% (142/154) | 0.84 |
| GPT vs human | 89.6% (138/154) | 0.79 |
| Claude vs human | 90.9% (140/154) | 0.82 |

All three raters agree on 133/154 rows (86.4%). Fleiss' kappa is 0.81. A human adjudicated by hand the 31 disagreements in the original run of 158 rows. The human removed 4 of them. The result is the **final benchmark of 154 rows** that this page uses.

## Raters

| Rater | Source | Label column |
| --- | --- | --- |
| `human` | `data/provenance/round-01/benchmark.csv` | `label` |
| `gpt` | `data/provenance/round-01/outputs/gpt.csv` | `llm_label` |
| `claude` | `data/provenance/round-01/outputs/claude.csv` | `llm_label` |

The canonical report uses the final benchmark of 154 rows as its reference. The GPT file and the Claude file are the historical outputs of 158 rows. The human adjudication removed four of their rows. The canonical comparison excludes these rows. The original human labels of 158 rows are in `data/provenance/round-01/human.csv`.

The table lists the sources of the canonical 154-row report. Only `benchmark.csv` has 154 rows. The GPT and Claude files have 158 rows each, and the canonical report uses 154 of them. The script's defaults do not read these paths. They read `results/evaluations/round-01/`, which git ignores (`scripts/interrater_agreement.py:36-41`, `.gitignore:18`). For the 158-row rebuild, the human input is `data/provenance/round-01/human.csv`, not `benchmark.csv`. The [fresh-clone recipe](#fresh-clone) shows the flags.

## Method

### Matching

The analysis matches rows by **exact sentence identity**. It does not use the row position. It indexes each file as a mapping from its `sentence` value to its label. The row order is therefore not important. Two files can have the same order, but the analysis still matches them by content. In the canonical Round 1 report, the 154 sentences in `benchmark.csv` define the reference set, and the four model rows outside that set are excluded (`rows_used` is 154 for both models in `agreement.json`). The script does not exclude rows. A rater whose sentence set differs from the reference fails, so the script does not reproduce that report.

The analysis fails with an error and gives no partial result in these cases:

- A file does not have the `sentence` column or the label column of the rater.
- A row is truncated.
- A label is not `yes` or `no` after the analysis trims it and changes it to lowercase.
- A rater has no rows.
- A sentence occurs more than one time in one file.
- The sentence set of a rater is different from the sentence set of the reference rater, in either direction.

The analysis computes a metric only when every rater covers exactly the same sentence set, one time for each sentence. It then processes the matched sentences in sorted order. Every output is therefore deterministic.

### Metrics

- **Observed agreement** is the share of matched sentences that two raters labeled the same.
- **Expected agreement** is the chance agreement from the label marginals of the two raters. It is the sum, over the labels, of the product of the share of that label for each rater.
- **Cohen's kappa** is `(observed - expected) / (1 - expected)`. When the expected agreement is exactly 1, both raters used one label everywhere. The observed agreement is also 1. The analysis then reports a kappa of 1.
- **Confusion matrix** gives the counts. The labels of the first rater are the rows. The labels of the second rater are the columns.
- **Three-rater agreement** is the number and the share of the sentences where all three raters chose the same label.
- **Fleiss' kappa** is a chance correction across all three raters at the same time. For each sentence, it measures the share of agreeing ordered rater pairs. It averages that share over the sentences. Then it corrects the average with the agreement that the pooled marginals predict. If the corpus has one single label, the analysis reports a kappa of 1.

The analysis lists the pairs over the rater names in sorted order: `claude`-`gpt`, `claude`-`human`, `gpt`-`human`.

### Adjudication

Each of the 31 non-unanimous sentences has one hand-assigned verdict in `final_label`: `yes`, `no`, or `remove`. The final benchmark takes the unanimous label where the raters agreed. It takes the verdict where they did not agree. It drops every `remove`.

The analysis validates the adjudication as strictly as the agreement. Every disagreement must have a verdict. No unanimous sentence can have a verdict. No verdict can be different from the three tokens above. Any violation stops the run.

## Results

The project regenerated the canonical report over the final benchmark. It has **154 matched rows**. The reference scope has no missing label, no duplicate label, and no invalid label. Each model source file has 158 rows. The canonical report excludes 4 rows because the final human benchmark does not have them. The script rejects such rows instead (see [Matching](#matching)). The project keeps the original report of 158 rows in `analysis/historical-158/`.

The input SHA-256 values, as `agreement.json` records them:

| Rater | Source rows | Rows used | SHA-256 |
| --- | --- | --- | --- |
| human benchmark | 154 | 154 | `ac185e51835eb626c932744009873382aa80027d26e93615dbde28d23af1d5fd` |
| gpt | 158 | 154 | `0e328e3507c497f8e03b907928719908b3a60342fd3e380f8b24d57def0a7cec` |
| claude | 158 | 154 | `8ec64988527077932aa4ec70a34203415795c604d96d693b0a01f6db0fce84a2` |

The two model hashes are the same as the hashes in [LLM labeling](llm-labeling.md). This confirms that the analysis read the published outputs without changes.

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

The rows are the label of the first rater. The columns are the label of the second rater.

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

Compared with the final human benchmark, the two models still prefer `yes`. Claude assigns `yes` to 13 sentences that the benchmark labels `no`. It assigns `no` to 1 benchmark `yes`. GPT assigns `yes` to 14 benchmark `no` sentences. It assigns `no` to 2 benchmark `yes` sentences. The models disagree with each other on 12 of the original 158 rows. Six of these rows stay in the reference scope of 154 rows.

### Three raters

| Metric | Value |
| --- | --- |
| Matched rows | 154 |
| Unanimous rows | 133 |
| Unanimous agreement | 0.8636 |
| Fleiss' kappa | 0.8144 |

Both models label `yes` more often than the final human benchmark. For the two error directions, Claude is 13-to-1 and GPT is 14-to-2. The remaining disagreement is a systematic offset between recall and precision. It is not scattered noise.

### Adjudication outcome

| Verdict | Sentences |
| --- | --- |
| `no` | 20 |
| `yes` | 7 |
| `remove` | 4 |

The adjudication kept 27 sentences. For these verdicts, it upheld the original label of the human 18 times, of Claude 13 times, and of GPT 11 times. This is consistent with the `yes` bias of the models, because two thirds of the verdicts went to `no`.

The result is `data/benchmark/v2-adjudicated.csv`. It is the human export with 4 rows removed and 9 labels replaced. It keeps its own columns and row order.

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
| [`data/interrater/disagreements.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/interrater/disagreements.csv) | yes | One row for each non-unanimous sentence. Generated. |
| [`data/interrater/adjudication.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/interrater/adjudication.csv) | yes | The same rows and the hand-assigned `final_label`. It is the only hand-edited file here. |
| `data/provenance/round-01/analysis/agreement.json` | yes | The canonical machine-readable Round 1 report against the benchmark of 154 rows. It has the label counts, the pairwise metrics, the confusion matrices, the three-rater agreement, the disagreements, the scope, and the hashes. |
| `data/provenance/round-01/analysis/historical-158/` | yes | The preserved original agreement report and disagreement table of 158 rows. |
| `data/provenance/round-01/` | yes | The complete immutable Round 1 snapshot: prompt, input, benchmark, model outputs, manifest, and analysis. |
| [`data/benchmark/v2-adjudicated.csv`](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/data/benchmark/v2-adjudicated.csv) | yes | The final benchmark of 154 rows. It has the nine columns of the human export. The project dropped the removed rows and applied the adjudicated labels. |

The committed tables and the immutable release provenance are in `data/`. The raw and intermediate artifacts stay in `results/` on the Seagate drive. Refer to the [Data catalogue](results.md) for every file in the project.

### Review table columns

Both CSV files in `data/interrater/` have the same shape:

| Column | Meaning |
| --- | --- |
| `minority_rater` | The rater that the other two raters outvoted. It is empty if no label has a strict majority. |
| `minority_label` | The label that the outvoted rater chose. |
| `human`, `gpt`, `claude` | The three labels for that sentence, in that order. |
| `final_label` | The adjudicated verdict: `yes`, `no`, or `remove`. It is only in `adjudication.csv`. |
| `sentence` | The sentence. It is immediately after the labels. |
| `source`, `region`, `polygon_name`, `source_url` | The origin of the sentence. |

The rows are sorted by `minority_rater`, then `minority_label`, then the sentence. Each systematic pattern is then one block:

| Outvoted rater | Their label | Rows |
| --- | --- | --- |
| human | `no` | 17 |
| human | `yes` | 2 |
| gpt | `yes` | 4 |
| gpt | `no` | 2 |
| claude | `yes` | 4 |
| claude | `no` | 2 |

## Rebuild the original adjudication

### Fresh clone

From the repository root, after `uv sync`, rebuild the original 158-row analysis with plain UV. No Seagate drive is needed. The command reads the committed inputs and writes to `results/interrater-rebuild/`, which git ignores:

```bash
uv run python scripts/interrater_agreement.py \
  --human data/provenance/round-01/human.csv \
  --gpt data/provenance/round-01/outputs/gpt.csv \
  --claude data/provenance/round-01/outputs/claude.csv \
  --adjudication data/interrater/adjudication.csv \
  --output-directory results/interrater-rebuild/round-01 \
  --review-csv results/interrater-rebuild/round-01/disagreements.csv \
  --benchmark-csv results/interrater-rebuild/round-01/v2-adjudicated.csv
```

No file under `data/` changes. The command does not regenerate the canonical 154-row report, which stays at `data/provenance/round-01/analysis/agreement.json`.

### Maintainer command on the Seagate drive

**Warning:** The defaults never write to `data/provenance/`. They do overwrite committed files outside it: `data/interrater/disagreements.csv` and `data/benchmark/v2-adjudicated.csv`. They also overwrite `results/evaluations/round-01/analysis/agreement.json` in the working mirror of the Round 1 snapshot, so back that file up first. Never point an output flag at `data/provenance/`.

```bash
./scripts/uv-seagate run python scripts/interrater_agreement.py
```

This command reads the 158-row inputs from `results/evaluations/round-01/` and recomputes the 158-row report, the review table and the benchmark. It only reads `data/interrater/adjudication.csv`, which is hand-edited and is not rebuilt. The wrapper exits with status 2 when the Seagate drive is not mounted. A missing input stops with a traceback and exit status 1, and writes nothing. Future prompt rounds use `scripts/evaluate_llm_round.py` and their own numbered directory. The outputs are byte-identical across repeated runs on unchanged inputs. A validation failure prints to standard error and returns exit status 1. It writes nothing.

The code is in `src/landuse_sentence_relevance/analysis/`. Unit tests in `tests/unit/analysis/` and `tests/unit/test_interrater_script.py` cover it, including `test_fresh_clone_recipe_reads_committed_inputs_and_writes_outside_data`. It is part of the mutation gate in [QA](qa.md).
