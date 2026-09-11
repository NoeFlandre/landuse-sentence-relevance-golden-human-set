# LLM evaluation rounds

Prompt tuning is recorded as independent, numbered rounds. A new prompt, model run, or input creates a new round; existing rounds are never overwritten.

## Round layout

Each round on the Seagate drive contains:

```text
results/evaluations/round-02/
├── manifest.json       # hashes, models, timestamps, and status
├── prompt.md           # exact prompt sent to both models
├── benchmark.csv       # labeled reference; never upload this to a model
├── input.csv           # same rows with the human label removed
├── outputs/
│   ├── gpt.csv
│   └── claude.csv
└── analysis/
    ├── agreement.json
    └── disagreements.csv
```

Round 1 is the historical 158-row evaluation. Its final adjudicated reference is the 154-row `data/benchmark/v2-adjudicated.csv`. Later rounds use that 154-row benchmark directly, so each model is evaluated against the final human ground truth.

## Start a new round

Save the exact prompt you will use to a Seagate-only file, then run:

```bash
./scripts/uv-seagate run python scripts/prepare_llm_round.py \
  --round-id round-02 \
  --prompt-file results/evaluations/prompts/round-02.md
```

The command creates a blinded `input.csv`, copies the labeled reference and prompt, records hashes, and refuses to overwrite an existing round. Send only `input.csv` to GPT and Claude. Save their unchanged CSV responses as `outputs/gpt.csv` and `outputs/claude.csv`; each must preserve the 154 rows and add only lowercase `llm_label` values.

Evaluate both outputs with:

```bash
./scripts/uv-seagate run python scripts/evaluate_llm_round.py \
  --round-directory results/evaluations/round-02 \
  --gpt-model "GPT 5.6 Extra High" \
  --claude-model "Claude Opus 5 Extra" \
  --gpt-run-at "2026-09-11T10:00:00+02:00" \
  --claude-run-at "2026-09-11T10:01:00+02:00"
```

The report includes human-vs-model and model-vs-model agreement, kappa, confusion matrices, and every disagreement. The command also records output hashes in `manifest.json`. Inspect the report and the disagreement CSV before deciding whether the prompt is satisfactory; if it is not, create `round-03` with a new prompt. Do not change the benchmark to improve agreement.

All round artifacts and caches stay on the Seagate drive. The workflow streams source data and does not download local model weights for chat-based evaluations.
