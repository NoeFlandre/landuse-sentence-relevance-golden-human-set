# LLM evaluation rounds

The project records prompt tuning as independent, numbered rounds. A new prompt, a new model run, or a new input creates a new round. Nobody overwrites an existing round.

## Round layout

Each round on the Seagate drive contains these items:

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

Round 1 is the historical evaluation of 158 rows. Its final adjudicated reference is the 154-row `data/benchmark/v2-adjudicated.csv`. The later rounds use that 154-row benchmark directly. Each model is then evaluated against the final human ground truth.

## Start a new round

Save the exact prompt that you will use in a file on the Seagate drive only. Then run:

```bash
./scripts/uv-seagate run python scripts/prepare_llm_round.py \
  --round-id round-02 \
  --prompt-file results/evaluations/prompts/round-02.md
```

The command creates a blinded `input.csv`. It copies the labeled reference and the prompt. It records the hashes. It does not overwrite an existing round.

Send only `input.csv` to GPT and Claude. Save their unchanged CSV responses as `outputs/gpt.csv` and `outputs/claude.csv`. Each file must keep the 154 rows. Each file must add only lowercase `llm_label` values.

Evaluate the two outputs:

```bash
./scripts/uv-seagate run python scripts/evaluate_llm_round.py \
  --round-directory results/evaluations/round-02 \
  --gpt-model "GPT 5.6 Extra High" \
  --claude-model "Claude Opus 5 Extra" \
  --gpt-run-at "2026-09-11T10:00:00+02:00" \
  --claude-run-at "2026-09-11T10:01:00+02:00"
```

The report includes the human-vs-model agreement and the model-vs-model agreement. It also includes the kappa, the confusion matrices, and every disagreement. The command also records the output hashes in `manifest.json`.

Examine the report and the disagreement CSV. Then decide if the prompt is satisfactory. If it is not, create `round-03` with a new prompt.

**Warning:** Do not change the benchmark to improve the agreement.

All round artifacts and caches stay on the Seagate drive. The workflow streams the source data. It does not download local model weights for the chat-based evaluations.
