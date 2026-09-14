# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is an immutable git tag. A release freezes both the code
and the committed dataset under `data/`; see [Versioning](docs/versioning.md)
for what a release guarantees and how the next one is built without touching it.

## [2.0.0] - 2026-09-14

The V2 milestone: a contextual annotation pipeline, a three-rater agreement
study, and a hand-adjudicated 154-row benchmark.

### The dataset

- **`data/benchmark/v2-adjudicated.csv` — 154 rows, 80 `yes` / 74 `no`.**
  The benchmark to use. Built from the human V2 export by resolving every
  three-rater disagreement by hand: 4 sentences removed, 9 labels overturned.
  Keeps the export's nine columns and row order. 100 Wikipedia rows
  (49 yes / 51 no) and 54 website rows (31 yes / 23 no).
- `data/interrater/adjudication.csv` — the 31 disagreements with their
  hand-assigned `final_label`. The only hand-edited file in `data/`, and the
  input that produces the benchmark.
- `data/interrater/disagreements.csv` — the same 31 rows as generated, before
  adjudication.
- `data/provenance/round-01/` — the immutable prompt, inputs, model outputs,
  manifest, and agreement reports for the Round 1 study.

### Added

- Contextual V2 annotation pipeline: English-only sentences taken from inside
  source text blocks, H3 resolution-3 stratification, deterministic maximin
  spacing across 100 distinct cells, and resumable candidate pools.
- Interrater agreement analysis (`src/landuse_sentence_relevance/analysis/`):
  pairwise agreement with Cohen's kappa and confusion matrices, three-rater
  unanimity with Fleiss' kappa, per-rater label counts, and a reviewable
  disagreement table. Rows are matched by exact sentence identity, never by
  position, and the run aborts on any missing, duplicate, misaligned, or
  invalid label.
- Adjudication as a validated input: verdicts of `yes`, `no`, or `remove`, with
  the run refusing to proceed unless every disagreement carries a verdict and no
  unanimous sentence does.
- Reproducible, numbered LLM evaluation rounds with blinded inputs, recorded
  prompts, and per-round manifests; existing rounds are never overwritten.
- Two machine-labeled reference runs (GPT 5.6 Extra High, Claude Opus 5 Extra)
  with their shared prompt and input/output hashes recorded in the docs.
- Architecture decision records under `docs/adr/`.

### Changed

- Data has one home per kind: committed tables under `data/`, generated and raw
  artifacts under `results/` on the project drive, runtime state under `state/`.
  Only `data/` is in Git.
- Documentation is a single MkDocs site with a data catalogue that names every
  file in the project and where it comes from.
- Mutation testing runs one worker per core instead of one in total; the
  verdict is unchanged and the analysis package's mutants drop from minutes to
  seconds.

### Quality

Released green on the full gauntlet: locked dependencies, Ruff format and lint,
TY, unit and browser acceptance tests above 95% coverage, CRAP below 6, mutation
testing with zero surviving mutants, and a pinned-revision streaming smoke check.

## [1.0.1] - 2026-08-27

Maintenance release of the V1 annotation UI.

## [1.0.0] - 2026-08-27

First release: the streamed, geographically stratified annotation UI and the
100-row V1 human golden set.

[2.0.0]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v2.0.0
[1.0.1]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v1.0.1
[1.0.0]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v1.0.0
