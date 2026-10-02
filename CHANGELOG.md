# Changelog

This file records all notable changes to the project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is an immutable git tag. A release freezes the code and
the committed dataset in `data/`. Refer to [Versioning](docs/versioning.md) for
what a release guarantees and how the project builds the next release without
changes to the frozen one.

## [Unreleased]

### Changed

- The V3 geographic map now renders Natural Earth 110m land vectors in an
  equirectangular projection. This matches the cartography of the companion
  OSM polygon datasets. It replaces the pinned OpenStreetMap zoom-2 raster
  mosaic. The pinned, offline, checksum-verified snapshot is at
  `data/benchmark/v3/assets/ne-110m-land.geojson`.

### Added

- Deterministic V3 reserve candidates for the source-by-label quota
  shortfalls. The validation keeps the reserve IDs, cells, and sources
  globally sound.
- Finalized V3 benchmark at `data/benchmark/v3/final/v3-final.csv`. Its human
  reference, independent GPT review, disagreement queue, and resolved
  decisions are in `data/benchmark/v3/`.

### Changed

- V3 progress now derives the candidate selection, the quota accounting, and
  the review rows from one validated session snapshot. This prevents repeated
  scans during UI refreshes.
- Fresh V3 seed selection uses bounded top-K ranking. The V3 documentation
  describes the reserve behavior and the Seagate-backed annotation command.

## [2.0.0] - 2026-09-14

The V2 milestone: a contextual annotation pipeline, a three-rater agreement
study, and a benchmark of 154 rows that the project adjudicated by hand.

### The dataset

- **`data/benchmark/v2-adjudicated.csv` - 154 rows, 80 `yes` / 74 `no`.**
  This is the benchmark to use. The project built it from the human V2 export.
  It resolved every three-rater disagreement by hand: 4 sentences removed,
  9 labels overturned. It keeps the nine columns and the row order of the
  export. It has 100 Wikipedia rows (49 yes / 51 no) and 54 website rows
  (31 yes / 23 no).
- `data/interrater/adjudication.csv` has the 31 disagreements with their
  hand-assigned `final_label`. It is the only hand-edited file in `data/`. It
  is the input that produces the benchmark.
- `data/interrater/disagreements.csv` has the same 31 rows as the project
  generated them, before adjudication.
- `data/provenance/round-01/` has the immutable prompt, inputs, model outputs,
  manifest, and agreement reports for the Round 1 study.

### Added

- Contextual V2 annotation pipeline: English-only sentences from inside the
  source text blocks, H3 resolution-3 stratification, deterministic maximin
  spacing across 100 distinct cells, and resumable candidate pools.
- Interrater agreement analysis (`src/landuse_sentence_relevance/analysis/`):
  pairwise agreement with Cohen's kappa and confusion matrices, three-rater
  unanimity with Fleiss' kappa, per-rater label counts, and a disagreement
  table for review. The analysis matches rows by exact sentence identity, never
  by position. The run stops on any missing, duplicate, misaligned, or invalid
  label.
- Adjudication as a validated input: the verdicts are `yes`, `no`, or
  `remove`. The run does not continue unless every disagreement has a verdict
  and no unanimous sentence has a verdict.
- Reproducible, numbered LLM evaluation rounds with blinded inputs, recorded
  prompts, and a manifest for each round. Nobody overwrites an existing round.
- Two machine-labeled reference runs (GPT 5.6 Extra High, Claude Opus 5 Extra).
  Their shared prompt and their input and output hashes are in the
  documentation.
- Architecture decision records in `docs/adr/`.

### Changed

- Each kind of data has one location: committed tables in `data/`, generated
  and raw artifacts in `results/` on the project drive, and runtime state in
  `state/`. Only `data/` is in Git.
- The documentation is a single MkDocs site. Its data catalogue names every
  file in the project and the origin of the file.
- Mutation testing runs one worker for each core instead of one worker in
  total. The verdict is the same. The mutants of the analysis package take
  seconds instead of minutes.

### Quality

The release passed the full gauntlet: locked dependencies, Ruff format and
lint, TY, unit and browser acceptance tests above 95% coverage, CRAP below 6,
mutation testing with zero surviving mutants, and a pinned-revision streaming
smoke check.

## [1.0.1] - 2026-08-27

Maintenance release of the V1 annotation UI.

## [1.0.0] - 2026-08-27

First release: the streamed, geographically stratified annotation UI and the
V1 human golden set of 100 rows.

[2.0.0]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v2.0.0
[1.0.1]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v1.0.1
[1.0.0]: https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/releases/tag/v1.0.0
