# Annotation contract

The UI shows one sentence and the minimal provenance: source, place, region, coordinates, H3 cell, source link, and progress. The annotator chooses **Yes** or **No**.

A sentence is relevant when it describes what a person can observe or characterize geographically at the place. The topics are: land use or land cover, soil or surface, vegetation, ecosystems, terrain, geomorphology, visible buildings or infrastructure, and the physical geographic setting, shape, position, or extent of the place.

## Final dataset

The project accepts the final public dataset only when it has these properties:

| Constraint | Requirement |
| --- | --- |
| Rows | 100 unique candidates |
| Source split | 50 Wikipedia, 50 website |
| Label split | 50 Yes, 50 No |
| Geography | 100 distinct H3 resolution-3 cells |
| Source geography | 50 Wikipedia cells and 50 website cells; no cell is shared |
| Geographic spread | The centers of the selected H3 cells are at least 500 km apart |

The V2 UI candidate pool has 512 candidates: 256 from each source, with one sentence for each H3 cell. The pool prefers the sentences inside each source text block. It uses a first-sentence fallback only when no later accepted sentence exists.

The annotator can label more than 100 candidates. The deterministic selection chooses one candidate for each cell. It also chooses the first subset that satisfies the quotas. The candidate pool, the JSONL session, and the runtime cache let the annotator quit and resume.

When the subset satisfies all constraints, the app uploads it publicly to the `v2` split of [the project dataset](https://huggingface.co/datasets/NoeFlandre/landuse-sentence-relevance-golden-human-set). The app does not ask for an extra confirmation. Then it deletes the runtime cache.

These are the V2 paths:

- The active V2 Wikipedia session is `results/annotations/sessions/v2-wikipedia.jsonl`.
- The candidate progress is `results/candidates/v2/progress.json`.
- The reusable V2 pool is `results/candidates/v2/pool.json`.
- The failed pre-optimization checkpoint is `results/candidates/v2/archive/failures/20260829.json`. The project keeps it.

The complete artifact inventory is in [Results](results.md). The V1 files stay separate. The app does not write raw streamed rows to the local disk.
