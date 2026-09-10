# Annotation contract

The UI shows one sentence and minimal provenance: source, place, region, coordinates, H3 cell, source link, and progress. The annotator chooses **Yes** or **No**.

Relevant means that the sentence describes what can be observed or geographically characterized at the place: land use or land cover, soil or surface, vegetation, ecosystems, terrain, geomorphology, visible buildings or infrastructure, or the place's physical geographic setting, shape, position, or extent.

## Final dataset

The final public dataset is accepted only when it has:

| Constraint | Requirement |
| --- | --- |
| Rows | 100 unique candidates |
| Source split | 50 Wikipedia, 50 website |
| Label split | 50 Yes, 50 No |
| Geography | 100 distinct H3 resolution-3 cells |
| Source geography | 50 Wikipedia cells and 50 website cells; no cell is shared |
| Geographic spread | Selected H3 cell centers are at least 500 km apart |

The V2 UI candidate pool contains 512 candidates: 256 from each source, one sentence per H3 cell. It prefers sentences inside each source text block and uses a first-sentence fallback only when no later accepted sentence exists. The annotator may label more than 100 candidates; deterministic selection chooses one candidate per cell and the first quota-satisfying subset. The candidate pool, JSONL session, and runtime cache make quitting resumable. Once the subset satisfies every constraint, it is uploaded publicly to the `v2` split of [the project dataset](https://huggingface.co/datasets/NoeFlandre/landuse-sentence-relevance-golden-human-set) without an extra confirmation step, and the runtime cache is deleted.

The active V2 Wikipedia session is `results/annotations/sessions/v2-wikipedia.jsonl`; candidate progress is `results/candidates/v2/progress.json`; and the reusable V2 pool is `results/candidates/v2/pool.json`. The failed pre-optimization checkpoint is preserved at `results/candidates/v2/archive/failures/20260829.json`; the complete artifact inventory is in [Results](results.md). V1 files remain separate. Raw streamed rows are not written locally.
