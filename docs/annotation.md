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
| Geography | 25 shared H3 resolution-3 cells |
| Cell/source quota | 2 Wikipedia and 2 website rows per cell |

Both source halves use the same cells. The annotator may label more than 100 candidates; deterministic selection chooses the first quota-satisfying subset by candidate ID. Once the subset satisfies every constraint, it is uploaded publicly to [the project dataset](https://huggingface.co/datasets/NoeFlandre/landuse-sentence-relevance-golden-human-set) without an extra confirmation step.

The local session contains labeled rows only at `state/annotations.jsonl`, which is ignored by Git. Raw streamed rows are not written locally.
