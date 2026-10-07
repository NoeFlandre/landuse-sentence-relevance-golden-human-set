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

## Crash recovery

Each annotation append flushes and syncs the file before returning. File replacement syncs a sibling temporary file before the rename, then syncs the containing directory. Newly created parent directory entries are also synced. A retry syncs the deepest existing directory entry before continuing, covering a crash between creating a directory and syncing its parent. Sync failures propagate to the caller; a failure after a rename can leave the new file visible even though its durability is uncertain. These guarantees rely on a filesystem that supports file and directory `fsync`.

Loading a session validates every newline-terminated record. A final fragment without a newline is recoverable only when it begins a JSON object and the parser reports an unfinished string or reaches the end while expecting more JSON. An incomplete UTF-8 sequence at the very end is recoverable only if the decoded prefix meets that same condition. Invalid syntax before the end, invalid UTF-8 elsewhere, incomplete Unicode escape sequences, and valid JSON with invalid annotation fields still raise an error. A complete corrupt record is never skipped.

Before recovering, the store writes and syncs an exact copy of the original file to a unique sibling named `.annotations.jsonl.<random>.recovery`, using the actual annotation filename. It syncs that directory entry before atomically restoring the validated prefix. The warning names both files and does not include sentence text. Recovery files remain on disk for manual inspection; the store never overwrites or removes them. If recovery fails before replacement, the original session stays unchanged. Recovery can be retried after a crash.

Valid final records without a newline remain valid. The next append inserts a newline separator. Appending also checks for a torn tail first, so restarting and continuing a recovered session cannot concatenate a new annotation onto damaged bytes. The existing single-writer session contract remains in effect; this does not add multi-process write coordination.
