# Versioning

The project releases immutable versions. A published benchmark does not change for the people who use it.

## What a release freezes

A release is an annotated git tag, `vMAJOR.MINOR.PATCH`. The tag freezes these items:

- the committed dataset in `data/`, above all the benchmark file;
- the code that regenerates the dataset from its inputs;
- the documentation that describes how the project built the dataset; and
- the quality gate that the release passed.

The tag contains everything that you need to reproduce the release. The drive-only trees (`results/`, `state/`) are working material. They are not part of the tag on purpose. The small immutable evaluation snapshots that the release provenance needs are in the tracked `data/provenance/` tree.

## Current release

**v2.0.0**. The benchmark is `data/benchmark/v2-adjudicated.csv`. It has 154 rows: 80 `yes` and 74 `no`. The project adjudicated it from the three-rater V2 study. Refer to the [Changelog](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/CHANGELOG.md) for the full contents. Refer to the [Data catalogue](results.md) for every file.

The completed V3 annotation round is a separate final artifact at `data/benchmark/v3/final/v3-final.csv`. It has its own reference, independent review, and resolution trail in `data/benchmark/v3/`. It does not change the tagged V2 release.

## Rules

1. **Nobody edits a tagged benchmark file.** Do not edit it to correct a label. Do not edit it to add a row. The file `data/benchmark/v2-adjudicated.csv` keeps the exact bytes that it had at `v2.0.0`, forever.
2. **A new version adds files. It does not replace files.** The next benchmark goes in its own versioned directory, for example `data/benchmark/v3/`. The final artifact goes in `final/`. The review trail goes in `independent-review/`. Both versions stay readable at `main`. The documentation says which version is current.
3. **The version numbers follow the dataset, not only the code.** A new or changed set of benchmark rows is a major bump. Any score against the old rows is then not comparable. New tooling or new documentation with an unchanged benchmark is a minor bump. A fix that leaves the benchmark byte-identical is a patch.
4. **Cut every release from a green gauntlet.** Refer to [QA](qa.md).
5. **Evaluation rounds are independent of releases.** The prompt-tuning rounds are numbered in `results/evaluations/round-NN/`. Nobody overwrites them. Refer to [LLM evaluation rounds](llm-evaluation-rounds.md). A round that is scored against a released benchmark stays valid while that benchmark exists.

## Start the next version

Work continues on `main`. Rules 1 and 2 keep the released files unchanged. You do not need a branch to protect v2.0.0. The tag protects it.

To recover the released state at any time, run:

```bash
git checkout v2.0.0
```

When the next benchmark is ready, do these steps:

1. Add the benchmark as a new file.
2. Update the documentation to name the new file as current.
3. Record the change in the changelog.
4. Cut the next tag.
