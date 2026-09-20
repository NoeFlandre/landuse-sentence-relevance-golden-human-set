# Versioning

The project is released as immutable versions so that a published benchmark
never changes underneath anyone using it.

## What a release freezes

A release is an annotated git tag, `vMAJOR.MINOR.PATCH`. Tagging freezes:

- the committed dataset under `data/`, above all the benchmark file;
- the code that regenerates that dataset from its inputs;
- the documentation describing how the dataset was built; and
- the quality gate the release passed.

Everything a release needs to be reproduced is in the tag. Drive-only trees
(`results/`, `state/`) are working material and are deliberately not part of it.
Small immutable evaluation snapshots required for release provenance live under
the tracked `data/provenance/` tree.

## Current release

**v2.0.0** — the benchmark is `data/benchmark/v2-adjudicated.csv`: 154 rows,
80 `yes` / 74 `no`, adjudicated from the three-rater V2 study. See the
[Changelog](https://github.com/NoeFlandre/landuse-sentence-relevance-golden-human-set/blob/main/CHANGELOG.md)
for the full contents and the [Data catalogue](results.md) for every file.

The completed V3 annotation round is a separate final artifact at
`data/benchmark/v3/final/v3-final.csv`. It has its own reference, independent
review, and resolution trail under `data/benchmark/v3/` and does not mutate
the tagged V2 release.

## Rules

1. **A tagged benchmark file is never edited.** Not to fix a label, not to add
   a row. `data/benchmark/v2-adjudicated.csv` keeps exactly the bytes it had at
   `v2.0.0`, forever.
2. **A new version adds files; it does not replace them.** The next benchmark
   arrives in its own versioned directory, such as `data/benchmark/v3/`, with
   the final artifact under `final/` and its review trail under
   `independent-review/`. Both versions remain readable at `main`, and the
   docs say which one is current.
3. **Version numbers follow the dataset, not just the code.** A new or changed
   set of benchmark rows is a major bump, because anything scored against the
   old rows is no longer comparable. Added tooling or documentation with an
   unchanged benchmark is a minor bump; a fix that leaves the benchmark
   byte-identical is a patch.
4. **Every release is cut from a green gauntlet.** See [QA](qa.md).
5. **Evaluation rounds are independent of releases.** Prompt-tuning rounds are
   numbered under `results/evaluations/round-NN/` and never overwritten; see
   [LLM evaluation rounds](llm-evaluation-rounds.md). A round scored against a
   released benchmark stays valid for as long as that benchmark exists.

## Starting the next version

Work continues on `main`. Because rule 1 and rule 2 keep the released files
untouched, no branch is needed to protect v2.0.0 — the tag already does that.
Recovering the released state at any time:

```bash
git checkout v2.0.0
```

When the next benchmark is ready, add it as a new file, update the docs to name
it as current, record the change in the changelog, and cut the next tag.
