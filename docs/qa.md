# QA procedure

## Gherkin and the UI

Gherkin is a short, readable test language with `Given`, `When`, and `Then` sentences. It describes the behavior a user should see; it is not a replacement for Python tests.

This project uses a Gherkin feature with a real Playwright browser. It checks that:

```gherkin
Given the annotator has a current sentence
When the annotator chooses Yes or No
Then the label is saved and the next sentence is shown
```

The feature is at `tests/acceptance/features/annotation.feature`. Its step definitions start the local app and drive the browser, so the test covers the text UI rather than only calling Python functions.

## Deterministic gauntlet

Run the complete gate with:

```bash
./scripts/uv-seagate run python scripts/gauntlet.py --skip-docker
```

The local command keeps UV and test scratch data on the Seagate drive. CI runs the same gauntlet with ordinary `uv` commands on its own ephemeral runner.

It runs, in order:

1. locked dependency check, Ruff format, Ruff lint, and TY;
2. unit and browser acceptance tests with at least 95% coverage;
3. CRAP score checking with a strict limit below 6;
4. serialized mutation testing with zero survivors;
5. a pinned-revision streaming smoke check; and
6. a Docker build.

Mutation testing targets the deterministic domain, storage, validation, and website-selection modules with their focused unit tests. The full suite, streaming smoke test, and CI container check cover the model, remote, and application-composition boundaries.

The gauntlet fixes `PYTHONHASHSEED=0`, serializes mutmut with one worker, and uses pinned upstream revisions. For offline local checks, `--skip-network --skip-docker` skips only the remote and container steps.

Docker is intentionally CI-only on the development Mac.
