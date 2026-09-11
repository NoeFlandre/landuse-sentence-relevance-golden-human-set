# QA procedure

## Test behavior in Gherkin

Gherkin uses short `Given`, `When`, and `Then` statements to describe behavior a person can observe. This project drives the text UI with a real Playwright browser, so the feature checks the complete request, save, and next-candidate flow rather than only calling Python functions.

The feature is at `tests/acceptance/features/annotation.feature`; its step definitions start the local app and drive the browser.

## Deterministic gauntlet

Run the local gate from the Seagate checkout with:

```bash
./scripts/uv-seagate run python scripts/gauntlet.py --skip-network --skip-docker
```

CI runs the same command without skip flags. The local command keeps UV, model, test, build, documentation, and mutation state under the Seagate `state/` directory. Docker and Hugging Face network access are intentionally CI-only on the development Mac.

The fixed order is:

1. record a pre-change baseline;
2. verify the lockfile, Ruff formatting, Ruff lint, and TY;
3. run unit tests, Hypothesis property tests, and Gherkin/Playwright acceptance tests while accumulating coverage;
4. enforce at least 95% line coverage and 90% branch coverage;
5. check dependency boundaries and circular imports;
6. enforce CRAP below 6;
7. run mutation testing and reject every survivor, timeout, or untested mutant;
8. build the wheel and source distribution and build MkDocs in strict mode;
9. run the pinned Hugging Face streaming smoke test and Docker build in CI; and
10. run the Git diff whitespace check, followed by human diff review.

The baseline is a process checkpoint before editing. The executable gate starts at the lock check because a dirty working tree is expected while developing a change.

Run focused stages directly when iterating:

```bash
./scripts/uv-seagate run pytest tests/unit -q
./scripts/uv-seagate run pytest tests/property -q
./scripts/uv-seagate run pytest tests/acceptance -m acceptance -q
./scripts/uv-seagate run python scripts/check_architecture.py
```

The quality gate fixes `PYTHONHASHSEED=0`, uses pinned upstream revisions, and uses deterministic Hypothesis settings. Mutation testing is protected by a nonblocking lock at `PROJECT_STATE_ROOT/mutation.lock`, so concurrent runs fail fast instead of competing for HDD I/O.
