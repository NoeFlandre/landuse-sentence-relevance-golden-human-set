# QA procedure

## Test behavior in Gherkin

Gherkin uses short `Given`, `When`, and `Then` statements. These statements describe behavior that a person can observe. This project drives the text UI with a real Playwright browser. The feature checks the complete flow of request, save, and next candidate. It does not only call Python functions.

The feature is at `tests/acceptance/features/annotation.feature`. Its step definitions start the local app and drive the browser.

## Deterministic gauntlet

Run the local gate from the Seagate checkout:

```bash
./scripts/uv-seagate run python scripts/gauntlet.py --skip-network --skip-docker
```

CI runs the same command without the skip flags. The local command keeps the UV, model, test, build, documentation, and mutation state in the Seagate `state/` directory. Only CI uses Docker and Hugging Face network access. This is intentional for the development Mac.

The gauntlet runs these steps in a fixed order:

1. Record a pre-change baseline.
2. Verify the lockfile, Ruff formatting, Ruff lint, and TY.
3. Run the unit tests, the Hypothesis property tests, and the Gherkin/Playwright acceptance tests. Accumulate the coverage.
4. Enforce at least 95% line coverage and 90% branch coverage.
5. Check the dependency boundaries and the circular imports.
6. Enforce CRAP below 6.
7. Run mutation testing. Reject every survivor, timeout, and untested mutant.
8. Build the wheel and the source distribution. Build MkDocs in strict mode.
9. In CI, run the pinned Hugging Face streaming smoke test and the Docker build.
10. Run the Git diff whitespace check. Then do a human review of the diff.

The baseline is a process checkpoint before you edit. The executable gate starts at the lock check. A dirty working tree is normal while you develop a change.

To iterate, run the focused stages directly:

```bash
./scripts/uv-seagate run pytest tests/unit -q
./scripts/uv-seagate run pytest tests/property -q
./scripts/uv-seagate run pytest tests/acceptance -m acceptance -q
./scripts/uv-seagate run python scripts/check_architecture.py
```

The quality gate sets `PYTHONHASHSEED=0`. It uses pinned upstream revisions and deterministic Hypothesis settings. A nonblocking lock at `PROJECT_STATE_ROOT/mutation.lock` protects mutation testing. A concurrent run fails at once. It does not compete for HDD I/O.
