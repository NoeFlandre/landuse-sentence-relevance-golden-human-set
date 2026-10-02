# ADR 0001: One deterministic quality gauntlet

Status: accepted

## Decision

`scripts/gauntlet.py` is the single executable quality gate. It runs these checks in a fixed order: locked dependency checks, Ruff, TY, unit tests, generated-input property tests, Gherkin/Playwright acceptance tests, coverage, architecture checks, CRAP, mutation testing, packaging, documentation, smoke checks, and diff hygiene.

The local Seagate command skips only the Hugging Face network access and Docker. CI runs the same gate without the skip flags. The remote streaming check and the container check therefore block a merge. The development Mac does not run Docker.

## Consequences

One command gives the contributors and CI the same deterministic verdict. The local results cannot prove the behavior of the external service or the container. For these two boundaries, the CI run is authoritative.
