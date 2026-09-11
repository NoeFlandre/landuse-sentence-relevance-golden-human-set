# ADR 0001: One deterministic quality gauntlet

Status: accepted

## Decision

`scripts/gauntlet.py` is the single executable quality gate. It runs locked dependency checks, Ruff, TY, unit tests, generated-input property tests, Gherkin/Playwright acceptance tests, coverage, architecture checks, CRAP, mutation testing, packaging, documentation, smoke checks, and diff hygiene in a fixed order.

The local Seagate command skips only Hugging Face network access and Docker. CI runs the same gate without skip flags, so remote streaming and the container remain merge-blocking checks while the development Mac does not run Docker.

## Consequences

One command gives contributors and CI the same deterministic verdict. Local results cannot prove external-service or container behavior; the CI run is authoritative for those two boundaries.
