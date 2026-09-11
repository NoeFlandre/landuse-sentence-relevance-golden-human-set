# ADR 0002: Keep domain logic independent

Status: accepted

## Decision

The package uses explicit layers: pure `domain` logic; `analysis` over domain; `sources` over domain and observability; `storage` over domain, configuration, and observability; `workflow` over domain and storage; and `web` as the application boundary. `bootstrap` is the composition root. The architecture checker rejects forbidden layer imports and circular imports.

## Consequences

Sampling, validation, and agreement logic stay testable without network, filesystem, UI, or model state. New cross-layer dependencies must be introduced deliberately by changing the checked boundary rather than through hidden coupling.
