# ADR 0002: Keep domain logic independent

Status: accepted

## Decision

The package uses explicit layers:

- `domain` has pure logic.
- `analysis` is over `domain`.
- `sources` is over `domain` and observability.
- `storage` is over `domain`, configuration, and observability.
- `workflow` is over `domain` and `storage`.
- `web` is the application boundary.

`bootstrap` is the composition root. The architecture checker rejects the forbidden layer imports and the circular imports.

## Consequences

The sampling, validation, and agreement logic stay testable without the network, the filesystem, the UI, or the model state. To add a new dependency between layers, change the checked boundary on purpose. Do not use hidden coupling.
