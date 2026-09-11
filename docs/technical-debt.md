# Known technical debt

These are deliberate limitations, not silent failures.

| Item | Why it exists | Cleanup path |
| --- | --- | --- |
| Hugging Face streaming smoke depends on network and pinned upstream files. | The project must validate the real streaming boundary without storing source data locally. | Keep revisions pinned; add a maintained local protocol fixture if offline CI becomes a requirement. |
| Docker is validated in CI, not on the development Mac. | The local workflow explicitly avoids Docker and keeps storage on the Seagate drive. | Re-run the CI gate or a disposable Linux runner when the image/runtime changes. |
| The acceptance suite emits Starlette's existing `httpx` deprecation warning. | The current test client remains compatible and the warning does not alter behavior. | Revisit the test-client dependency when the supported Starlette/httpx combination is updated. |
| The architecture check is static AST analysis. | Importing the application during a gate would load optional models and side effects. | Add an explicit rule and test whenever a dynamic internal dependency is introduced. |
