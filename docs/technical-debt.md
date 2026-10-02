# Known technical debt

These items are deliberate limitations. They are not silent failures.

| Item | Why it exists | Cleanup path |
| --- | --- | --- |
| The Hugging Face streaming smoke test needs the network and the pinned upstream files. | The project must validate the real streaming boundary. It does not store source data locally. | Keep the revisions pinned. If offline CI becomes necessary, add a maintained local protocol fixture. |
| CI validates Docker. The development Mac does not. | The local workflow does not use Docker on purpose. It keeps the storage on the Seagate drive. | When the image or the runtime changes, run the CI gate again or use a disposable Linux runner. |
| The acceptance suite emits the existing `httpx` deprecation warning from Starlette. | The current test client stays compatible. The warning does not change the behavior. | When the supported Starlette and httpx combination is updated, review the test-client dependency. |
| The architecture check is a static AST analysis. | If the gate imports the application, it loads optional models and side effects. | When you introduce a dynamic internal dependency, add an explicit rule and a test. |
