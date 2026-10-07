# Land-use sentence relevance golden human set

This project builds a golden set of 100 sentences. A human annotator labels the sentences in a small local UI.

The V2 app streams two public Hugging Face datasets. When possible, it keeps the English sentences from inside the source text blocks. It shows candidates from unique cells across the world. It selects the candidates with deterministic H3 maximin spacing. One human annotator assigns the relevance label Yes or No.

The V2 runtime benchmark is `data/benchmark/v2-adjudicated.csv`. It is the human V2 set after the adjudication of its three-rater disagreements. The finalized V3 annotation benchmark is `data/benchmark/v3/final/v3-final.csv`. A separate page describes its review and resolution trail. The [Data catalogue](results.md) lists every data file.

The project also releases the V3 benchmark as 85 language-aligned CSV files. Refer to the [V3 multilingual benchmark](v3-translations.md). It gives the translation method, the file layout, the integrity contract, and the geographic coverage map.

For the project terms, refer to the [Glossary](glossary.md).

## Run locally

```bash
./scripts/uv-seagate sync --extra models
./scripts/uv-seagate run landuse-annotate
```

Open <http://127.0.0.1:8000>. The app binds to `127.0.0.1:8000` by default, so other computers cannot connect. Set `ANNOTATION_HOST` and `ANNOTATION_PORT` to change the address. The app streams the source datasets. The V2 candidate bank and the annotation logs are in `results/`. The auth files, the virtual environment, the UV cache, the temporary files, and the model cache are in `state/`. By default, both trees are on `/Volumes/Seagate M3/projects/landuse-sentence-relevance-golden-human-set`.

Use `scripts/uv-seagate` for local UV commands. It does not use the internal storage of the Mac. It stops when the Seagate drive is not mounted.

If necessary, log in one time. The login is stored separately from the disposable model and data files.

```bash
./scripts/uv-seagate run hf auth login
```

The terminal shows the startup stages, the sparse stream checkpoints, the annotation counts, and the final upload and cleanup. It does not print sentence text or raw rows. On the next start, the app moves an existing login from the old application cache automatically.

After the public final upload succeeds, the app deletes that exact cache automatically.

The final upload starts automatically when all quotas are complete. Refer to [annotation](annotation.md) for the contract. Refer to [QA](qa.md) for the deterministic checks.

The project also has a design for a follow-up set of 300 rows. The oversized candidate pool, the geographic preflight, and the deterministic annotation seed are available. The annotation is paused. The sources, quotas, storage, and runbook are in [V3 workflow](v3.md).


## UI security and container access

The UI has no authentication. Keep it on loopback or behind access controls for a trusted network. Any client that can reach it can read candidate text and change annotations. In V2, completing the quotas can start a public Hugging Face upload with the operator's saved login. V3 does not publish automatically.

The app accepts only the exact Host names `127.0.0.1`, `localhost`, and `[::1]` by default. An `ANNOTATION_HOST` that is a loopback IP address (for example `127.0.0.2`) is also accepted as a Host name. It checks every state-changing request against the full request origin, including scheme and port. Browser forms send an Origin header. If Origin is absent, the app requires a matching Referer. Missing, null, malformed, duplicate, or cross-origin values are rejected before annotation or publication. Command-line clients must send a matching Origin header explicitly. These checks prevent cross-site form submission and DNS rebinding; they do not authenticate a remote client.

Set `ANNOTATION_HOST=0.0.0.0` only to opt into listening on all interfaces. This emits a warning. To allow an external hostname, also set `ANNOTATION_TRUSTED_HOSTS` to its exact name or IP address. Separate multiple entries with commas, use brackets for IPv6, and omit schemes, ports, paths, and wildcards. Local hosts remain allowed. The bundled launcher does not trust forwarded headers.

For a container accessed only from the host computer, bind the container port to host loopback and explicitly enable the container listener:

```bash
docker build -t landuse-annotation .
docker run -p 127.0.0.1:8000:8000 \
  -e ANNOTATION_HOST=0.0.0.0 \
  -v annotation-state:/app/state \
  -v annotation-cache:/cache \
  landuse-annotation
```

Open <http://127.0.0.1:8000>. If you map a different host port, use that port in the browser URL. The original Dockerfile exposes port 8000 but does not override the loopback default. Running it without the explicit listener setting does not make the app reachable through a published port.

For access from another computer, prefer an SSH tunnel to the loopback listener. Direct network access needs both an explicit listener and the trusted hostname setting, plus network access controls. Do not publish this unauthenticated UI on the public internet.
