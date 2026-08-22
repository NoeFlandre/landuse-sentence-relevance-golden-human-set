FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    MODEL_CACHE_DIR=/cache/models \
    SESSION_PATH=/app/state/annotations.jsonl

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra models --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra models

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/state /cache \
    && chown -R app:app /app /cache
USER app

EXPOSE 8000
VOLUME ["/app/state", "/cache"]
CMD ["uv", "run", "landuse-annotate"]
