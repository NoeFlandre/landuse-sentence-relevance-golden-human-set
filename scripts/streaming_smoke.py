from __future__ import annotations

from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig


def _first_row(config: HuggingFaceRowConfig) -> None:
    rows = HuggingFaceDatasetRows(config)()
    row = next(iter(rows), None)
    if row is None:
        raise RuntimeError(f"stream returned no rows for {config.dataset_id}:{config.config}")
    print(f"streamed {config.dataset_id}:{config.config}/{config.split} ({len(row)} fields)")


def main() -> int:
    settings = Settings.from_env()
    for config in ("polygons", "polygon_document_links", "wikipedia_sections"):
        _first_row(
            HuggingFaceRowConfig(
                dataset_id=settings.wikipedia_dataset_id,
                revision=settings.wikipedia_dataset_revision,
                split=config,
                config=config,
            )
        )
    _first_row(
        HuggingFaceRowConfig(
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            split=settings.website_split,
            config=settings.website_config,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
