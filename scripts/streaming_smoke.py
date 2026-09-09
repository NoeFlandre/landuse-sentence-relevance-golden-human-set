from __future__ import annotations

from landuse_sentence_relevance.config import Settings
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig
from landuse_sentence_relevance.sources.remote_files import pinned_remote_file_urls


def _first_row(config: HuggingFaceRowConfig) -> None:
    rows = HuggingFaceDatasetRows(config)()
    row = next(iter(rows), None)
    if row is None:
        raise RuntimeError(f"stream returned no rows for {config.dataset_id}:{config.config}")
    print(f"streamed {config.dataset_id}:{config.config}/{config.split} ({len(row)} fields)")


def main() -> int:
    settings = Settings.from_env()
    wikipedia_files = pinned_remote_file_urls(
        settings.wikipedia_dataset_id,
        settings.wikipedia_dataset_revision,
        {
            "polygons": "polygons",
            "polygon_document_links": "polygon_document_links",
            "wikipedia_sections": "wikipedia/sections",
        },
        target_count=1,
        token=settings.hf_token,
    )
    for config in ("polygons", "polygon_document_links", "wikipedia_sections"):
        _first_row(
            HuggingFaceRowConfig(
                dataset_id=settings.wikipedia_dataset_id,
                revision=settings.wikipedia_dataset_revision,
                split=config,
                config=config,
                remote_files=wikipedia_files[config],
            )
        )
    website_files = pinned_remote_file_urls(
        settings.website_dataset_id,
        settings.website_dataset_revision,
        {"polygons": "polygons"},
        target_count=1,
        token=settings.hf_token,
    )
    _first_row(
        HuggingFaceRowConfig(
            dataset_id=settings.website_dataset_id,
            revision=settings.website_dataset_revision,
            split=settings.website_split,
            config=settings.website_config,
            remote_files=website_files["polygons"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
