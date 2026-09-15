"""Open every pinned upstream stream for real, reading one bounded row each.

The V3 half of this check opens all five streams with the exact production
projections and pinned revisions, so a column a pinned revision does not publish
fails here instead of failing a run with ``ArrowInvalid``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any

from landuse_sentence_relevance.bootstrap import (
    V3StreamSpec,
    build_v3_source_adapters,
    v3_row_config,
    v3_stream_specs,
    v3_streams_by_dataset,
)
from landuse_sentence_relevance.config import Settings, V3Settings
from landuse_sentence_relevance.domain.models import Candidate
from landuse_sentence_relevance.sources.huggingface import HuggingFaceDatasetRows, HuggingFaceRowConfig
from landuse_sentence_relevance.sources.remote_files import pinned_remote_file_urls

type Row = Mapping[str, Any]
type RemoteFilesFor = Callable[..., Mapping[str, tuple[str, ...]]]
type OpenRows = Callable[[HuggingFaceRowConfig], Iterable[Row]]

SMOKE_REMOTE_FILE_COUNT = 1
SMOKE_ROWS_PER_SHARD = 4_000
SMOKE_CANDIDATE_COUNT = 3


@dataclass(frozen=True, slots=True)
class StreamReport:
    """What one opened stream returned, and which projected columns it lacked."""

    name: str
    dataset_id: str
    revision: str
    fields: tuple[str, ...]
    missing_columns: tuple[str, ...]

    def line(self) -> str:
        state = "ok" if not self.missing_columns else f"MISSING {','.join(self.missing_columns)}"
        return f"{self.name}: {self.dataset_id}@{self.revision[:7]} {len(self.fields)} fields [{state}]"


def _open_rows(config: HuggingFaceRowConfig) -> Iterable[Row]:
    return HuggingFaceDatasetRows(config)()


def _first_row(config: HuggingFaceRowConfig, open_rows: OpenRows) -> Row:
    row = next(iter(open_rows(config)), None)
    if row is None:
        raise RuntimeError(f"stream returned no rows for {config.dataset_id}:{config.config}")
    return row


def _report(spec: V3StreamSpec, row: Row) -> StreamReport:
    return StreamReport(
        name=spec.name,
        dataset_id=spec.dataset_id,
        revision=spec.revision,
        fields=tuple(row),
        missing_columns=tuple(column for column in spec.columns if column not in row),
    )


def check_v3_streams(
    settings: V3Settings,
    *,
    remote_files_for: RemoteFilesFor = pinned_remote_file_urls,
    open_rows: OpenRows = _open_rows,
    first_row_only: bool = True,
) -> tuple[StreamReport, ...]:
    """Open all five V3 streams with the production projections and pinned revisions."""

    remote_files: dict[str, tuple[str, ...]] = {}
    for dataset_id, revision, specs in v3_streams_by_dataset(settings):
        remote_files.update(
            remote_files_for(
                dataset_id,
                revision,
                {spec.name: spec.directory for spec in specs},
                SMOKE_REMOTE_FILE_COUNT if first_row_only else settings.remote_file_sample_count,
                token=settings.hf_token,
            )
        )
    return tuple(
        _report(spec, _first_row(v3_row_config(spec, remote_files[spec.name]), open_rows))
        for spec in v3_stream_specs(settings)
    )


def check_v2_streams(
    settings: Settings,
    *,
    remote_files_for: RemoteFilesFor = pinned_remote_file_urls,
    open_rows: OpenRows = _open_rows,
) -> tuple[StreamReport, ...]:
    """Open the V2 streams the released pipeline reads, unchanged."""

    wikipedia_files = remote_files_for(
        settings.wikipedia_dataset_id,
        settings.wikipedia_dataset_revision,
        {
            "polygons": "polygons",
            "polygon_document_links": "polygon_document_links",
            "wikipedia_sections": "wikipedia/sections",
        },
        SMOKE_REMOTE_FILE_COUNT,
        token=settings.hf_token,
    )
    website_files = remote_files_for(
        settings.website_dataset_id,
        settings.website_dataset_revision,
        {"polygons": "polygons"},
        SMOKE_REMOTE_FILE_COUNT,
        token=settings.hf_token,
    )
    reports = []
    for name in ("polygons", "polygon_document_links", "wikipedia_sections"):
        config = HuggingFaceRowConfig(
            dataset_id=settings.wikipedia_dataset_id,
            revision=settings.wikipedia_dataset_revision,
            split=name,
            config=name,
            remote_files=wikipedia_files[name],
        )
        row = _first_row(config, open_rows)
        reports.append(
            StreamReport(
                name=f"v2_{name}",
                dataset_id=config.dataset_id,
                revision=config.revision,
                fields=tuple(row),
                missing_columns=(),
            )
        )
    config = HuggingFaceRowConfig(
        dataset_id=settings.website_dataset_id,
        revision=settings.website_dataset_revision,
        split=settings.website_split,
        config=settings.website_config,
        remote_files=website_files["polygons"],
    )
    row = _first_row(config, open_rows)
    reports.append(
        StreamReport(
            name="v2_website_polygons",
            dataset_id=config.dataset_id,
            revision=config.revision,
            fields=tuple(row),
            missing_columns=(),
        )
    )
    return tuple(reports)


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """The first bounded candidates one V3 source builds from real pinned rows."""

    name: str
    candidates: tuple[Candidate, ...]

    def line(self) -> str:
        first = self.candidates[0]
        return (
            f"{self.name}: {len(self.candidates)} candidates, "
            f"first region={first.region!r} url={first.source_url!r}"
        )


def _smoke_settings(settings: V3Settings) -> V3Settings:
    return replace(
        settings,
        remote_file_sample_count=SMOKE_REMOTE_FILE_COUNT,
        max_rows_per_shard=SMOKE_ROWS_PER_SHARD,
    )


def check_v3_candidates(
    settings: V3Settings,
    *,
    build_adapters: Callable[[V3Settings], Any] = build_v3_source_adapters,
) -> tuple[CandidateReport, ...]:
    """Build a few real candidates per V3 source from one pinned shard each.

    Opening the streams proves the projections exist; this proves the adapters
    still emit English, geolocated, provenanced candidates from the pinned rows
    those projections return.
    """

    adapters = build_adapters(_smoke_settings(settings))
    reports = []
    for name, source in (
        ("description", adapters.description),
        ("wikipedia", adapters.wikipedia),
        ("website", adapters.website),
    ):
        candidates = tuple(islice(source.iter_candidates(), SMOKE_CANDIDATE_COUNT))
        if not candidates:
            raise RuntimeError(f"V3 {name} source built no candidates from its pinned shard")
        reports.append(CandidateReport(name=name, candidates=candidates))
    return tuple(reports)


def main() -> int:
    reports = (*check_v2_streams(Settings.from_env()), *check_v3_streams(V3Settings.from_env()))
    for report in reports:
        print(f"streamed {report.line()}")
    incomplete = tuple(report for report in reports if report.missing_columns)
    if incomplete:
        for report in incomplete:
            print(f"projection names columns {report.dataset_id} does not publish: {report.line()}")
        return 1
    for candidate_report in check_v3_candidates(V3Settings.from_env()):
        print(f"built {candidate_report.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
