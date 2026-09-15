from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from scripts.streaming_smoke import check_v3_streams

from landuse_sentence_relevance.bootstrap import v3_stream_specs
from landuse_sentence_relevance.config import V3Settings

RECORDED_SCHEMA = json.loads(
    (Path(__file__).parents[1] / "fixtures/v3_upstream_schema.json").read_text(encoding="utf-8")
)


def _recorded_row(stream_name: str) -> dict[str, Any]:
    columns = RECORDED_SCHEMA["streams"][stream_name]["columns"]
    return dict.fromkeys(columns, None)


def _remote_files(
    dataset_id: str,
    revision: str,
    directories: Mapping[str, str],
    target_count: int,
    token: str | None = None,
) -> dict[str, tuple[str, ...]]:
    return {key: (f"https://example.test/{dataset_id}/{revision}/{key}.parquet",) for key in directories}


def test_the_smoke_check_opens_every_v3_stream_with_the_production_projection() -> None:
    opened: list[tuple[str, tuple[str, ...] | None, str, str]] = []

    def open_rows(config: Any) -> Iterable[Mapping[str, Any]]:
        opened.append((config.dataset_id, config.columns, config.revision, config.split))
        stream = next(spec for spec in v3_stream_specs(V3Settings()) if spec.columns == config.columns)
        return [_recorded_row(stream.name)]

    reports = check_v3_streams(
        V3Settings(), remote_files_for=_remote_files, open_rows=open_rows, first_row_only=True
    )

    specs = v3_stream_specs(V3Settings())
    assert len(reports) == 5
    assert [spec.name for spec in specs] == [report.name for report in reports]
    assert opened == [(spec.dataset_id, spec.columns, spec.revision, spec.split) for spec in specs]
    assert all(report.missing_columns == () for report in reports)


def test_the_smoke_check_reports_a_projected_column_the_stream_does_not_publish() -> None:
    def open_rows(config: Any) -> Iterable[Mapping[str, Any]]:
        return [{"polygon_id": "p1"}]

    reports = check_v3_streams(
        V3Settings(), remote_files_for=_remote_files, open_rows=open_rows, first_row_only=True
    )

    assert all(report.missing_columns for report in reports)


def test_the_smoke_check_fails_when_a_stream_returns_no_rows() -> None:
    def open_rows(config: Any) -> Iterable[Mapping[str, Any]]:
        return []

    with pytest.raises(RuntimeError, match="returned no rows"):
        check_v3_streams(
            V3Settings(), remote_files_for=_remote_files, open_rows=open_rows, first_row_only=True
        )


def test_the_smoke_check_lists_the_recorded_upstream_directories() -> None:
    recorded = {name: stream["directory"] for name, stream in RECORDED_SCHEMA["streams"].items()}
    listed: dict[str, str] = {}

    def remote_files_for(
        dataset_id: str,
        revision: str,
        directories: Mapping[str, str],
        target_count: int,
        token: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        listed.update(directories)
        assert target_count == 1
        return {key: ("https://example.test/shard.parquet",) for key in directories}

    check_v3_streams(
        V3Settings(),
        remote_files_for=remote_files_for,
        open_rows=lambda config: [_recorded_row(_stream_name(config))],
        first_row_only=True,
    )

    assert listed == recorded


def _stream_name(config: Any) -> str:
    return next(spec.name for spec in v3_stream_specs(V3Settings()) if spec.columns == config.columns)


def test_the_smoke_check_builds_bounded_candidates_from_each_v3_source() -> None:
    from dataclasses import dataclass

    from scripts.streaming_smoke import (
        SMOKE_CANDIDATE_COUNT,
        SMOKE_REMOTE_FILE_COUNT,
        SMOKE_ROWS_PER_SHARD,
        check_v3_candidates,
    )

    from landuse_sentence_relevance.domain.models import Candidate, Source

    def candidate(index: int, source: Source) -> Candidate:
        return Candidate(
            candidate_id=f"{source.value}:{index}",
            sentence="An upstream sentence.",
            source=source,
            source_record_id=f"r{index}",
            source_field="field",
            h3_cell="8928308280fffff",
            h3_resolution=3,
            latitude=45.0,
            longitude=2.0,
            region="afghanistan",
            source_url="https://example.test",
        )

    class FakeSource:
        def __init__(self, source: Source) -> None:
            self._source = source

        def iter_candidates(self):
            return (candidate(index, self._source) for index in range(10))

    @dataclass(frozen=True)
    class FakeAdapters:
        description: FakeSource
        wikipedia: FakeSource
        website: FakeSource

    seen: list[V3Settings] = []

    def build_adapters(settings: V3Settings) -> FakeAdapters:
        seen.append(settings)
        return FakeAdapters(
            description=FakeSource(Source.DESCRIPTION),
            wikipedia=FakeSource(Source.WIKIPEDIA),
            website=FakeSource(Source.WEBSITE),
        )

    reports = check_v3_candidates(V3Settings(), build_adapters=build_adapters)

    assert [report.name for report in reports] == ["description", "wikipedia", "website"]
    assert all(len(report.candidates) == SMOKE_CANDIDATE_COUNT for report in reports)
    assert seen[0].remote_file_sample_count == SMOKE_REMOTE_FILE_COUNT
    assert seen[0].max_rows_per_shard == SMOKE_ROWS_PER_SHARD
    assert seen[0].description_dataset_revision == V3Settings().description_dataset_revision


def test_the_smoke_check_fails_when_a_v3_source_builds_no_candidates() -> None:
    from dataclasses import dataclass

    from scripts.streaming_smoke import check_v3_candidates

    class EmptySource:
        def iter_candidates(self):
            return iter(())

    @dataclass(frozen=True)
    class FakeAdapters:
        description: EmptySource
        wikipedia: EmptySource
        website: EmptySource

    with pytest.raises(RuntimeError, match="V3 description source built no candidates"):
        check_v3_candidates(
            V3Settings(),
            build_adapters=lambda settings: FakeAdapters(EmptySource(), EmptySource(), EmptySource()),
        )
