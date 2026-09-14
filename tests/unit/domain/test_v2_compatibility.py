"""Guards that adding a third source never changes the released V2 contract."""

from __future__ import annotations

import csv
from pathlib import Path

from landuse_sentence_relevance.domain.constraints import DEFAULT_QUOTAS, V2_SOURCES
from landuse_sentence_relevance.domain.models import Label, Source
from landuse_sentence_relevance.domain.stratification import DEFAULT_SOURCES

ROOT = Path(__file__).parents[3]
BENCHMARK = ROOT / "data/benchmark/v2-adjudicated.csv"


def test_v2_source_set_excludes_sources_added_after_the_release() -> None:
    assert V2_SOURCES == (Source.WIKIPEDIA, Source.WEBSITE)
    assert DEFAULT_SOURCES == V2_SOURCES
    assert Source.DESCRIPTION not in V2_SOURCES


def test_default_quotas_keep_the_released_v2_marginals() -> None:
    assert (DEFAULT_QUOTAS.total, DEFAULT_QUOTAS.per_source, DEFAULT_QUOTAS.per_label) == (100, 50, 50)
    assert (DEFAULT_QUOTAS.cell_count, DEFAULT_QUOTAS.rows_per_cell) == (100, 1)
    assert DEFAULT_QUOTAS.sources == V2_SOURCES
    assert DEFAULT_QUOTAS.labels == (Label.YES, Label.NO)


def test_released_benchmark_never_contains_a_source_added_after_it() -> None:
    with BENCHMARK.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 154
    assert {row["source"] for row in rows} == {Source.WIKIPEDIA.value, Source.WEBSITE.value}
    assert {row["label"] for row in rows} == {Label.YES.value, Label.NO.value}
