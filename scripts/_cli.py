"""Shared helpers for the V3 build CLIs: JSON summaries and count mappings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Source
from landuse_sentence_relevance.domain.profile import QuotaKey
from landuse_sentence_relevance.domain.v3_annotation import V3SeedRow


def run_json_cli[Built](
    argv: Sequence[str] | None,
    *,
    description: str | None,
    label: str,
    build: Callable[[V3Settings], Built],
    summarize: Callable[[V3Settings, Built], dict[str, object]],
) -> int:
    """Build a V3 artifact and print its compact JSON summary; errors go to stderr."""

    parser = argparse.ArgumentParser(description=description)
    parser.parse_args(argv)
    settings = V3Settings.from_env()
    try:
        built = build(settings)
    except (OSError, ValueError) as error:
        print(f"{label} failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summarize(settings, built), sort_keys=True))
    return 0


def source_mapping(values: Mapping[Source, int]) -> dict[str, int]:
    return {source.value: int(value) for source, value in values.items()}


def source_counts(rows: Iterable[V3SeedRow], sources: Sequence[Source]) -> dict[str, int]:
    counts = Counter(row.candidate.source for row in rows)
    return source_mapping({source: counts[source] for source in sources})


def quota_counts(rows: Iterable[V3SeedRow], quota_keys: Iterable[QuotaKey]) -> dict[str, int]:
    counts = Counter(row.quota_key for row in rows)
    return {f"{source.value}/{label.value}": counts[(source, label)] for source, label in quota_keys}
