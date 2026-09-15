"""Build or resume the deterministic V3 annotation seed."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence

from landuse_sentence_relevance.bootstrap import build_v3_annotation_seed
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.profile import QuotaKey
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed, V3SeedRow


def main(argv: Sequence[str] | None = None) -> int:
    """Build the V3 seed and print only compact, non-text evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    settings = V3Settings.from_env()
    try:
        state = build_v3_annotation_seed(settings)
    except (OSError, ValueError) as error:
        print(f"V3 annotation seed failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(settings, state), sort_keys=True))
    return 0


def _summary(settings: V3Settings, state: V3AnnotationSeed) -> dict[str, object]:
    return {
        "annotation_seed_path": str(settings.annotation_seed_path),
        "benchmark_sha256": state.benchmark_sha256,
        "excluded_v2": [
            {
                "candidate_id": annotation.candidate.candidate_id,
                "reason": state.excluded_v2_reasons[annotation.candidate.candidate_id],
            }
            for annotation in sorted(state.excluded_v2_rows, key=lambda item: item.candidate.candidate_id)
        ],
        "pending_count": state.pending_row_count,
        "pending_count_by_source": _source_counts(state.pending_rows, state.quotas.sources),
        "pending_count_by_source_label": _quota_counts(state.pending_rows, state.quotas.counts),
        "quota_counts": _quota_counts(state.rows, state.quotas.counts),
        "reserved_v2_cell_count": len(state.reserved_v2_cells),
        "seeded_count": state.seeded_row_count,
        "seeded_count_by_source": _source_counts(state.seeded_rows, state.quotas.sources),
        "seeded_count_by_source_label": _quota_counts(state.seeded_rows, state.quotas.counts),
        "total_rows": state.total_rows,
        "rows_by_source": _source_counts(state.rows, state.quotas.sources),
    }


def _source_counts(rows: Iterable[V3SeedRow], sources: Sequence) -> dict[str, int]:
    counts = Counter(row.candidate.source for row in rows)
    return {source.value: counts[source] for source in sources}


def _quota_counts(rows: Iterable[V3SeedRow], quota_keys: Iterable[QuotaKey]) -> dict[str, int]:
    counts = Counter(row.quota_key for row in rows)
    return {f"{source.value}/{label.value}": counts[(source, label)] for source, label in quota_keys}


if __name__ == "__main__":
    raise SystemExit(main())
