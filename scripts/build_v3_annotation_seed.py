"""Build or resume the deterministic V3 annotation seed."""

from __future__ import annotations

from collections.abc import Sequence

from landuse_sentence_relevance.bootstrap import build_v3_annotation_seed
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.v3_annotation import V3AnnotationSeed
from scripts._cli import quota_counts, run_json_cli, source_counts


def main(argv: Sequence[str] | None = None) -> int:
    """Build the V3 seed and print only compact, non-text evidence."""

    return run_json_cli(
        argv,
        description=__doc__,
        label="V3 annotation seed",
        build=build_v3_annotation_seed,
        summarize=_summary,
    )


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
        "pending_count_by_source": source_counts(state.pending_rows, state.quotas.sources),
        "pending_count_by_source_label": quota_counts(state.pending_rows, state.quotas.counts),
        "quota_counts": quota_counts(state.rows, state.quotas.counts),
        "reserved_v2_cell_count": len(state.reserved_v2_cells),
        "seeded_count": state.seeded_row_count,
        "seeded_count_by_source": source_counts(state.seeded_rows, state.quotas.sources),
        "seeded_count_by_source_label": quota_counts(state.seeded_rows, state.quotas.counts),
        "total_rows": state.total_rows,
        "rows_by_source": source_counts(state.rows, state.quotas.sources),
    }


if __name__ == "__main__":
    raise SystemExit(main())
