"""Build and preflight the resumable V3 candidate reservoir."""

from __future__ import annotations

from collections.abc import Sequence

from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult, build_v3_candidate_pool
from landuse_sentence_relevance.config import V3Settings
from scripts._cli import run_json_cli, source_mapping


def main(argv: Sequence[str] | None = None) -> int:
    """Build the V3 pool and print only compact preflight evidence."""

    return run_json_cli(
        argv,
        description=__doc__,
        label="V3 candidate pool",
        build=build_v3_candidate_pool,
        summarize=_summary,
    )


def _summary(settings: V3Settings, result: V3CandidatePoolResult) -> dict[str, object]:
    report = result.preflight
    return {
        "candidate_cells_by_source": source_mapping(report.candidate_cells_by_source),
        "candidate_count_by_source": source_mapping(report.candidate_count_by_source),
        "candidate_pool_path": str(settings.candidate_pool_path),
        "candidate_progress_path": str(settings.candidate_progress_path),
        "required_new_rows_by_source": source_mapping(report.required_new_rows_by_source),
        "reserved_v2_cell_count": len(report.reserved_v2_cells),
        "total_candidate_count": report.total_candidate_count,
        "total_required_new_rows": report.total_required_new_rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
