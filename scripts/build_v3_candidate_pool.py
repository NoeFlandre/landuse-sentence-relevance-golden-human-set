"""Build and preflight the resumable V3 candidate reservoir."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence

from landuse_sentence_relevance.bootstrap import V3CandidatePoolResult, build_v3_candidate_pool
from landuse_sentence_relevance.config import V3Settings
from landuse_sentence_relevance.domain.models import Source


def main(argv: Sequence[str] | None = None) -> int:
    """Build the V3 pool and print only compact preflight evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    settings = V3Settings.from_env()
    try:
        result = build_v3_candidate_pool(settings)
    except (OSError, ValueError) as error:
        print(f"V3 candidate pool failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(settings, result), sort_keys=True))
    return 0


def _summary(settings: V3Settings, result: V3CandidatePoolResult) -> dict[str, object]:
    report = result.preflight
    return {
        "candidate_cells_by_source": _source_mapping(report.candidate_cells_by_source),
        "candidate_count_by_source": _source_mapping(report.candidate_count_by_source),
        "candidate_pool_path": str(settings.candidate_pool_path),
        "candidate_progress_path": str(settings.candidate_progress_path),
        "required_new_rows_by_source": _source_mapping(report.required_new_rows_by_source),
        "reserved_v2_cell_count": len(report.reserved_v2_cells),
        "total_candidate_count": report.total_candidate_count,
        "total_required_new_rows": report.total_required_new_rows,
    }


def _source_mapping(values: Mapping[Source, int]) -> dict[str, int]:
    return {source.value: int(value) for source, value in values.items()}


if __name__ == "__main__":
    raise SystemExit(main())
