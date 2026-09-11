from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from landuse_sentence_relevance.analysis.llm_rounds import RoundPreparationError, prepare_round

DEFAULT_BENCHMARK = Path("data/benchmark/v2-adjudicated.csv")
DEFAULT_OUTPUT_ROOT = Path("results/evaluations")


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a reproducible LLM evaluation round.")
    parser.add_argument("--round-id", required=True, help="New round directory name, for example round-02")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    round_directory = arguments.output_root / arguments.round_id
    try:
        manifest = prepare_round(
            round_directory,
            arguments.benchmark,
            arguments.prompt_file,
            round_id=arguments.round_id,
        )
    except (RoundPreparationError, OSError) as error:
        print(f"LLM round preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"prepared {manifest['round_id']}: {round_directory}")
    print(f"input rows: {manifest['input']['rows']}")
    print(f"prompt SHA-256: {manifest['prompt']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
