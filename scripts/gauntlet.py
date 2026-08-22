from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_step(name: str, command: list[str], environment: dict[str, str]) -> None:
    print(f"\n== {name} ==")
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic project QA gauntlet.")
    parser.add_argument("--skip-network", action="store_true", help="Skip the remote streaming smoke check")
    parser.add_argument("--skip-docker", action="store_true", help="Skip the Docker build check")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["MUTMUT_MAX_CHILDREN"] = "1"
    run_step("lock", ["uv", "lock", "--check"], environment)
    run_step("format", ["uv", "run", "ruff", "format", "--check", "src", "tests", "scripts"], environment)
    run_step("ruff", ["uv", "run", "ruff", "check", "src", "tests", "scripts"], environment)
    run_step("ty", ["uv", "run", "ty", "check", "src", "tests", "scripts"], environment)
    run_step(
        "tests and coverage",
        [
            "uv",
            "run",
            "pytest",
            "tests/unit",
            "tests/acceptance",
            "-q",
            "--cov=landuse_sentence_relevance",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
        ],
        environment,
    )
    run_step("crap", ["uv", "run", "python", "scripts/check_crap.py"], environment)
    run_step("mutation", ["uv", "run", "mutmut", "run", "--max-children", "1"], environment)
    run_step("zero surviving mutants", ["uv", "run", "python", "scripts/check_mutants.py"], environment)
    if not args.skip_network:
        run_step("streaming smoke", ["uv", "run", "python", "scripts/streaming_smoke.py"], environment)
    if not args.skip_docker:
        run_step(
            "docker build",
            ["docker", "build", "--tag", "landuse-sentence-relevance-golden-human-set:qa", "."],
            environment,
        )
    print("\nGAUNTLET PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
