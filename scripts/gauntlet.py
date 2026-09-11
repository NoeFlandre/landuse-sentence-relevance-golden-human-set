from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

QUALITY_DIRECTORIES = ("src", "tests", "scripts")


class MutationGateBusyError(RuntimeError):
    """Raised when another mutation gate already owns the project lock."""


def mutation_lock_path() -> Path:
    """Return the lock path in the configured project state directory."""

    return Path(os.environ.get("PROJECT_STATE_ROOT") or "state") / "mutation.lock"


@contextmanager
def mutation_lock(path: Path) -> Iterator[None]:
    """Hold a nonblocking process lock for the mutation result tree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MutationGateBusyError(f"mutation gate already running: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def default_mutation_workers() -> int:
    """Use every core for mutation testing; each mutant is independent of the others."""

    return max(1, os.cpu_count() or 1)


def quality_paths(project_root: Path = Path(".")) -> tuple[str, ...]:
    """Return configured quality roots that exist in this checkout."""

    return tuple(directory for directory in QUALITY_DIRECTORIES if (project_root / directory).is_dir())


def run_step(name: str, command: list[str], environment: dict[str, str]) -> None:
    print(f"\n== {name} ==")
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic project QA gauntlet.")
    parser.add_argument("--skip-network", action="store_true", help="Skip the remote streaming smoke check")
    parser.add_argument("--skip-docker", action="store_true", help="Skip the Docker build check")
    parser.add_argument(
        "--mutation-workers",
        type=int,
        default=default_mutation_workers(),
        help="Mutmut worker processes; defaults to the core count",
    )
    args = parser.parse_args()
    workers = str(args.mutation_workers)

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["MUTMUT_MAX_CHILDREN"] = workers
    paths = list(quality_paths())
    run_step("lock", ["uv", "lock", "--check"], environment)
    run_step("format", ["uv", "run", "ruff", "format", "--check", *paths], environment)
    run_step("ruff", ["uv", "run", "ruff", "check", *paths], environment)
    run_step("ty", ["uv", "run", "ty", "check", *paths], environment)
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
    try:
        with mutation_lock(mutation_lock_path()):
            run_step("mutation", ["uv", "run", "mutmut", "run", "--max-children", workers], environment)
            run_step(
                "zero surviving mutants", ["uv", "run", "python", "scripts/check_mutants.py"], environment
            )
    except MutationGateBusyError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
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
