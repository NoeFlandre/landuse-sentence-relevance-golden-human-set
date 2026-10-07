from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

UNFINISHED_MARKERS = ("survived", "🙁", "timeout", "untested", "no tests")


def _line_status(line: str) -> str:
    """Return the status part of a ``<mutant name>: <status>`` line, lowercased."""

    return line.rpartition(":")[2].lower()


def has_unfinished_mutants(output: str) -> bool:
    """Return True when any line's status marks a mutant as not killed."""

    return any(marker in _line_status(line) for line in output.splitlines() for marker in UNFINISHED_MARKERS)


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Fail when mutmut reports surviving mutants.").parse_args(argv)
    result = subprocess.run(
        ["mutmut", "results", "--all=true"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}{result.stderr}"
    print(output, end="")
    return 1 if result.returncode != 0 or has_unfinished_mutants(output) else 0


if __name__ == "__main__":
    raise SystemExit(main())
