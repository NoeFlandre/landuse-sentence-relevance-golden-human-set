from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence


def has_unfinished_mutants(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in ("survived", "🙁", "timeout", "untested", "no tests"))


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
