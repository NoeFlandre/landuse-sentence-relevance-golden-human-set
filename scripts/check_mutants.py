from __future__ import annotations

import subprocess
import sys


def has_unfinished_mutants(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in ("survived", "🙁", "timeout", "untested", "no tests"))


def main() -> int:
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
    sys.exit(main())
