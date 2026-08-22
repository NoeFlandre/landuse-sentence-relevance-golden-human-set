from __future__ import annotations

import subprocess
import sys


def main() -> int:
    result = subprocess.run(
        ["mutmut", "results", "--all=true"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}{result.stderr}"
    print(output, end="")
    lowered = output.lower()
    unfinished = "survived" in lowered or "🙁" in output or "timeout" in lowered or "untested" in lowered
    return 1 if result.returncode != 0 or unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
