from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit

CRAP_LIMIT = 6.0


def crap_score(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def _coverage_lines(coverage_data: dict[str, Any], path: Path) -> tuple[set[int], set[int]]:
    for filename, data in coverage_data.get("files", {}).items():
        candidate = Path(filename)
        if candidate == path or candidate.resolve() == path.resolve() or str(path).endswith(filename):
            return set(data.get("executed_lines", [])), set(data.get("missing_lines", []))
    return set(), set()


def check(source_root: Path, coverage_path: Path) -> list[tuple[str, float]]:
    coverage_data = json.loads(coverage_path.read_text(encoding="utf-8"))
    failures: list[tuple[str, float]] = []
    for path in sorted(source_root.rglob("*.py")):
        executed, missing = _coverage_lines(coverage_data, path)
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        for block in cc_visit(source):
            if type(block).__name__ != "Function":
                continue
            if "pragma: no cover" in source_lines[block.lineno - 1]:
                continue
            first_line = int(block.lineno)
            last_line = int(getattr(block, "endline", first_line))
            lines = set(range(first_line, last_line + 1))
            executable = lines & (executed | missing)
            coverage = len(executable & executed) / len(executable) if executable else 1.0
            score = crap_score(int(block.complexity), coverage)
            print(f"{path}:{first_line} {block.name}: CRAP {score:.3f}")
            if score >= CRAP_LIMIT:
                failures.append((f"{path}:{first_line} {block.name}", score))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce a strict CRAP score below six.")
    parser.add_argument("--source-root", type=Path, default=Path("src/landuse_sentence_relevance"))
    parser.add_argument("--coverage", type=Path, default=Path("coverage.json"))
    args = parser.parse_args()
    failures = check(args.source_root, args.coverage)
    if failures:
        print("CRAP gate failed:")
        for name, score in failures:
            print(f"  {name}: {score:.3f} >= {CRAP_LIMIT}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
