from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_cli_regenerates_the_repository_asset(tmp_path: Path) -> None:
    output = tmp_path / "map.png"
    completed = subprocess.run(
        [sys.executable, "scripts/build_v3_world_map.py", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.stat().st_size > 10_000


def test_cli_exits_one_when_the_benchmark_is_missing(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_v3_world_map.py",
            "--benchmark",
            str(tmp_path / "missing.csv"),
            "--output",
            str(tmp_path / "map.png"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "V3 world map failed" in completed.stderr
    assert "Traceback" not in completed.stderr
