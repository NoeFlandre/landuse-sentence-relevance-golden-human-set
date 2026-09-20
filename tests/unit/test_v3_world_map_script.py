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
