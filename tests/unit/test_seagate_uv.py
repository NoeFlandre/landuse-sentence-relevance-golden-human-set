from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_seagate_uv_routes_project_state_to_the_external_drive(tmp_path: Path) -> None:
    fake_uv_dir = tmp_path / "bin"
    fake_uv_dir.mkdir()
    fake_uv = fake_uv_dir / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        "printf 'UV_CACHE_DIR=%s\\n' \"$UV_CACHE_DIR\"\n"
        "printf 'UV_PROJECT_ENVIRONMENT=%s\\n' \"$UV_PROJECT_ENVIRONMENT\"\n"
        "printf 'TMPDIR=%s\\n' \"$TMPDIR\"\n"
        "printf 'XDG_CACHE_HOME=%s\\n' \"$XDG_CACHE_HOME\"\n"
        "printf 'PYTHONPYCACHEPREFIX=%s\\n' \"$PYTHONPYCACHEPREFIX\"\n"
        "printf 'USE_TORCH=%s\\n' \"$USE_TORCH\"\n"
        "printf 'LANGUAGE_MODEL_DEVICE=%s\\n' \"$LANGUAGE_MODEL_DEVICE\"\n"
        "printf 'HF_HOME=%s\\n' \"$HF_HOME\"\n"
        "printf 'HF_TOKEN_PATH=%s\\n' \"$HF_TOKEN_PATH\"\n"
        "printf 'UV_PYTHON_INSTALL_DIR=%s\\n' \"$UV_PYTHON_INSTALL_DIR\"\n"
        "printf 'PWD=%s\\n' \"$PWD\"\n"
        "printf 'ARGS=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    data_root = tmp_path / "seagate-project"
    data_root.mkdir()
    environment = os.environ | {
        "PATH": f"{fake_uv_dir}:{os.environ['PATH']}",
        "PROJECT_DATA_ROOT": str(data_root),
    }
    for variable in (
        "UV_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PYTHONPYCACHEPREFIX",
        "USE_TORCH",
        "LANGUAGE_MODEL_DEVICE",
        "HF_HOME",
        "HF_TOKEN_PATH",
        "UV_PYTHON_INSTALL_DIR",
    ):
        environment.pop(variable, None)

    result = subprocess.run(
        [str(ROOT / "scripts/uv-seagate"), "run", "python", "-V"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    values = dict(line.split("=", maxsplit=1) for line in result.stdout.splitlines())
    assert values == {
        "UV_CACHE_DIR": str(data_root / "uv-cache"),
        "UV_PROJECT_ENVIRONMENT": str(data_root / "uv-environment"),
        "TMPDIR": str(data_root / "tmp"),
        "XDG_CACHE_HOME": str(data_root / "xdg-cache"),
        "PYTHONPYCACHEPREFIX": str(data_root / "python-cache"),
        "USE_TORCH": "0",
        "LANGUAGE_MODEL_DEVICE": "cpu",
        "HF_HOME": str(data_root / "huggingface-auth"),
        "HF_TOKEN_PATH": str(data_root / "huggingface-auth" / "token"),
        "UV_PYTHON_INSTALL_DIR": str(data_root / "uv-python"),
        "PWD": str(ROOT),
        "ARGS": "run python -V",
    }
    assert all(
        (data_root / directory).is_dir()
        for directory in (
            "uv-cache",
            "uv-environment",
            "tmp",
            "xdg-cache",
            "python-cache",
        )
    )


def test_seagate_uv_refuses_to_fallback_to_local_storage(tmp_path: Path) -> None:
    missing_root = tmp_path / "not-mounted"
    environment = os.environ | {"PROJECT_DATA_ROOT": str(missing_root)}

    result = subprocess.run(
        [str(ROOT / "scripts/uv-seagate"), "run", "python", "-V"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert str(missing_root) in result.stderr
