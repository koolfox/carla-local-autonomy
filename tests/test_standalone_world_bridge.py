from __future__ import annotations

import ast
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from carla_vision.operator.situations import (
    PROP_PRESETS as OPERATOR_PROP_PRESETS,
)
from carla_vision.operator.situations import (
    WEATHER_PRESETS as OPERATOR_WEATHER_PRESETS,
)

ROOT = Path(__file__).resolve().parents[1]
WORLD_WORKER = ROOT / "carla_vision" / "native" / "world_worker.py"


def _copy_worker(tmp_path: Path) -> Path:
    copied = tmp_path / "world_worker.py"
    shutil.copyfile(WORLD_WORKER, copied)
    assert copied.read_bytes() == WORLD_WORKER.read_bytes()
    return copied


def test_world_worker_is_a_single_stdlib_only_script() -> None:
    source = WORLD_WORKER.read_text(encoding="utf-8")
    import_roots: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "standalone World Worker must not use relative imports"
            if node.module:
                import_roots.add(node.module.partition(".")[0])

    assert import_roots <= sys.stdlib_module_names
    assert 'import_module("carla")' in source
    assert 'import_module("agents.navigation.global_route_planner")' in source
    assert 'import_module("numpy")' in source
    assert 'import_module("cv2")' in source
    assert "save_to_disk" not in source


def test_exact_world_worker_file_runs_help_in_isolation(tmp_path: Path) -> None:
    copied = _copy_worker(tmp_path)
    result = subprocess.run(
        [sys.executable, "-I", str(copied), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "authenticated CARLA 0.9.16 LAN World Worker" in result.stdout
    assert "--carla-host" in result.stdout
    assert "--allow-lan" in result.stdout


def test_embedded_presets_match_operator_presets(tmp_path: Path) -> None:
    copied = _copy_worker(tmp_path)
    namespace = runpy.run_path(str(copied), run_name="standalone_world_worker_test")

    assert namespace["WEATHER_PRESETS"] == OPERATOR_WEATHER_PRESETS
    assert namespace["PROP_PRESETS"] == OPERATOR_PROP_PRESETS
