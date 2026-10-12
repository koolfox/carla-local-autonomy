from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("project_restore", ROOT / "restore_project.py")
assert SPEC is not None and SPEC.loader is not None
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def _backup(path: Path, payloads: dict[str, bytes], corrupt: str | None = None) -> None:
    files = [
        {
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mode": 0o600,
        }
        for name, data in payloads.items()
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": 1, "files": files}))
        for name, data in payloads.items():
            archive.writestr("data/" + name, b"damaged" if name == corrupt else data)


def test_private_backup_restores_data_without_recreating_process_ownership(tmp_path: Path) -> None:
    backup = tmp_path / "local-state.zip"
    root = tmp_path / "project"
    root.mkdir()
    payloads = {
        ".env.local": b"CARLA_HOST='auto'\n",
        "operator_sessions/history.json": b'{"historical":true}',
        "native_tasks/package/worker.py": b"print('retained source')\n",
        ".launcher/services.json": b'{"services":{"web":{"pid":1234}}}',
        ".launcher/launcher.lock": b"0",
        ".launcher/control/stop": b"stop",
        "native_tasks/package/__pycache__/worker.pyc": b"old bytecode",
        ":memory:.ses": b"scratch",
    }
    _backup(backup, payloads)
    restore.restore_backup(root, backup)
    for name in (".env.local", "operator_sessions/history.json", "native_tasks/package/worker.py"):
        assert (root / name).read_bytes() == payloads[name]
    assert not (root / ".launcher/services.json").exists()
    assert not (root / ".launcher/launcher.lock").exists()
    assert not (root / ".launcher/control/stop").exists()
    assert not (root / "native_tasks/package/__pycache__").exists()
    if os.name != "nt":
        assert (root / ":memory:.ses").read_bytes() == b"scratch"
        assert stat.S_IMODE((root / ".env.local").stat().st_mode) == 0o600


@pytest.mark.parametrize("corrupt", [False, True])
def test_backup_conflict_or_corruption_writes_no_private_files(
    tmp_path: Path, corrupt: bool
) -> None:
    root = tmp_path / "project"
    (root / "operator_sessions").mkdir(parents=True)
    existing = root / "operator_sessions/session.json"
    existing.write_bytes(b"newer local session")
    backup = tmp_path / "local-state.zip"
    payloads = {
        ".env.local": b"private settings",
        "operator_sessions/session.json": b"older session",
    }
    _backup(backup, payloads, corrupt="operator_sessions/session.json" if corrupt else None)
    with pytest.raises(restore.RestoreError):
        restore.restore_backup(root, backup)
    assert existing.read_bytes() == b"newer local session"
    assert not (root / ".env.local").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "case_collision"])
def test_unsafe_zip_is_rejected_before_extracting_any_file(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / "source.zip"
    target = tmp_path / "target"
    target.mkdir()
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("project/good.txt", "good")
        if kind == "traversal":
            source.writestr("project/../../escaped.txt", "bad")
        elif kind == "symlink":
            info = zipfile.ZipInfo("project/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            source.writestr(info, "../../escaped.txt")
        else:
            source.writestr("project/GOOD.txt", "collision")
    with pytest.raises(restore.RestoreError):
        restore.extract_zip(archive, target)
    assert list(target.iterdir()) == []
    assert not (tmp_path / "escaped.txt").exists()


def test_fresh_restore_refuses_existing_project_before_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    existing = root / "user.txt"
    existing.write_text("valuable work")
    monkeypatch.setattr(
        restore, "download", lambda *args: pytest.fail("Existing project was touched")
    )
    with pytest.raises(restore.RestoreError, match="new or empty"):
        restore.obtain_source(root, "a" * 40, False)
    assert existing.read_text() == "valuable work"


def test_failed_download_preserves_destination_and_removes_partial_file(
    tmp_path: Path, monkeypatch
) -> None:
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(b"wrong download")

    monkeypatch.setattr(restore.urllib.request, "build_opener", lambda *args: Opener())
    target = tmp_path / "weights.pt"
    target.write_bytes(b"existing file")
    with pytest.raises(restore.RestoreError, match="checksum"):
        restore.download("https://example.invalid/weights.pt", target, "0" * 64)
    assert target.read_bytes() == b"existing file"
    assert list(tmp_path.iterdir()) == [target]


def test_changed_full_model_is_preserved_instead_of_rehydrated(tmp_path: Path, monkeypatch) -> None:
    models = tmp_path / "models"
    models.mkdir()
    expected = b"published weight"
    (models / "assets.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "weights.pt",
                        "storage": "git-lfs",
                        "size_bytes": len(expected),
                        "sha256": hashlib.sha256(expected).hexdigest(),
                    }
                ]
            }
        )
    )
    weight = models / "weights.pt"
    weight.write_bytes(b"user changed weight")
    monkeypatch.setattr(
        restore, "download", lambda *args: pytest.fail("Changed weights must be preserved")
    )
    with pytest.raises(restore.RestoreError, match="changed"):
        restore.hydrate_models(tmp_path, "a" * 40)
    assert weight.read_bytes() == b"user changed weight"


def test_launcher_lock_blocks_other_restore_but_allows_child_setup(tmp_path: Path) -> None:
    child = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('restore_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    with module.restoration_lock(pathlib.Path(sys.argv[2]), sys.argv[3]):
        pass
except module.RestoreError:
    sys.exit(7)
"""
    command = [sys.executable, "-B", "-c", child, str(ROOT / "restore_project.py"), str(tmp_path)]
    with restore.restoration_lock(tmp_path, "launcher.lock"):
        assert subprocess.run(command + ["launcher.lock"], check=False).returncode == 7
        assert subprocess.run(command + ["operation.lock"], check=False).returncode == 0
    assert subprocess.run(command + ["launcher.lock"], check=False).returncode == 0


def test_plan_uses_main_without_network_or_creating_destination(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        restore, "resolve_commit", lambda *args: pytest.fail("Plan must stay offline")
    )
    target = tmp_path / "new project"
    assert restore.main(["--plan", "--role", "web", "--destination", str(target)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ref"] == "main"
    assert plan["role"] == "web"
    assert plan["server_start"] is False
    assert not target.exists()


def test_published_assets_match_inventories_or_exact_lfs_pointers() -> None:
    for folder, filename in (("models", "assets.json"), ("samples", "ARCHIVE.json")):
        inventory = json.loads((ROOT / folder / filename).read_text())
        for entry in inventory["files"]:
            path = ROOT / folder / entry["path"]
            assert path.is_file() and not path.is_symlink(), entry["path"]
            if entry.get("storage") == "git-lfs" and path.stat().st_size < 1024:
                assert (
                    path.read_bytes().replace(b"\r\n", b"\n")
                    == (
                        "version https://git-lfs.github.com/spec/v1\n"
                        f"oid sha256:{entry['sha256']}\nsize {entry['size_bytes']}\n"
                    ).encode()
                ), entry["path"]
            else:
                assert path.stat().st_size == entry["size_bytes"], entry["path"]
                assert restore.file_hash(path) == entry["sha256"], entry["path"]
