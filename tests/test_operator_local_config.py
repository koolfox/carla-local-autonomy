from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path

import pytest

from carla_vision.operator import garage_server
from carla_vision.operator import server as base
from carla_vision.operator.local_entrypoint import (
    LocalConfigGarageRequestHandler,
    console_available,
    console_content_security_policy,
    load_local_env,
    prepare_launch,
    render_operator_index,
    resolve_console_asset,
)


def write_env(root: Path, text: str) -> None:
    (root / ".env.local").write_text(text, encoding="utf-8")


def test_local_env_loads_connection_secret_and_detector_default(tmp_path: Path) -> None:
    write_env(
        tmp_path,
        "\n".join(
            [
                "CARLA_WORLD_WORKER_TOKEN='P8cksgzdmrs24PRDMWc6YGvSjYOnLh+/hxIpsUpTlQs='",
                "CARLA_WORLD_WORKER_URL='http://192.168.1.108:8766'",
                "CARLA_HOST='192.168.1.108'",
                "CARLA_PORT='2000'",
                "detector.enabled=true",
                "overlay_frame_sequence=1",
            ]
        ),
    )

    plan = prepare_launch([], cwd=tmp_path, environ={})

    assert plan.detector_enabled is True
    assert plan.env_file == tmp_path / ".env.local"
    assert plan.env_updates["CARLA_WORLD_WORKER_TOKEN"].endswith("TlQs=")
    assert "P8cksgzdmrs24PRDMWc6YGvSjYOnLh+/hxIpsUpTlQs=" not in " ".join(plan.argv)
    args = base.parse_args(plan.argv)
    assert args.carla_host == "192.168.1.108"
    assert args.carla_port == 2000
    assert args.world_worker_url == "http://192.168.1.108:8766"


def test_process_environment_overrides_file_and_cli_overrides_environment(tmp_path: Path) -> None:
    write_env(
        tmp_path,
        "CARLA_HOST=192.168.1.108\nCARLA_PORT=2000\ndetector.enabled=false\n",
    )
    process_env = {
        "CARLA_HOST": "10.0.0.50",
        "CARLA_PORT": "2001",
        "CARLA_DETECTOR_ENABLED": "true",
    }

    plan = prepare_launch(
        ["--carla-host", "127.0.0.1", "--carla-port", "3000"],
        cwd=tmp_path,
        environ=process_env,
    )
    args = base.parse_args(plan.argv)

    assert args.carla_host == "127.0.0.1"
    assert args.carla_port == 3000
    assert plan.detector_enabled is True


def test_explicit_no_experimental_overrides_environment(tmp_path: Path) -> None:
    plan = prepare_launch(
        ["--no-enable-experimental"],
        cwd=tmp_path,
        environ={"CARLA_ENABLE_EXPERIMENTAL": "true"},
    )

    assert "--enable-experimental" not in plan.argv
    args = base.parse_args(plan.argv)
    assert args.enable_experimental is False


def test_detector_default_rewrites_only_the_checkbox_state() -> None:
    html = '<input id="drive-detector-enabled" type="checkbox" checked>\n<div>stable</div>'

    disabled = render_operator_index(html, detector_enabled=False)
    enabled = render_operator_index(disabled, detector_enabled=True)

    assert 'id="drive-detector-enabled" type="checkbox">' in disabled
    assert 'id="drive-detector-enabled" type="checkbox" checked>' in enabled
    assert enabled.count("checked") == 1
    assert "<div>stable</div>" in enabled


def test_runtime_state_is_ignored_as_configuration(tmp_path: Path) -> None:
    write_env(tmp_path, "overlay_frame_sequence=1\nCARLA_PORT=2000\n")

    values = load_local_env(tmp_path / ".env.local")

    assert values == {"CARLA_PORT": "2000"}
    plan = prepare_launch([], cwd=tmp_path, environ={})
    assert base.parse_args(plan.argv).carla_port == 2000


def test_markdown_world_worker_url_is_rejected(tmp_path: Path) -> None:
    write_env(
        tmp_path,
        "CARLA_WORLD_WORKER_URL='[http://192.168.1.108:8766](http://192.168.1.108:8766)'\n",
    )

    with pytest.raises(ValueError, match="raw URL"):
        prepare_launch([], cwd=tmp_path, environ={})


def test_unknown_keys_and_duplicate_keys_fail_loudly(tmp_path: Path) -> None:
    write_env(tmp_path, "SOMETHING_UNKNOWN=value\n")
    with pytest.raises(ValueError, match="unsupported"):
        load_local_env(tmp_path / ".env.local")

    write_env(tmp_path, "CARLA_HOST=one\nCARLA_HOST=two\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_local_env(tmp_path / ".env.local")


def test_missing_env_file_preserves_existing_defaults(tmp_path: Path) -> None:
    plan = prepare_launch([], cwd=tmp_path, environ={})
    args = base.parse_args(plan.argv)

    assert plan.env_file is None
    assert plan.detector_enabled is True
    assert args.carla_host == "auto"
    assert args.carla_port == 2000
    assert args.port == 8765


def _write_console(root: Path) -> tuple[str, bytes]:
    asset = root / "_app" / "immutable" / "entry" / "start.test.js"
    asset.parent.mkdir(parents=True)
    asset_bytes = b"export const garage = true;\n"
    asset.write_bytes(asset_bytes)
    script = "\nwindow.__garage_bootstrap__ = true;\n"
    (root / "index.html").write_text(
        "<!doctype html><html><head>"
        '<link href="/_app/immutable/entry/start.test.js" rel="modulepreload">'
        f"</head><body><div id=\"svelte\"></div><script>{script}</script></body></html>",
        encoding="utf-8",
    )
    return script, asset_bytes


def test_console_csp_hashes_exact_bootstrap_without_unsafe_inline(tmp_path: Path) -> None:
    script, _ = _write_console(tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    policy = console_content_security_policy(html)

    import base64

    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    assert f"'sha256-{digest}'" in policy
    assert "'unsafe-inline'" not in policy
    assert "script-src 'self'" in policy
    assert "style-src 'self'" in policy


def test_console_asset_resolution_is_allow_listed_and_symlink_safe(tmp_path: Path) -> None:
    _, asset_bytes = _write_console(tmp_path)

    resolved = resolve_console_asset("/_app/immutable/entry/start.test.js", tmp_path)

    assert resolved.read_bytes() == asset_bytes
    assert console_available(tmp_path) is True
    with pytest.raises(ValueError):
        resolve_console_asset("/_app/../secret.js", tmp_path)
    (tmp_path / "_app" / "immutable" / "entry" / "not-served.exe").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="type"):
        resolve_console_asset("/_app/immutable/entry/not-served.exe", tmp_path)

    target = tmp_path / "_app" / "immutable" / "entry" / "target.js"
    target.write_text("export {};", encoding="utf-8")
    link = tmp_path / "_app" / "immutable" / "entry" / "linked.js"
    try:
        link.symlink_to(target)
    except OSError:
        return
    with pytest.raises(ValueError, match="symlinks"):
        resolve_console_asset("/_app/immutable/entry/linked.js", tmp_path)


def test_product_entrypoint_serves_svelte_root_assets_and_explicit_legacy_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, asset_bytes = _write_console(tmp_path / "console")
    console_root = tmp_path / "console"
    monkeypatch.setattr(LocalConfigGarageRequestHandler, "console_root", console_root)
    monkeypatch.setattr(
        garage_server,
        "GarageOperatorRequestHandler",
        LocalConfigGarageRequestHandler,
    )
    server = garage_server.create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "operator_sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        root = f"http://{host}:{port}"
        with urllib.request.urlopen(f"{root}/", timeout=3.0) as response:
            html = response.read().decode("utf-8")
            csp = response.headers["Content-Security-Policy"]
            assert response.status == 200
            assert "__garage_bootstrap__" in html
            assert "unsafe-inline" not in csp
            expected = __import__("base64").b64encode(
                hashlib.sha256(script.encode("utf-8")).digest()
            ).decode("ascii")
            assert f"sha256-{expected}" in csp

        with urllib.request.urlopen(
            f"{root}/_app/immutable/entry/start.test.js", timeout=3.0
        ) as response:
            assert response.read() == asset_bytes
            assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"

        with urllib.request.urlopen(f"{root}/legacy/", timeout=3.0) as response:
            legacy = response.read().decode("utf-8")
            assert 'id="drive-start-form"' in legacy
            assert "/static/app.js" in legacy
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def test_product_entrypoint_falls_back_to_legacy_when_bundle_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_console = tmp_path / "missing-console"
    monkeypatch.setattr(LocalConfigGarageRequestHandler, "console_root", missing_console)
    monkeypatch.setattr(
        garage_server,
        "GarageOperatorRequestHandler",
        LocalConfigGarageRequestHandler,
    )
    server = garage_server.create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "operator_sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=3.0) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
            assert 'id="drive-start-form"' in html
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()
