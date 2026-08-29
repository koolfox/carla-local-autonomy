from __future__ import annotations

import base64
import hashlib
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from carla_vision.operator import garage_server
from carla_vision.operator.product_console import (
    CONSOLE_ROOT,
    ProductConsoleRequestHandler,
    console_available,
    console_content_security_policy,
    resolve_console_asset,
)


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


def test_source_checkout_contains_deployable_console_bundle() -> None:
    assert console_available(CONSOLE_ROOT) is True
    html = (CONSOLE_ROOT / "index.html").read_text(encoding="utf-8")
    assert "/_app/immutable/" in html
    assert any((CONSOLE_ROOT / "_app" / "immutable").rglob("*.js"))


def test_product_console_serves_svelte_root_assets_and_legacy_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script, asset_bytes = _write_console(tmp_path / "console")
    console_root = tmp_path / "console"
    monkeypatch.setattr(ProductConsoleRequestHandler, "console_root", console_root)
    monkeypatch.setattr(
        garage_server,
        "GarageOperatorRequestHandler",
        ProductConsoleRequestHandler,
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
            expected = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode(
                "ascii"
            )
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


def test_product_console_fails_clearly_when_bundle_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProductConsoleRequestHandler, "console_root", tmp_path / "missing-console")
    monkeypatch.setattr(
        garage_server,
        "GarageOperatorRequestHandler",
        ProductConsoleRequestHandler,
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
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"http://{host}:{port}/", timeout=3.0)
        assert caught.value.code == 503
        html = caught.value.read().decode("utf-8")
        assert "Garage console build is missing" in html
        assert 'href="/legacy/"' in html

        with urllib.request.urlopen(f"http://{host}:{port}/legacy/", timeout=3.0) as response:
            legacy = response.read().decode("utf-8")
            assert response.status == 200
            assert 'id="drive-start-form"' in legacy
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()
