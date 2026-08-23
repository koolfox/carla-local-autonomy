from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from carla_vision.operator import garage_server
from carla_vision.operator.garage_server import create_server

STATIC_ROOT = Path(__file__).parents[1] / "carla_vision" / "operator" / "static"


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=3.0) as response:
        return (
            response.status,
            response.headers.get_content_type(),
            response.read().decode("utf-8"),
        )


def test_garage_server_serves_single_canonical_static_shell(
    tmp_path: Path,
) -> None:
    (tmp_path / "models" / "imitation").mkdir(parents=True)
    (tmp_path / "models" / "imitation" / "best.pt").write_bytes(b"checkpoint")
    server = create_server(
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

        status, content_type, html = _get(f"{root}/")
        assert status == 200
        assert content_type == "text/html"
        assert html == (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        assert 'id="drive-start-form"' in html
        assert 'id="panel-tools"' in html
        assert html.count("/static/app.css") == 1
        assert html.count("/static/app.js") == 1
        assert "garage-integration" not in html
        assert "garage-research.js" not in html

        _, base_js_type, base_js = _get(f"{root}/static/app.js")
        assert base_js_type in {"text/javascript", "application/javascript"}
        assert base_js == (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        assert 'request("/api/drive/start"' in base_js
        assert 'request("/api/drive/control"' in base_js

        _, base_css_type, base_css = _get(f"{root}/static/app.css")
        assert base_css_type == "text/css"
        assert base_css == (STATIC_ROOT / "app.css").read_text(encoding="utf-8")

        _, catalog_type, catalog_text = _get(f"{root}/api/drive/catalog")
        assert catalog_type == "application/json"
        catalog = json.loads(catalog_text)
        modes = {row["id"] for row in catalog["control_modes"]}
        assert modes == {"manual"}
        assert catalog["capabilities"]["garage_experimental"] is False
        assert "policy_checkpoints" not in catalog
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def test_garage_server_propagates_world_worker_to_replacement_drive_manager(
    tmp_path: Path,
) -> None:
    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "operator_sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
        world_worker_url="http://127.0.0.1:8766",
        world_worker_token="test-token",
    )
    try:
        assert server.application.world_worker is not None
        assert server.application.drive._world_worker is server.application.world_worker
        assert server.application.experimental_enabled is False
    finally:
        server.server_close()
        server.application.jobs.shutdown()


def test_garage_main_forwards_world_worker_cli_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeJobs:
        def shutdown(self) -> None:
            pass

    class FakeServer:
        server_address = ("127.0.0.1", 8765)
        application = type("Application", (), {"jobs": FakeJobs()})()

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval == 0.25

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    def fake_create_server(**kwargs: object) -> FakeServer:
        captured.update(kwargs)
        return FakeServer()

    monkeypatch.setenv("GARAGE_WORKER_TOKEN", "secret-token")
    monkeypatch.setattr(garage_server, "create_server", fake_create_server)

    result = garage_server.main(
        [
            "--workspace",
            str(tmp_path),
            "--carla-host",
            "127.0.0.1",
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--world-worker-token-env",
            "GARAGE_WORKER_TOKEN",
            "--enable-experimental",
        ]
    )

    assert result == 0
    assert captured["world_worker_url"] == "http://127.0.0.1:8766"
    assert captured["world_worker_token"] == "secret-token"
    assert captured["enable_experimental"] is True


def test_experimental_opt_in_advertises_modes_and_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "policy.pt").write_bytes(b"checkpoint")
    server = create_server(
        workspace=tmp_path,
        bind="127.0.0.1",
        port=0,
        sessions_root=tmp_path / "operator_sessions",
        carla_host="127.0.0.1",
        carla_port=65534,
        enable_experimental=True,
    )
    try:
        catalog = server.application.drive.catalog()
        assert {row["id"] for row in catalog["control_modes"]} == {
            "manual",
            "behavior",
            "imitation",
            "voxel",
        }
        assert catalog["capabilities"]["garage_experimental"] is True
        assert "models/policy.pt" in catalog["policy_checkpoints"]
    finally:
        server.server_close()
        server.application.jobs.shutdown()


def test_production_server_rejects_research_jobs(tmp_path: Path) -> None:
    server = create_server(
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
        payload = json.dumps(
            {"schema_version": "1.0", "kind": "closed_loop_evaluate", "parameters": {}}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://{host}:{port}/api/garage/jobs",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Operator-Token": server.application.token,
            },
        )
        try:
            urllib.request.urlopen(request, timeout=3.0)
        except urllib.error.HTTPError as error:
            assert error.code == 409
            response = json.loads(error.read().decode("utf-8"))
            assert response["error"]["type"] == "PermissionError"
            assert "--enable-experimental" in response["error"]["message"]
        else:
            raise AssertionError("production server accepted an experimental research job")
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()


def test_noncanonical_static_assets_are_rejected(tmp_path: Path) -> None:
    server = create_server(
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
        for asset in (
            "garage-integration.js",
            "garage-research.js",
            "garage-integration.css",
            "not-allowed.js",
        ):
            try:
                urllib.request.urlopen(
                    f"http://{host}:{port}/static/{asset}",
                    timeout=3.0,
                )
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError(f"noncanonical static asset unexpectedly served: {asset}")
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()
