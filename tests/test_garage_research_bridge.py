from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from carla_vision.operator import garage_server
from carla_vision.operator.garage_server import GARAGE_STATIC_ROOT, create_server

STATIC_ROOT = Path(__file__).parents[1] / "carla_vision" / "operator" / "static"


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=3.0) as response:
        return (
            response.status,
            response.headers.get_content_type(),
            response.read().decode("utf-8"),
        )


def test_garage_server_preserves_base_operator_and_injects_only_additive_assets(
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
        assert 'id="drive-start-form"' in html
        assert 'id="panel-tools"' in html
        assert html.count("/static/garage-integration.js") == 1
        assert html.count("/static/garage-research.js") == 1
        assert html.count("/static/garage-integration.css") == 1

        _, javascript_type, javascript = _get(f"{root}/static/garage-integration.js")
        assert javascript_type in {"text/javascript", "application/javascript"}
        assert "drive-garage-control-mode" in javascript
        assert "BehaviorAgent" in javascript
        assert "VOXEL PLANNER" in javascript
        assert 'startJob("verify"' in javascript
        assert 'startJob("analyze"' in javascript
        assert "refreshBootstrap()" in javascript
        assert "selectEvidence(path)" in javascript
        assert "/api/drive/control" not in javascript
        assert "apply_control" not in javascript

        _, research_type, research = _get(f"{root}/static/garage-research.js")
        assert research_type in {"text/javascript", "application/javascript"}
        for workflow in (
            "teacher_capture",
            "imitation_train",
            "voxel_capture",
            "voxel_flow_capture",
            "voxel_train",
            "voxel_flow_train",
            "voxel_shadow",
            "voxel_benchmark",
            "closed_loop_evaluate",
        ):
            assert workflow in research
        assert "/api/garage/jobs" in research
        assert "apply_control" not in research
        assert "shell" not in research.lower()

        _, css_type, css = _get(f"{root}/static/garage-integration.css")
        assert css_type == "text/css"
        assert ".drive-research-actions" in css
        assert ".garage-drive-mode" in css
        assert ".garage-research-launcher" in css
        assert "body.garage-autonomous" in css

        _, base_js_type, base_js = _get(f"{root}/static/app.js")
        assert base_js_type in {"text/javascript", "application/javascript"}
        assert 'request("/api/drive/start"' in base_js
        assert 'request("/api/drive/control"' in base_js

        _, catalog_type, catalog_text = _get(f"{root}/api/drive/catalog")
        assert catalog_type == "application/json"
        catalog = json.loads(catalog_text)
        modes = {row["id"] for row in catalog["control_modes"]}
        assert modes == {"manual", "behavior", "imitation", "voxel"}
        assert "models/imitation/best.pt" in catalog["policy_checkpoints"]
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
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--world-worker-token-env",
            "GARAGE_WORKER_TOKEN",
        ]
    )

    assert result == 0
    assert captured["world_worker_url"] == "http://127.0.0.1:8766"
    assert captured["world_worker_token"] == "secret-token"


def test_bridge_assets_are_separate_from_existing_static_files() -> None:
    assert (GARAGE_STATIC_ROOT / "garage-integration.js").is_file()
    assert (GARAGE_STATIC_ROOT / "garage-research.js").is_file()
    assert (GARAGE_STATIC_ROOT / "garage-integration.css").is_file()

    base_html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    base_js = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "garage-integration.js" not in base_html
    assert "garage-research.js" not in base_html
    assert "drive-research-actions" not in base_js
    assert "drive-garage-control-mode" not in base_js


def test_unknown_extra_static_asset_is_still_rejected(tmp_path: Path) -> None:
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
        with urllib.request.urlopen(
            f"http://{host}:{port}/static/not-allowed.js",
            timeout=3.0,
        ):
            raise AssertionError("unknown static asset unexpectedly succeeded")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.server_close()
        server.application.jobs.shutdown()
