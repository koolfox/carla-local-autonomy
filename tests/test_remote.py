from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from carla_vision import remote
from carla_vision.launcher.config import LauncherError
from carla_vision.operator.garage_server import create_server


def private_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "remote.env"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_local_console_needs_no_credentials(tmp_path: Path) -> None:
    base, headers = remote.resolve_target(None, tmp_path / "missing.env", environ={})
    assert base == "http://127.0.0.1:8765/"
    assert headers == {}


def test_remote_console_requires_and_sends_access_headers(tmp_path: Path) -> None:
    env = private_env(
        tmp_path,
        "CARLA_REMOTE_URL=https://carla.example.com\n"
        "CF_ACCESS_CLIENT_ID='abc.access'\n"
        'CF_ACCESS_CLIENT_SECRET="s3cret"\n'
        "UNRELATED=ignored\n",
    )
    base, headers = remote.resolve_target(None, env, environ={})
    assert base == "https://carla.example.com/"
    assert headers == {"CF-Access-Client-Id": "abc.access", "CF-Access-Client-Secret": "s3cret"}

    with pytest.raises(LauncherError, match="service token"):
        remote.resolve_target("https://carla.example.com", tmp_path / "missing.env", environ={})


def test_credentials_are_never_sent_over_plain_http(tmp_path: Path) -> None:
    env = private_env(tmp_path, "CF_ACCESS_CLIENT_ID=a\nCF_ACCESS_CLIENT_SECRET=b\n")
    with pytest.raises(LauncherError, match="https"):
        remote.resolve_target("http://192.168.1.20:8765", env, environ={})
    with pytest.raises(LauncherError, match="both"):
        remote.resolve_target(
            "https://carla.example.com", tmp_path / "missing.env", environ={"CF_ACCESS_CLIENT_ID": "a"}
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_readable_env_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "remote.env"
    path.write_text("CF_ACCESS_CLIENT_SECRET=b\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(LauncherError, match="chmod 600"):
        remote.load_env_file(path)


@pytest.fixture
def operator_url(tmp_path: Path) -> Iterator[str]:
    server = create_server(workspace=tmp_path, port=0, carla_host="127.0.0.1", carla_port=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        server.application.jobs.shutdown()
        thread.join(timeout=5)


def test_status_reads_an_offline_console(operator_url: str) -> None:
    lines = remote.status(operator_url, {})
    assert lines[1] == "CARLA       offline"
    assert lines[2] == "Worker      not connected"
    assert lines[-1] == "Live check  Live check has not run yet."


def test_snapshot_reports_that_no_camera_is_live(operator_url: str, tmp_path: Path) -> None:
    with pytest.raises(LauncherError, match="no garage frame"):
        remote.snapshot(operator_url, {}, "garage", tmp_path / "frame.jpg")
    assert not (tmp_path / "frame.jpg").exists()


def test_cli_live_check_without_worker_exits_with_reason(
    operator_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = remote.main(["--url", operator_url, "--env-file", str(tmp_path / "none.env"), "live-check"])
    assert code == 2
    assert "connect the World Worker" in capsys.readouterr().err


class AccessLoginPage(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"<html><body>Sign in with Cloudflare Access</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_access_login_page_is_reported_as_a_token_problem() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), AccessLoginPage)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        with pytest.raises(LauncherError, match="Access service token"):
            remote.status(base, {})
        with pytest.raises(LauncherError, match="Access token"):
            remote.snapshot(base, {}, "garage", Path(os.devnull))
    finally:
        server.shutdown()
        server.server_close()
