"""Launcher contracts without a CARLA server or native installation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_vision.launcher import config, install, network, services
from carla_vision.launcher.config import LauncherConfig, LauncherError


def test_mac_defaults_and_existing_remote_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    (tmp_path / ".env.local").write_text(
        "CARLA_HOST=192.168.1.10\nCARLA_WORLD_WORKER_URL=http://192.168.1.10:8766\n"
        "CARLA_WORLD_WORKER_TOKEN=existing-private-token\n"
    )
    selected = LauncherConfig.load(tmp_path)
    assert (selected.mode, selected.profile, selected.carla_host) == ("web", "lean", "192.168.1.10")
    assert config.remote_token(tmp_path) == "existing-private-token"
    selected.save(tmp_path)
    assert "existing-private-token" not in (tmp_path / ".launcher/config.json").read_text()


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@host:8766",
        "file:///tmp/worker",
        "http://host:0",
        "http://host:bad",
        "http://host/v1/health",
    ],
)
def test_invalid_worker_urls_fail_before_requests(url):
    with pytest.raises(LauncherError):
        LauncherConfig(mode="web", remote_worker_url=url)


@pytest.mark.parametrize("address", ["0.0.0.0", "8.8.8.8", "224.0.0.1", "::1"])
def test_worker_binding_cannot_expose_public_or_wildcard_interfaces(address):
    with pytest.raises(LauncherError):
        LauncherConfig(worker_bind=address, share_worker=True)


def test_pairing_roundtrip_and_file_preservation(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "native_platform", lambda: "Windows")
    monkeypatch.setattr(network, "lan_addresses", lambda: ["192.168.1.10"])
    host = services.LocalStack(tmp_path / "host", lambda _: None)
    host.configure(LauncherConfig(mode="local", share_worker=True, worker_bind="192.168.1.10"))
    destination = tmp_path / "connection.json"
    network.export_connection(host, destination)
    monkeypatch.setattr(
        services,
        "http_json",
        lambda *_, **__: {"ready": True, "carla": {"server_version": "0.9.16"}},
    )
    client = services.LocalStack(tmp_path / "client", lambda _: None)
    network.import_connection(client, destination)
    assert client.config.mode == "web"
    assert client.config.worker_url == "http://192.168.1.10:8766"
    assert client.config.carla_host == "192.168.1.10"
    assert client.credential() == host.credential()
    assert client.credential() not in json.dumps(client.snapshot())
    previous = destination.read_bytes()
    with pytest.raises(LauncherError, match="existing files"):
        network.export_connection(host, destination)
    assert destination.read_bytes() == previous


def test_bad_import_preserves_saved_configuration_and_secret(tmp_path):
    stack = services.LocalStack(tmp_path, lambda _: None)
    stack.configure(LauncherConfig(mode="web"), token="original")
    incoming = tmp_path / "bad.json"
    incoming.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "carla_version": "0.9.16",
                "token": "new",
                "worker_url": "http://user:secret@host",
                "carla_host": "host",
                "carla_port": 2000,
            }
        )
    )
    with pytest.raises(LauncherError):
        network.import_connection(stack, incoming)
    assert stack.config.remote_worker_url == ""
    assert stack.credential() == "original"


def test_web_start_has_no_native_dependency_and_never_spawns_remote_services(tmp_path, monkeypatch):
    selected = LauncherConfig(
        mode="web", carla_host="192.168.1.10", remote_worker_url="http://192.168.1.10:8766"
    )
    selected.save(tmp_path)
    config.save_remote_token(tmp_path, "private-token")
    stack = services.LocalStack(tmp_path, lambda _: None)
    commands = []

    def spawn(role, command, env, state):
        commands.append((role, command, env))
        state["services"][role] = {"role": role}
        return state["services"][role]

    monkeypatch.setattr(stack, "_spawn", spawn)
    monkeypatch.setattr(stack, "snapshot", lambda: {"ready": True})
    monkeypatch.setattr(services, "_process", lambda entry: object() if entry else None)
    monkeypatch.setattr(services, "port_open", lambda _: False)
    monkeypatch.setattr(services, "web_ready", lambda _: None)
    monkeypatch.setattr(
        services, "native_platform", lambda: pytest.fail("native platform probed in web mode")
    )
    monkeypatch.setattr(
        services.importlib.metadata,
        "version",
        lambda _: pytest.fail("native SDK inspected in web mode"),
    )
    assert stack.start(open_browser=False)["ready"]
    role, command, env = commands[0]
    assert len(commands) == 1 and role == "web"
    assert "private-token" not in " ".join(command)
    assert env["CARLA_WORLD_WORKER_TOKEN"] == "private-token"
    assert env["PATH"].startswith(str(tmp_path / ".launcher/bin"))
    assert services.read_json(stack.state_path, {})["services"].keys() == {"web"}


def test_offline_web_start_disables_stale_worker_settings(tmp_path, monkeypatch):
    LauncherConfig(mode="web").save(tmp_path)
    stack = services.LocalStack(tmp_path, lambda _: None)
    seen = []
    monkeypatch.setattr(
        stack, "_spawn", lambda role, command, env, state: seen.append(command) or {"pid": 1}
    )
    monkeypatch.setattr(stack, "snapshot", lambda: {"ready": True})
    monkeypatch.setattr(services, "_process", lambda _: object())
    monkeypatch.setattr(stack, "active", lambda: False)
    monkeypatch.setattr(services, "port_open", lambda _: False)
    monkeypatch.setattr(services, "web_ready", lambda _: None)
    stack.start(open_browser=False)
    assert "--no-world-worker" in seen[0]
    assert "--world-worker-url" not in seen[0]


def test_web_stop_only_cleans_local_owners_when_remote_worker_has_other_work(tmp_path, monkeypatch):
    stack = services.LocalStack(tmp_path, lambda _: None)
    selected = LauncherConfig(mode="web", remote_worker_url="http://192.168.1.10:8766")
    config.save_remote_token(tmp_path, "secret")
    calls = []
    monkeypatch.setattr(
        services,
        "http_json",
        lambda url, **kwargs: calls.append((url, kwargs)) or {"token": "operator"},
    )
    monkeypatch.setattr(
        stack,
        "_worker_health",
        lambda _: {"ready": True, "busy": True, "active_scene": "another-owner"},
    )
    monkeypatch.setattr(
        stack,
        "_owner_states",
        lambda _: (
            {"status": "idle", "recording": False},
            {"active": False},
            {"active": False, "holds_world": False, "phase": "idle"},
        ),
    )
    stack._cleanup_scene(selected)
    assert all("body" not in kwargs for _, kwargs in calls)


def test_repair_web_profile_never_installs_native_sdk(tmp_path, monkeypatch):
    LauncherConfig(mode="web", profile="lean").save(tmp_path)
    for name in ["pyproject.toml", "uv.lock"]:
        (tmp_path / name).touch()
    commands = []
    monkeypatch.setattr(install, "uv_binary", lambda _: "uv")
    monkeypatch.setattr(
        install, "run_install_command", lambda command, *_: commands.append(command)
    )
    monkeypatch.setattr(install.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=0))
    monkeypatch.setattr(install, "prepare_ffmpeg", lambda *_: None)
    monkeypatch.setattr(install, "hydrate_models", lambda *_: None)
    monkeypatch.setattr(install, "native_platform", lambda: pytest.fail("native setup attempted"))
    install.setup(tmp_path, report=lambda _: None)
    assert "launcher" in commands[0]
    assert "native-host" not in commands[0] and "vision" not in commands[0]


def test_web_configuration_can_be_checked_while_owned_app_runs(tmp_path, monkeypatch):
    stack = services.LocalStack(tmp_path)
    selected = LauncherConfig(mode="web")
    selected.save(tmp_path)
    config.save_remote_token(tmp_path, "secret")
    monkeypatch.setattr(stack, "active", lambda: True)
    stack.configure(selected, token="secret")
    with pytest.raises(LauncherError, match="Stop"):
        stack.configure(replace(selected, web_port=9876), token="secret")


def test_lan_worker_start_uses_selected_address_and_no_token_argument(tmp_path, monkeypatch):
    selected = LauncherConfig(mode="local", share_worker=True, worker_bind="192.168.1.10")
    selected.save(tmp_path)
    stack = services.LocalStack(tmp_path, lambda _: None)
    commands = []

    def spawn(role, command, env, state):
        commands.append((role, command, env))
        state["services"][role] = {"role": role, "descendants": []}
        return state["services"][role]

    monkeypatch.setattr(stack, "_spawn", spawn)
    monkeypatch.setattr(stack, "_remember_children", lambda *_: None)
    monkeypatch.setattr(stack, "_server_facts", lambda _: {"version": "0.9.16", "map": "Town01"})
    monkeypatch.setattr(stack, "snapshot", lambda: {"ready": True})
    monkeypatch.setattr(stack, "active", lambda: False)
    monkeypatch.setattr(services, "_process", lambda entry: object() if entry else None)
    monkeypatch.setattr(services, "port_open", lambda _: False)
    monkeypatch.setattr(services, "native_platform", lambda: "Windows")
    monkeypatch.setattr(services.importlib.metadata, "version", lambda _: "0.9.16")
    monkeypatch.setattr(services, "model_status", lambda _: {"ready": True})
    monkeypatch.setattr(services, "find_carla", lambda *_: tmp_path / "CarlaUE4.exe")
    monkeypatch.setattr(services, "web_ready", lambda _: None)
    monkeypatch.setattr(
        services,
        "http_json",
        lambda *_, **__: {"ready": True, "carla": {"server_version": "0.9.16"}},
    )
    stack.start(open_browser=False)
    assert [role for role, *_ in commands] == ["carla", "worker", "web"]
    worker = commands[1][1]
    assert worker[worker.index("--bind") + 1] == "192.168.1.10"
    assert "--allow-lan" in worker
    secret = config.worker_token(tmp_path)
    assert secret and all(secret not in " ".join(command) for _, command, _ in commands)
    assert commands[2][2]["CARLA_WORLD_WORKER_URL"] == "http://192.168.1.10:8766"


def test_windows_firewall_is_explicit_uac_and_private_subnet_only(tmp_path, monkeypatch):
    import base64
    import re

    LauncherConfig(mode="local", share_worker=True, worker_bind="192.168.1.10").save(tmp_path)
    stack = services.LocalStack(tmp_path, lambda _: None)
    monkeypatch.setattr(network, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(network, "lan_addresses", lambda: ["192.168.1.10"])
    monkeypatch.setattr(network, "find_carla", lambda *_: None)
    calls = []
    monkeypatch.setattr(
        network.subprocess,
        "run",
        lambda command, **_: calls.append(command) or SimpleNamespace(returncode=0),
    )
    network.windows_firewall(stack)
    assert len(calls) == 1
    request = calls[0][-1]
    assert "-Verb RunAs" in request
    encoded = re.search(r"-EncodedCommand ([A-Za-z0-9+/=]+)", request).group(1)
    script = base64.b64decode(encoded).decode("utf-16le")
    assert "-LocalPort 8766" in script and "-LocalAddress '192.168.1.10'" in script
    assert "-Profile Private -RemoteAddress LocalSubnet" in script
    assert "Public" not in script


def test_windows_prerequisite_installation_never_forces_restart(tmp_path, monkeypatch):
    LauncherConfig(mode="local").save(tmp_path)
    installer = tmp_path / "Engine/Extras/Redist/en-us/UE4PrereqSetup_x64.exe"
    installer.parent.mkdir(parents=True)
    installer.touch()
    messages, commands = [], []
    stack = services.LocalStack(tmp_path, messages.append)
    monkeypatch.setattr(network, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(network, "find_carla", lambda *_: tmp_path / "CarlaUE4.exe")
    monkeypatch.setattr(
        network.subprocess,
        "run",
        lambda command, **_: commands.append(command) or SimpleNamespace(returncode=3010),
    )
    network.windows_prerequisites(stack)
    assert "/norestart" in commands[0][-1] and "-Verb RunAs" in commands[0][-1]
    assert any("Restart Windows" in message for message in messages)


def test_tui_modes_and_private_token_at_terminal_sizes(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from carla_vision.launcher.tui import LauncherApp

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr("carla_vision.launcher.tui.default_mode", lambda: "local")
    monkeypatch.setattr(
        services.LocalStack,
        "snapshot",
        lambda _: {
            "services": {
                role: {"status": "external" if role != "web" else "stopped", "owned": False}
                for role in ("carla", "worker", "web")
            },
            "models": {"ready": True},
            "config": {"mode": "web"},
            "ready": False,
            "owned_services_active": False,
        },
    )

    async def exercise(size):
        app = LauncherApp(tmp_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert app.query_one("#mode", Select).value == "web"
            assert app.query_one("#remote-settings").display
            assert not app.query_one("#local-settings").display
            assert app.query_one("#worker-token", Input).password
            app.query_one("#worker-url", Input).value = "http://192.168.1.10:8766"
            assert app._settings().remote_worker_url == "http://192.168.1.10:8766"
            from textual.widgets import TabbedContent

            app.query_one(TabbedContent).active = "run"
            await pilot.pause()
            assert app.query_one("#stop").region.bottom < size[1]
            app._set_busy(True)
            assert app.query_one("#start").disabled
            app._set_busy(False)
            # A failed save must retain the editor's selected mode, even when
            # the persisted configuration still has the previous mode.
            app.query_one("#mode", Select).value = "local"
            await pilot.pause()
            app._set_busy(True)
            app._set_busy(False)
            assert app.query_one("#local-settings").display
            assert not app.query_one("#remote-settings").display

    for size in [(80, 24), (120, 40)]:
        asyncio.run(exercise(size))


def test_web_restoration_persists_web_mode_on_linux(tmp_path: Path, monkeypatch) -> None:
    import restore_project as restore

    monkeypatch.setattr(restore.platform, "system", lambda: "Linux")
    monkeypatch.setattr(restore.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install, "prepare_ffmpeg", lambda *_: None)
    args = SimpleNamespace(lean=True, research=False)
    restore.finish_setup(tmp_path, "web", args)
    selected = LauncherConfig.load(tmp_path)
    assert selected.mode == "web" and selected.profile == "lean"
    assert (tmp_path / ".env.local").is_file()
    assert not (tmp_path / ".launcher/worker.token").exists()
