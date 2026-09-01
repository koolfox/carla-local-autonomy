from __future__ import annotations

from pathlib import Path

import pytest

from carla_vision.operator import server as base
from carla_vision.operator.local_entrypoint import (
    load_local_env,
    prepare_launch,
    render_operator_index,
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
    assert plan.detector_enabled is False
    assert args.carla_host == "auto"
    assert args.carla_port == 2000
    assert args.port == 8765


def test_worker_token_automatically_tracks_the_resolved_carla_host(tmp_path: Path) -> None:
    write_env(
        tmp_path,
        "CARLA_WORLD_WORKER_TOKEN=abcdefghijklmnopqrstuvwxyz123456\n",
    )

    plan = prepare_launch([], cwd=tmp_path, environ={})
    args = base.parse_args(plan.argv)

    assert args.carla_host == "auto"
    assert args.world_worker_url == "auto"
    assert plan.env_updates["CARLA_WORLD_WORKER_TOKEN"] == "abcdefghijklmnopqrstuvwxyz123456"
