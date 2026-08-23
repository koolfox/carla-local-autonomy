from __future__ import annotations

import inspect
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from carla_vision.native.world_worker import (
    CompressedCameraConfig,
    CompressedCameraRelay,
    WorkerError,
)
from carla_vision.operator.garage_preview import GaragePreviewConfig

ROOT = Path(__file__).parents[1]
STATIC_ROOT = ROOT / "carla_vision" / "operator" / "static"


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.elements[element_id] = (tag, attributes)


def _camera_payload(fps: float) -> dict[str, object]:
    return {
        "lease_token": "test-lease",
        "mode": "drive",
        "width": 1280,
        "height": 720,
        "fps": fps,
        "fov": 90.0,
    }


def _garage_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "map_name": "Town10HD_Opt",
        "weather_preset": "clear-day",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "255,0,0",
        "seed": 20260809,
        "traffic_count": 0,
        "walker_count": 0,
        "prop_preset": "none",
    }
    payload.update(overrides)
    return payload


def test_worker_camera_is_bounded_at_sixty_and_encodes_without_disk_io() -> None:
    assert CompressedCameraConfig.from_mapping(_camera_payload(60.0)).fps == 60.0
    with pytest.raises(WorkerError, match=r"\[1\.0, 60\.0\]"):
        CompressedCameraConfig.from_mapping(_camera_payload(60.01))

    worker_source = (ROOT / "carla_vision" / "native" / "world_worker.py").read_text(
        encoding="utf-8"
    )
    relay_source = inspect.getsource(CompressedCameraRelay)
    assert "save_to_disk" not in relay_source
    assert "tempfile" not in worker_source
    assert 'import_module("numpy")' in worker_source
    assert 'import_module("cv2")' in worker_source
    assert "cv2.imencode" in worker_source
    assert "_pending_image" in relay_source


def test_persistent_mjpeg_endpoints_exist_at_every_remote_hop() -> None:
    worker = (ROOT / "carla_vision" / "native" / "world_worker.py").read_text(encoding="utf-8")
    client = (ROOT / "carla_vision" / "operator" / "world_worker_client.py").read_text(
        encoding="utf-8"
    )
    operator = (ROOT / "carla_vision" / "operator" / "server.py").read_text(encoding="utf-8")
    garage = (ROOT / "carla_vision" / "operator" / "garage_server.py").read_text(encoding="utf-8")

    assert "camera/stream\\.mjpg" in worker
    assert '"persistent_mjpeg_camera_relay": in_memory_encoder' in worker
    assert '"camera_60_fps": in_memory_encoder' in worker
    assert '"in_memory_jpeg_encoder_available": in_memory_encoder' in worker
    assert "/camera/stream.mjpg" in client
    assert "multipart/x-mixed-replace" in worker
    assert "/api/drive/stream.mjpg" in operator
    assert "multipart/x-mixed-replace" in operator
    assert "/api/garage/preview/stream.mjpg" in garage
    assert "multipart/x-mixed-replace" in garage


def test_browser_defaults_to_720p30_but_allows_explicit_720p60() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    parser = _DocumentParser()
    parser.feed(html)

    fps_tag, fps = parser.elements["drive-camera-fps"]
    assert fps_tag == "input"
    assert fps["value"] == "30"
    assert fps["max"] == "60"

    resolution = re.search(
        r'<select\s+id="drive-resolution"[^>]*>(?P<body>.*?)</select>',
        html,
        re.DOTALL,
    )
    assert resolution is not None
    assert re.search(r'<option\s+value="1280x720"\s+selected>', resolution.group("body"))

    profiles = re.search(
        r'<select\s+id="drive-camera-profile"[^>]*>(?P<body>.*?)</select>',
        html,
        re.DOTALL,
    )
    assert profiles is not None
    selected_profile = re.search(
        r"<option[^>]*selected[^>]*>(?P<label>.*?)</option>",
        profiles.group("body"),
        re.DOTALL,
    )
    assert selected_profile is not None
    assert "30 FPS" in selected_profile.group("label")
    assert "60 FPS" in profiles.group("body")
    _, detector = parser.elements["drive-detector-enabled"]
    assert "checked" not in detector


def test_garage_uses_the_same_bounded_camera_profiles() -> None:
    default = GaragePreviewConfig.from_mapping(_garage_payload())
    assert (default.width, default.height, default.fps) == (1280, 720, 30.0)

    high_refresh = GaragePreviewConfig.from_mapping(_garage_payload(profile="high-refresh"))
    assert (high_refresh.width, high_refresh.height, high_refresh.fps) == (
        1280,
        720,
        60.0,
    )

    with pytest.raises(ValueError, match="profile"):
        GaragePreviewConfig.from_mapping(_garage_payload(profile="unbounded"))

    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    payload_source = script.split("function garagePreviewPayload()", maxsplit=1)[1].split(
        "function garagePayloadKey", maxsplit=1
    )[0]
    assert 'profile: $("drive-camera-profile").value' in payload_source


def test_browser_uses_persistent_streams_instead_of_per_frame_polling() -> None:
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "/api/garage/preview/stream.mjpg" in script
    assert "/api/drive/stream.mjpg" in script
    assert "/api/drive/frame.jpg" not in script
    assert "function refreshDriveFrame(" not in script


def test_high_bandwidth_profiles_cannot_fall_back_to_raw_bgra() -> None:
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    drive = (ROOT / "carla_vision" / "operator" / "drive.py").read_text(encoding="utf-8")
    garage = (ROOT / "carla_vision" / "operator" / "garage_preview.py").read_text(encoding="utf-8")

    assert 'select.value = "compatibility"' in script
    assert "_MAX_RAW_CAMERA_BYTES_PER_SECOND" in drive
    assert "selected video profile requires the persistent compressed" in drive
    assert 'self.config.profile != "compatibility"' in garage
    assert "selected Garage video profile requires the persistent" in garage
