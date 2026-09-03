"""Garage deltas must preserve the scene that is already ready for Drive."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_garage_preview import preview_payload
from test_garage_scene_handoff import HandoffWorker, drive_config, ready_preview, ready_scene

from carla_vision.operator import garage_preview
from carla_vision.operator.garage_preview import GaragePreviewConfig
from carla_vision.operator.world_worker_client import WorldWorkerError


class DeltaWorker(HandoffWorker):
    def __init__(self):
        super().__init__()
        self.configured = []
        self.cameras = []
        self.orbits = []
        self.actual = None
        self.failure = None

    def configure_scene(self, scene, payload):
        self.configured.append(dict(payload))
        updated = replace(
            scene,
            traffic_count=payload["traffic_count"],
            walker_count=payload["walker_count"],
            traffic_count_requested=payload["traffic_count"],
            walker_count_requested=payload["walker_count"],
            route_mode=payload["route_mode"],
            pedestrian_crossing_factor=payload.get("pedestrian_crossing_factor", 0.2),
            speed_difference_percent=payload.get("speed_difference_percent", 12.0),
            following_distance_metres=payload.get("following_distance_metres", 2.0),
        )
        self.actual = {
            "status": "prepared",
            "scene": {
                **asdict(updated),
                "prop_actor_ids": list(updated.prop_actor_ids),
                "cleanup_errors": [],
                "config": {
                    "pedestrian_crossing_factor": 0.2,
                    "speed_difference_percent": 12.0,
                    "following_distance_metres": 2.0,
                    **payload,
                },
            },
        }
        if self.failure:
            raise self.failure
        return updated

    def current_scene(self):
        return self.actual

    def start_camera(self, scene, **kwargs):
        self.cameras.append(kwargs)
        return {"camera": {"actor_id": 56}}

    def orbit_camera(self, scene, **kwargs):
        self.orbits.append(kwargs)


def prepared(worker):
    scene = ready_scene(
        capabilities={
            "prepared_scene_handoff": True,
            "compressed_camera_relay": True,
            "prepared_scene_reconfigure": True,
        }
    )
    return ready_preview(worker, scene=scene)


@pytest.mark.parametrize(
    "changes",
    [
        {"traffic_count": 25},
        {"weather_preset": "wet-day"},
        {"walker_count": 20},
        {"pedestrian_crossing_factor": 1.0},
        {"speed_difference_percent": -10.0},
        {"following_distance_metres": 5.0},
        {"prop_preset": "none"},
        {"route_mode": "random_destination"},
        {"color": "0,0,255"},
        {"vehicle_blueprint": "vehicle.audi.tt"},
        {"traffic_count": 20, "weather_preset": "wet-day"},
    ],
)
def test_scene_deltas_preserve_preview_camera_stream_and_drive_handoff(tmp_path, changes):
    worker = DeltaWorker()
    manager, preview, stream = prepared(worker)
    response = manager.configure(preview_payload(**changes))
    assert response["configure_action"] == "updated"
    assert response["updating"] is False
    assert manager._session is preview
    assert preview._camera_id == 55
    assert preview._stream is stream
    assert not stream.closed
    assert worker.cameras == worker.stopped == worker.prepared == []
    assert len(worker.configured) == 1
    handoff = manager.take_for_drive(drive_config(tmp_path, **changes))
    assert handoff is not None
    assert handoff.scene_id == "preview-scene"
    assert worker.stopped == worker.prepared == []


def test_only_camera_is_replaced_without_waiting_for_first_frame(monkeypatch):
    worker = DeltaWorker()
    manager, preview, old_stream = prepared(worker)
    new_stream = Mock()
    monkeypatch.setattr(garage_preview, "WorldWorkerCameraStream", lambda *_a, **_k: new_stream)
    monkeypatch.setattr(preview, "_start_frame_pump", Mock())
    response = manager.configure(preview_payload(profile="detail", fov=100.0))
    assert response["configure_action"] == "updated"
    assert preview._camera_id == 56
    assert preview._stream is new_stream
    assert old_stream.closed
    assert preview._jpeg == b"ready-frame"
    assert worker.configured == worker.stopped == worker.prepared == []
    assert len(worker.cameras) == 1
    new_stream.wait_for_frame.assert_not_called()
    assert preview.config.width == 1920
    assert response["camera_stale"] is True
    assert response["updating"] is False
    # Sensor sequences restart at zero, browser subscribers must still advance.
    preview._cache_frame(SimpleNamespace(sequence=0, jpeg=b"new-camera"))
    assert preview.wait_for_frame(1, timeout=0) == (2, b"new-camera")
    preview._cache_frame(SimpleNamespace(sequence=1, jpeg=b"next-camera"))
    assert preview.wait_for_frame(2, timeout=0) == (3, b"next-camera")


def test_camera_committed_before_orbit_failure_and_retry_does_not_respawn(monkeypatch):
    worker = DeltaWorker()
    manager, preview, _ = prepared(worker)
    new_stream = Mock()
    monkeypatch.setattr(garage_preview, "WorldWorkerCameraStream", lambda *_a, **_k: new_stream)
    monkeypatch.setattr(preview, "_start_frame_pump", Mock())
    worker.orbit_camera = Mock(side_effect=RuntimeError("orbit failed"))
    with pytest.raises(RuntimeError, match="orbit failed"):
        manager.configure(preview_payload(profile="detail"))
    assert preview.config.profile == "detail"
    assert preview._camera_id == 56
    # Simulate the authoritative readback after the first operation.
    scene = asdict(preview._scene)
    scene.update(
        prop_actor_ids=list(scene["prop_actor_ids"]),
        cleanup_errors=[],
        config={
            **preview.config.scene_payload(),
            "pedestrian_crossing_factor": 0.2,
            "speed_difference_percent": 12.0,
            "following_distance_metres": 2.0,
        },
    )
    worker.actual = {"status": "prepared", "scene": scene}
    worker.orbit_camera = Mock()
    manager.configure(preview_payload(profile="detail"))
    assert len(worker.cameras) == 1
    assert preview.snapshot()["error"] is None


def test_partial_failure_reads_back_actual_settings_and_rejects_stale_handoff(tmp_path):
    worker = DeltaWorker()
    manager, preview, stream = prepared(worker)
    worker.failure = WorldWorkerError("response interrupted after dynamics applied")
    with pytest.raises(WorldWorkerError):
        manager.configure(preview_payload(speed_difference_percent=-10.0))
    assert preview.config.speed_difference_percent == -10.0
    assert preview._configuration_confirmed
    assert preview._updating is False
    assert not stream.closed
    assert worker.stopped == []
    assert manager.take_for_drive(drive_config(tmp_path)) is None


def test_unconfirmed_timeout_cannot_silently_restart_or_handoff(tmp_path):
    worker = DeltaWorker()
    manager, preview, _ = prepared(worker)
    worker.failure = WorldWorkerError("timed out")
    worker.current_scene = Mock(return_value={"status": "preparing"})
    with pytest.raises(WorldWorkerError):
        manager.configure(preview_payload(traffic_count=25))
    assert not preview._configuration_confirmed
    with pytest.raises(RuntimeError, match="still applying"):
        manager.take_for_drive(drive_config(tmp_path))
    with pytest.raises(RuntimeError, match="still applying"):
        manager.configure(preview_payload())
    assert worker.stopped == worker.prepared == []


def test_missing_sensor_after_failed_swap_is_recreated_on_retry(monkeypatch):
    worker = DeltaWorker()
    manager, preview, _ = prepared(worker)
    scene = asdict(preview._scene)
    scene.update(
        prop_actor_ids=[],
        cleanup_errors=[],
        camera=None,
        camera_config=None,
        config={
            **preview.config.scene_payload(),
            "pedestrian_crossing_factor": 0.2,
            "speed_difference_percent": 12.0,
            "following_distance_metres": 2.0,
        },
    )
    worker.actual = {"status": "prepared", "scene": scene}
    worker.start_camera = Mock(
        side_effect=WorldWorkerError("spawn failed after old camera removal")
    )
    with pytest.raises(WorldWorkerError):
        manager.configure(preview_payload(profile="detail"))
    assert preview._camera_id is None
    assert preview._configuration_confirmed
    worker.start_camera = Mock(return_value={"camera": {"actor_id": 57}})
    monkeypatch.setattr(garage_preview, "WorldWorkerCameraStream", lambda *_a, **_k: Mock())
    monkeypatch.setattr(preview, "_start_frame_pump", Mock())
    # Selecting the original profile must still create its missing sensor.
    manager.configure(preview_payload())
    worker.start_camera.assert_called_once()
    assert preview._camera_id == 57
    assert preview.snapshot()["error"] is None
    assert worker.stopped == worker.prepared == []


def test_ego_identity_and_camera_configuration_are_reconciled_after_timeout():
    worker = DeltaWorker()
    _, preview, _ = prepared(worker)
    scene = asdict(preview._scene)
    scene.update(
        ego_actor_id=99,
        prop_actor_ids=[],
        cleanup_errors=[],
        camera={"actor_id": 72},
        camera_config={"mode": "garage", "width": 1920, "height": 1080, "fps": 30.0, "fov": 100.0},
        config={
            **preview.config.scene_payload(),
            "pedestrian_crossing_factor": 0.2,
            "speed_difference_percent": 12.0,
            "following_distance_metres": 2.0,
        },
    )
    worker.actual = {"status": "prepared", "scene": scene}
    preview._configuration_confirmed = False
    preview.ensure_configuration_confirmed()
    assert preview._vehicle_id == 99
    assert preview._camera_id == 72
    assert preview.config.profile == "detail"
    assert preview.config.fov == 100.0
    assert preview._configuration_confirmed


def test_terminal_camera_error_during_apply_does_not_queue_scene_destruction():
    _, preview, _ = prepared(DeltaWorker())
    preview._stream = Mock()
    preview._stream.wait_for_frame.side_effect = WorldWorkerError("transport exhausted retries")
    preview._updating = True
    preview.close = Mock()
    preview._frame_loop()
    preview.close.assert_not_called()
    assert preview._status == "running"
    assert preview._reader_needs_restart


@pytest.mark.parametrize("changes", [{"map_name": "Town05"}, {"seed": 9}])
def test_map_and_seed_still_require_a_new_scene(changes):
    _, preview, _ = prepared(DeltaWorker())
    assert not preview.can_reconfigure(GaragePreviewConfig.from_mapping(preview_payload(**changes)))


def test_old_worker_does_not_receive_new_scene_endpoint():
    worker = HandoffWorker()
    _, preview, _ = ready_preview(worker)
    assert not preview.can_reconfigure(
        GaragePreviewConfig.from_mapping(preview_payload(traffic_count=20))
    )
