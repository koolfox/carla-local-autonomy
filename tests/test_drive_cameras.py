"""Drive rig tests use fake CARLA actors, not a live simulator."""

import json
import shutil
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import test_world_worker as fakes

from carla_vision.native.world_worker import WorkerError, WorldWorker
from carla_vision.operator import drive_cameras
from carla_vision.operator.configuration import build_legacy_drive_request, session_defaults


def rig():
    return {
        "schema_version": "1.0",
        "primary_view": {
            "mount": {"x": 1.2, "y": 0, "z": 2, "pitch": -5, "yaw": 0, "roll": 2},
            "fov_degrees": 85,
        },
        "additional_views": [
            {
                "id": "rear",
                "mount": {
                    "x": -1.5,
                    "y": 0,
                    "z": 2,
                    "pitch": -8,
                    "yaw": 180,
                    "roll": 0,
                },
                "fov_degrees": 110,
            }
        ],
    }


def test_session_adapter_retains_rig_without_changing_control_or_main_camera(tmp_path):
    session = session_defaults(detector_enabled=False, worker_configured=True)
    session["identity"]["runId"] = "drive-rig-test"
    session["vehicle"]["blueprint"] = "vehicle.tesla.model3"
    session["recording"].update(cameraRig=rig(), cameraRigFps=5)
    raw = build_legacy_drive_request(
        session,
        carla_host="localhost",
        carla_port=2000,
        worker_connected=True,
        capabilities={"autopilot": True},
    )
    # Garage-only fields are stripped by the same production adapter.
    from carla_vision.operator.garage_drive import GarageDriveStartConfig

    config = GarageDriveStartConfig.from_mapping(
        raw,
        workspace=tmp_path,
        expected_host="localhost",
        expected_port=2000,
        world_worker_configured=True,
    )
    assert config.recording_rig == rig()
    assert config.initial_control_mode == "autopilot"
    assert config.camera_fov == session["camera"]["fov"]
    assert config.manifest_config()["recording_rig_fps"] == 5


@pytest.mark.parametrize(
    "modify",
    [
        lambda value: value["additional_views"].append(value["additional_views"][0]),
        lambda value: value["primary_view"]["mount"].update(yaw=float("nan")),
        lambda value: value["additional_views"][0].update(id="../escape"),
        lambda value: value["primary_view"].update(fov_degrees=180),
        lambda value: value["primary_view"]["mount"].update(x=101),
    ],
)
def test_invalid_rig_rejected_before_native_work(modify):
    value = rig()
    modify(value)
    with pytest.raises((ValueError, TypeError)):
        drive_cameras.recording_views(value, width=640, height=384, fps=5, fov=90)


@pytest.fixture
def native(monkeypatch):
    library = fakes.FakeBlueprintLibrary()
    library.blueprints["sensor.camera.rgb"] = fakes.FakeBlueprint(
        "sensor.camera.rgb",
        {
            key: fakes.FakeAttribute("")
            for key in ("role_name", "fov", "image_size_x", "image_size_y", "sensor_tick")
        },
    )
    world = fakes.FakeWorld("Town10HD_Opt", library)
    client = fakes.FakeClient(world, library, fakes.FakeTrafficManager(8000))
    worker = WorldWorker(
        carla_loader=lambda: fakes.FakeCarla(client),
        route_planner_loader=lambda: None,
        start_monitor=False,
    )
    attached = []

    class Sensor(fakes.FakeActor):
        def listen(self, callback):
            self.callback = callback

    def spawn(blueprint, transform, attach_to=None):
        sensor = Sensor(world, world.next_actor_id, blueprint, transform)
        world.next_actor_id += 1
        world.actors[sensor.id] = sensor
        attached.append((sensor, attach_to))
        return sensor

    monkeypatch.setattr(world, "spawn_actor", spawn, raising=False)
    monkeypatch.setattr(
        fakes.FakeTransform, "get_matrix", lambda self: np.eye(4).tolist(), raising=False
    )
    monkeypatch.setattr(
        "carla_vision.native.world_worker._load_in_memory_jpeg_encoder",
        lambda: lambda image, quality: b"jpeg",
    )
    prepared = worker.prepare({})
    scene = prepared["scene"]
    request = {
        "lease_token": scene["lease_token"],
        "width": 640,
        "height": 384,
        "fps": 5,
        "views": drive_cameras.recording_views(rig(), width=640, height=384, fps=5, fov=90),
    }
    yield worker, world, attached, scene, request
    worker.close()


def test_native_rig_attaches_angles_to_same_ego_and_normal_stop_cleans_it(native):
    worker, world, attached, scene, request = native
    initial_ids = set(world.actors)
    response = worker.recording_cameras(scene["scene_id"], request)
    assert len(attached) == 2
    front, rear = [sensor for sensor, _ in attached]
    assert front.transform.rotation.pitch == -5
    assert rear.transform.rotation.yaw == 180
    assert rear.attributes["fov"] == "110.0"
    assert all(ego.id in initial_ids for _, ego in attached)
    assert worker._scene.camera_relay is None  # Existing live camera is untouched.
    assert set(response["camera_to_ego"]) == {"front", "rear"}
    owned_scene = worker._scene
    worker.stop(scene["scene_id"], {"lease_token": scene["lease_token"]})
    assert not world.actors
    assert not owned_scene.recording_relays


def test_partial_spawn_failure_keeps_sensors_owned_for_normal_stop(native, monkeypatch):
    worker, world, attached, scene, request = native
    original = world.spawn_actor

    def fail_second(*args, **kwargs):
        if attached:
            raise RuntimeError("GPU refused second camera")
        return original(*args, **kwargs)

    monkeypatch.setattr(world, "spawn_actor", fail_second)
    with pytest.raises(RuntimeError, match="second camera"):
        worker.recording_cameras(scene["scene_id"], request)
    assert len(worker._scene.recording_relays) == 1
    worker.stop(scene["scene_id"], {"lease_token": scene["lease_token"]})
    assert not world.actors


def test_rig_reconfiguration_and_unknown_view_are_rejected(native):
    worker, _, _, scene, request = native
    worker.recording_cameras(scene["scene_id"], request)
    with pytest.raises(WorkerError, match="before Drive"):
        worker.recording_cameras(scene["scene_id"], request)
    with pytest.raises(WorkerError, match="not active"):
        worker.camera_frame(
            scene["scene_id"], scene["lease_token"], view="unknown", after_sequence=-1, timeout=0.1
        )


def test_each_video_index_matches_written_frames_and_reports_gaps(tmp_path, monkeypatch):
    finished = threading.Event()

    class Stream:
        def __init__(self, *args, view, **kwargs):
            self.closed = False

        def wait_for_frame(self, after_sequence, timeout):
            if after_sequence >= 3:
                finished.set()
                while not self.closed:
                    time.sleep(0.001)
                raise RuntimeError("closed")
            seq = 1 if after_sequence == -1 else 3
            return SimpleNamespace(
                sequence=seq,
                frame=100 + seq,
                timestamp=seq / 5,
                transform=[0] * 6,
                bgr=lambda: np.zeros((180, 320, 3), np.uint8),
            )

        def close(self):
            self.closed = True

    class Writer:
        def __init__(self, path, size, fps):
            self.images = []

        def write(self, image):
            self.images.append(image)

        def release(self):
            pass

    monkeypatch.setattr(drive_cameras, "WorldWorkerCameraStream", Stream)
    monkeypatch.setattr(drive_cameras, "BrowserVideoWriter", Writer)
    view = drive_cameras._ViewRecording(None, None, "rear", tmp_path, 320, 180, 5)
    assert finished.wait(2)
    result = view.close()
    rows = [json.loads(line) for line in view.index.read_text().splitlines()]
    assert [row["carla_frame"] for row in rows] == [101, 103]
    assert [row["video_frame"] for row in rows] == [0, 1]
    assert len(view.writer.images) == result["frames_written"] == 2
    assert result["relay_frames_skipped"] == 1
    assert result["error"] is None


def test_failed_capture_sync_mode_is_reset_by_normal_garage_prepare(native, monkeypatch):
    worker, world, _, scene, _ = native
    worker.stop(scene["scene_id"], {"lease_token": scene["lease_token"]})
    worker.research_reset_pending = True
    world.settings.synchronous_mode = True
    world.settings.fixed_delta_seconds = 0.1
    monkeypatch.setattr(world, "apply_settings", lambda value: None, raising=False)
    assert worker.prepare({})["scene"]["status"] == "prepared"
    assert not world.settings.synchronous_mode
    assert world.settings.fixed_delta_seconds is None
    assert not worker.research_reset_pending


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires recording encoder")
def test_http_rig_records_two_real_mp4s_without_reloading_world(native, tmp_path, monkeypatch):
    import cv2

    from carla_vision.native.world_worker import create_server
    from carla_vision.operator.world_worker_client import WorldWorkerClient, WorldWorkerScene

    worker, world, attached, _, _ = native
    token = "drive-camera-http-test"
    server = create_server(bind="127.0.0.1", port=0, token=token, worker=worker)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = WorldWorkerClient(f"http://127.0.0.1:{server.server_port}", token)
    scene = WorldWorkerScene.from_response(worker.current_scene())
    monkeypatch.setattr(
        "carla_vision.native.world_worker._load_in_memory_jpeg_encoder",
        lambda: lambda image, quality: image.jpeg,
    )
    recording = drive_cameras.DriveCameraRecording(tmp_path / "cameras")
    done = threading.Event()
    config = SimpleNamespace(
        recording_rig=rig(), width=640, height=384, recording_rig_fps=5, camera_fov=90
    )
    artifacts = []
    tracker = SimpleNamespace(
        register_artifact=lambda path, **kwargs: artifacts.append((path, kwargs))
    )
    publisher = None
    try:
        recording.start(client, scene, config)
        jpeg = cv2.imencode(".jpg", np.zeros((384, 640, 3), np.uint8))[1].tobytes()

        def publish():
            frame = 100
            while not done.wait(0.05):
                frame += 1
                for sensor, _ in attached:
                    sensor.callback(
                        SimpleNamespace(
                            width=640,
                            height=384,
                            frame=frame,
                            timestamp=frame / 20,
                            fov=float(sensor.attributes["fov"]),
                            transform=sensor.transform,
                            jpeg=jpeg,
                        )
                    )

        publisher = threading.Thread(target=publish, daemon=True)
        publisher.start()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and any(view.written < 3 for view in recording.views):
            recording.check_health()
            time.sleep(0.05)
        assert all(view.written >= 3 for view in recording.views)
        assert recording.close(tracker) == []
        for view in recording.views:
            video = cv2.VideoCapture(str(view.video))
            try:
                assert video.isOpened()
                count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                rows = view.index.read_text().splitlines()
                assert count == len(rows) >= 3
            finally:
                video.release()
        metadata = json.loads((recording.root / "rig.json").read_text())
        assert metadata["synchronized"] is False
        assert metadata["views"]["rear"]["mount"]["yaw"] == 180
        assert world is worker._scene.world
        assert len(artifacts) == 5  # Rig metadata, two videos, two indexes.
    finally:
        done.set()
        if publisher:
            publisher.join(timeout=1)
        recording.close(SimpleNamespace(register_artifact=lambda *args, **kwargs: None))
        client.stop_scene(scene)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)
