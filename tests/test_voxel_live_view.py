from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from carla_vision.operator.drive import DriveSession
from carla_vision.operator.drive_contracts import DriveStartConfig
from carla_vision.voxel.contracts import VoxelGridSpec
from carla_vision.voxel.live_view import VoxelViewWorker, render_voxel_view


def result():
    spec = VoxelGridSpec(x_max=10, y_min=-5, y_max=5, z_min=-2, z_max=3)
    occupancy = np.full(spec.shape, -1, dtype=np.int8)
    occupancy[4, 10, 8] = 1
    return SimpleNamespace(spec=spec, occupancy=occupancy, metadata={"grid": spec.as_dict()})


def frame(sequence):
    return SimpleNamespace(sequence=sequence, frame=sequence + 100, timestamp=sequence / 10,
                           received_monotonic=time.monotonic(), fov=90,
                           bgr=lambda: np.full((120, 160, 3), sequence, dtype=np.uint8))


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_render_contains_exact_source_and_real_occupancy():
    source = np.full((120, 160, 3), (15, 50, 75), dtype=np.uint8)
    occupied = result()
    canvas = render_voxel_view(source, occupied, frame=101)
    assert canvas.shape == (720, 1280, 3)
    assert tuple(canvas[120, 50]) == (15, 50, 75)
    empty = result()
    empty.occupancy[:] = -1
    assert np.any(canvas != render_voxel_view(source, empty, frame=101))


def test_loading_is_background_and_uses_only_latest_rgb():
    gate = threading.Event()
    seen = []

    class Predictor:
        def load(self):
            gate.wait(3)

        def predict(self, rgb, fov, **metadata):
            seen.append((rgb.copy(), fov, metadata))
            return result()

    worker = VoxelViewWorker(device="cpu", predictor_factory=Predictor, max_fps=10)
    try:
        worker.submit(frame(1))
        worker.submit(frame(2))
        worker.submit(frame(3))
        assert worker.snapshot()["status"] == "loading"
        assert worker.latest() is None
        gate.set()
        wait_until(lambda: worker.latest() is not None)
        latest = worker.latest()
        assert latest.sequence == 3 and latest.source_frame == 103
        assert latest.record()["actuated"] is False
        assert seen[0][2] == {"frame": 103, "timestamp": 0.3, "sequence": 3}
        assert np.all(seen[0][0] == 3)
        assert worker.snapshot()["dropped"] == 2
        assert cv2.imdecode(np.frombuffer(latest.jpeg, np.uint8), 1).shape == (720, 1280, 3)
    finally:
        gate.set()
        worker.close()


def test_failure_is_status_not_control_exception():
    class Broken:
        def load(self):
            raise RuntimeError("test missing weights")

    worker = VoxelViewWorker(device="cpu", predictor_factory=Broken)
    try:
        wait_until(lambda: worker.snapshot()["status"] == "failed")
        worker.submit(frame(1))
        assert worker.latest() is None
        assert "missing weights" in worker.snapshot()["error"]
        assert worker.snapshot()["actuated"] is False
    finally:
        worker.close()


def test_stop_during_inference_never_publishes_late_result():
    entered, release = threading.Event(), threading.Event()

    class Blocking:
        def load(self):
            pass

        def predict(self, *_args, **_kwargs):
            entered.set()
            release.wait(3)
            return result()

    worker = VoxelViewWorker(device="cpu", predictor_factory=Blocking)
    worker.submit(frame(1))
    assert entered.wait(2)
    worker.close()
    release.set()
    worker._thread.join(3)
    assert not worker._thread.is_alive()
    assert worker.latest() is None
    assert worker.snapshot()["status"] == "stopped"


def test_drive_voxel_cache_does_not_replace_raw_or_overlay(tmp_path):
    config = DriveStartConfig(
        run_id="voxel-test", host="127.0.0.1", port=2000,
        vehicle_blueprint="vehicle.audi.tt", color=None, seed=7,
        weather_preset="keep", prop_preset="none", detector_enabled=False,
        detector="rtdetr", weights=None, device="cpu", image_size=640, confidence=0.2,
        width=1280, height=720, camera_fps=30, camera_fov=90,
        record_video=False, spectator_follow=False,
    )
    session = DriveSession(replace(config, voxel_enabled=True), workspace=tmp_path)
    session._cache_frame("raw", 80, b"raw")
    session._cache_frame("overlay", 70, b"overlay")
    session._cache_frame("voxel", 60, b"voxel")
    assert session.frame("raw") == (80, b"raw")
    assert session.frame("overlay") == (70, b"overlay")
    assert session.wait_for_frame("voxel", 59, timeout=0.1) == (60, b"voxel")
    assert session.snapshot()["voxel"]["actuated"] is False
