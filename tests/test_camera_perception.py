from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_perception import BlockingDetector, make_frame

from carla_vision.contracts import Detection
from carla_vision.operator.camera_perception import CameraPerception
from carla_vision.perception import PerceptionWorker


class ControlledDetector(BlockingDetector):
    detection_only_name = "without-signs"

    def __init__(self):
        super().__init__()
        self.calls = queue.Queue()
        self.advance = threading.Semaphore(0)
        self.close_count = 0

    def infer(self, image):
        return self._infer(image, True)

    def infer_without_signs(self, image):
        return self._infer(image, False)

    def _infer(self, image, signs):
        value = int(image[0, 0, 0])
        self.calls.put((value, signs))
        assert self.advance.acquire(timeout=3)
        if value == 99:
            raise RuntimeError("bad rear input")
        return (
            Detection(0, "pixel", 1, (0, 0, 2, 2), attributes={"value": value, "signs": signs}),
        )

    def close(self):
        self.close_count += 1


def test_shared_model_fair_turns_latest_only_and_independent_camera_sequences():
    detector = ControlledDetector()
    worker = PerceptionWorker(
        detector,
        camera_ids=("front", "rig:rear", "rig:left"),
        sign_cameras=frozenset({"front", "rig:left"}),
    )
    try:
        worker.submit(make_frame(1))
        assert detector.calls.get(timeout=1) == (1, True)
        worker.submit(replace(make_frame(11), sequence=1), camera_id="rig:rear")
        worker.submit(replace(make_frame(21), sequence=1), camera_id="rig:left")
        worker.submit(make_frame(2))
        worker.submit(make_frame(3))
        detector.advance.release()
        assert detector.calls.get(timeout=1) == (11, False)
        # Flood front while rear is executing; front still cannot overtake left.
        for sequence in range(4, 9):
            worker.submit(make_frame(sequence))
        detector.advance.release()
        assert detector.calls.get(timeout=1) == (21, True)
        detector.advance.release()
        assert detector.calls.get(timeout=1) == (8, True)
        detector.advance.release()
        front = worker.wait_for_result(7, timeout=1)
        rear = worker.wait_for_result(camera_id="rig:rear", timeout=1)
        left = worker.wait_for_result(camera_id="rig:left", timeout=1)
        assert front.sequence == 8
        assert rear.sequence == left.sequence == 1
        assert rear.source_bgr[0, 0, 0] == 11 and left.source_bgr[0, 0, 0] == 21
        assert rear.detector_name == "without-signs"
        assert worker.stats(camera_id="front").dropped_before_inference == 6
        assert worker.stats().processed == 4
        with pytest.raises(ValueError, match="Unknown"):
            worker.submit(make_frame(1), camera_id="unregistered")
    finally:
        detector.advance.release(10)
        worker.close()
        worker.close()
    assert detector.close_count == 1


def test_camera_inference_failure_does_not_poison_other_cameras():
    detector = ControlledDetector()
    worker = PerceptionWorker(detector, camera_ids=("front", "rig:rear"))
    try:
        worker.submit(make_frame(99), camera_id="rig:rear")
        detector.advance.release()
        with pytest.raises(RuntimeError, match="bad rear input"):
            worker.wait_for_result(camera_id="rig:rear", timeout=1)
        worker.submit(make_frame(2))
        detector.advance.release()
        assert worker.wait_for_result(timeout=1).carla_frame == 102
    finally:
        detector.advance.release(10)
        worker.close()


def test_overlay_video_and_json_are_the_exact_camera_result(tmp_path, monkeypatch):
    from carla_vision.operator import camera_perception

    written = []

    class Writer:
        def __init__(self, path, *args):
            path.touch()

        def write(self, image):
            written.append(image.copy())

        def release(self):
            pass

    monkeypatch.setattr(camera_perception, "BrowserVideoWriter", Writer)
    detector = BlockingDetector()
    detector.release.set()
    worker = PerceptionWorker(detector, camera_ids=("rig:rear",))
    published = []
    observer = CameraPerception(
        worker,
        "rear",
        tmp_path,
        width=6,
        height=4,
        fps=5,
        publish=lambda *args: published.append(args),
        hud={"MODEL": "test"},
    )
    try:
        observer.submit(make_frame(7))
        deadline = time.monotonic() + 2
        while not observer.written and time.monotonic() < deadline:
            time.sleep(0.005)
        result = observer.close()
        assert result["error"] is None and result["frames_written"] == 1
        row = json.loads(observer.index.read_text())
        assert row["camera_id"] == "rear" and row["carla_frame"] == 107
        assert row["video_frame"] == 0 and row["sequence"] == 7
        assert row["detections"][0]["attributes"]["value"] == 7
        assert published[0][:2] == ("rig:rear:overlay", 7)
        assert written[0].shape == (4, 6, 3)
    finally:
        observer.close()
        worker.close()


def test_front_and_rig_cache_use_separate_namespaces(tmp_path):
    from test_drive_cameras import rig
    from test_drive_console import CARLA_HOST, CARLA_PORT, valid_start

    from carla_vision.operator.drive import DriveSession
    from carla_vision.operator.drive_contracts import DriveStartConfig

    (tmp_path / "detector.pt").touch()
    config = DriveStartConfig.from_mapping(
        valid_start(
            weights="detector.pt", recording_rig=rig(), recording_perception={"rear": "detections"}
        ),
        workspace=tmp_path,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
        world_worker_configured=True,
    )
    session = DriveSession(config, workspace=tmp_path, world_worker=SimpleNamespace())
    session._cache_frame("raw", 1, b"front")
    session._cache_frame("rig:rear:raw", 1, b"rear")
    session._cache_frame("rig:rear:overlay", 1, b"annotated rear")
    assert session.frame("raw") == (1, b"front")
    assert session.wait_for_frame("rig:rear:overlay", timeout=0.01) == (1, b"annotated rear")
    assert session.frame("rig:rear:raw") == (1, b"rear")
    with pytest.raises(ValueError):
        session.frame("rig:unknown:raw")
    with pytest.raises(ValueError):
        session.frame("rig:front:overlay")  # No perception requested for rig front.
    assert config.manifest_config()["runtime_sensor_contract"] == "rgb_only_selected_cameras"


@pytest.mark.parametrize(
    "modes",
    [{"missing": "detections"}, {"rear": "invalid"}, {"rear": "signs"}, {"rear": []}, [], "", 1],
)
def test_invalid_camera_selection_fails_before_spawning(tmp_path, modes):
    from test_drive_cameras import rig
    from test_drive_console import CARLA_HOST, CARLA_PORT, valid_start

    from carla_vision.operator.drive_contracts import DriveStartConfig

    (tmp_path / "detector.pt").touch()
    with pytest.raises(ValueError):
        DriveStartConfig.from_mapping(
            valid_start(weights="detector.pt", recording_rig=rig(), recording_perception=modes),
            workspace=tmp_path,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
            world_worker_configured=True,
        )


def test_rig_observer_error_does_not_stop_raw_recording(tmp_path, monkeypatch):
    from carla_vision.operator import drive_cameras

    class Stream:
        def __init__(self, *args, **kwargs):
            self.closed = threading.Event()

        def wait_for_frame(self, after_sequence, timeout):
            if after_sequence >= 2:
                self.closed.wait(2)
                raise EOFError("done")
            return make_frame(after_sequence + 2)

        def close(self):
            self.closed.set()

    monkeypatch.setattr(drive_cameras, "WorldWorkerCameraStream", Stream)
    monkeypatch.setattr(
        drive_cameras,
        "BrowserVideoWriter",
        lambda *args: SimpleNamespace(write=lambda image: None, release=lambda: None),
    )
    # submit handles the failed observer internally; raw frame writing continues.
    observer = CameraPerception.__new__(CameraPerception)
    observer.stop = threading.Event()
    observer.error = None
    observer.camera_id = "rig:rear"
    observer.worker = SimpleNamespace(
        submit=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model failed"))
    )
    view = drive_cameras._ViewRecording(
        None, None, "rear", tmp_path, 6, 4, 5, on_frame=observer.submit
    )
    try:
        deadline = time.monotonic() + 1
        while view.written < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert view.written == 2 and view.error is None
        assert observer.error == "model failed"
    finally:
        view.close()
