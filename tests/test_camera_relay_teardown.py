from __future__ import annotations

import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from test_world_worker import (
    FakeActor,
    FakeAttribute,
    FakeBlueprint,
    FakeBlueprintLibrary,
    FakeCarla,
    FakeClient,
    FakeTrafficManager,
    FakeTransform,
    FakeWorld,
)

from carla_vision.native.world_worker import CompressedCameraRelay, WorkerError, WorldWorker


class Sensor:
    id = 71

    def __init__(self) -> None:
        self.callback: Any = None
        self.stop_calls = 0
        self.stopped = threading.Event()

    def listen(self, callback: Any) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


def image(frame: int = 1) -> SimpleNamespace:
    return SimpleNamespace(frame=frame, timestamp=0.1, width=320, height=180, fov=90.0)


def test_close_drains_native_callbacks_after_stop_before_destroy() -> None:
    sensor = Sensor()
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay.listen()
    events: list[str] = []

    def drain(seconds: float) -> None:
        assert sensor.stopped.is_set()
        assert 0.0 < seconds <= 0.4
        assert relay._encoder_thread is not None
        assert not relay._encoder_thread.is_alive()
        # A queued native delivery may still arrive after stop detached the
        # listener. The relay must reject it throughout the drain interval.
        sensor.callback(image())
        events.append("drained")

    with mock.patch("carla_vision.native.world_worker.time.sleep", side_effect=drain):
        relay.close()
        events.append("destroy")
        relay.close()

    assert events == ["drained", "destroy"]
    assert sensor.stop_calls == 1
    assert relay.snapshot()["telemetry"]["frames_received"] == 0


@pytest.mark.parametrize("encode_fails", [False, True])
def test_close_waits_for_encoder_and_discards_pending_and_late_frames(encode_fails: bool) -> None:
    sensor = Sensor()
    encoding = threading.Event()
    release_encoder = threading.Event()
    close_finished = threading.Event()
    failures: list[BaseException] = []
    encoded_frames: list[int] = []

    def encode(frame: Any, quality: int) -> bytes:
        del quality
        encoded_frames.append(frame.frame)
        encoding.set()
        assert release_encoder.wait(2.0)
        if encode_fails:
            raise RuntimeError("encoder failed during teardown")
        return b"jpeg"

    relay = CompressedCameraRelay(sensor, jpeg_encoder=encode)
    relay._STOP_DRAIN_SECONDS = 0.0
    relay.listen()
    sensor.callback(image(1))
    assert encoding.wait(1.0)
    sensor.callback(image(2))

    def close_then_destroy() -> None:
        try:
            relay.close()
            # Returning from close permits the owner to destroy the sensor.
            assert relay._encoder_thread is not None
            assert not relay._encoder_thread.is_alive()
            close_finished.set()
        except BaseException as error:
            failures.append(error)

    closer = threading.Thread(target=close_then_destroy)
    closer.start()
    try:
        assert sensor.stopped.wait(1.0)
        sensor.callback(image(3))
        assert not close_finished.wait(0.02)
    finally:
        release_encoder.set()
        closer.join(timeout=2.0)

    assert not closer.is_alive()
    assert not failures
    assert close_finished.is_set()
    assert encoded_frames == [1]
    snapshot = relay.snapshot()
    assert snapshot["telemetry"]["frames_received"] == 2
    assert snapshot["telemetry"]["frames_encoded"] == 0
    assert snapshot["error"] is None
    assert relay._pending_image is None
    with pytest.raises(WorkerError, match="not active"):
        relay.wait(-1, 0.01)


def test_concurrent_close_callers_both_wait_for_completed_drain() -> None:
    sensor = Sensor()
    drain_started = threading.Event()
    release_drain = threading.Event()
    second_started = threading.Event()
    first_finished = threading.Event()
    second_finished = threading.Event()
    failures: list[BaseException] = []
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay.listen()

    def drain(seconds: float) -> None:
        del seconds
        drain_started.set()
        assert release_drain.wait(2.0)

    def close(finished: threading.Event, started: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            relay.close()
            finished.set()
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=close, args=(first_finished,))
    second = threading.Thread(target=close, args=(second_finished, second_started))
    with mock.patch("carla_vision.native.world_worker.time.sleep", side_effect=drain):
        first.start()
        try:
            assert drain_started.wait(1.0)
            second.start()
            assert second_started.wait(1.0)
            assert not second_finished.wait(0.02)
            assert not first_finished.is_set()
        finally:
            release_drain.set()
            first.join(timeout=2.0)
            if second.ident is not None:
                second.join(timeout=2.0)

    assert not failures
    assert first_finished.is_set()
    assert second_finished.is_set()
    assert sensor.stop_calls == 1


def test_encoder_timeout_does_not_permit_destroy_and_close_is_retryable() -> None:
    sensor = Sensor()
    encoding = threading.Event()
    release_encoder = threading.Event()
    destroyed = False

    def encode(frame: Any, quality: int) -> bytes:
        del frame, quality
        encoding.set()
        assert release_encoder.wait(2.0)
        return b"jpeg"

    relay = CompressedCameraRelay(sensor, jpeg_encoder=encode)
    relay._STOP_DRAIN_SECONDS = 0.0
    relay._ENCODER_JOIN_TIMEOUT_SECONDS = 0.01
    relay.listen()
    sensor.callback(image())
    assert encoding.wait(1.0)
    try:
        with pytest.raises(RuntimeError, match="teardown timeout"):
            relay.close()
            destroyed = True
        assert not destroyed
        assert not relay._close_complete
        sensor.callback(image(2))
        assert relay.snapshot()["telemetry"]["frames_received"] == 1
    finally:
        release_encoder.set()
        assert relay._encoder_thread is not None
        relay._encoder_thread.join(timeout=2.0)
        relay.close()

    assert relay._close_complete
    assert sensor.stop_calls == 1


def test_stop_failure_does_not_permit_destroy_and_is_retried() -> None:
    sensor = Sensor()
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay._STOP_DRAIN_SECONDS = 0.0
    relay.listen()
    with mock.patch.object(sensor, "stop", side_effect=RuntimeError("detach failed")):
        with pytest.raises(RuntimeError, match="detach failed"):
            relay.close()
    assert not relay._close_complete
    sensor.callback(image())
    assert relay.snapshot()["telemetry"]["frames_received"] == 0

    relay.close()
    assert relay._close_complete
    assert sensor.stop_calls == 1
    assert relay._encoder_thread is not None
    assert not relay._encoder_thread.is_alive()


def test_already_destroyed_sensor_still_joins_encoder_and_drains_callbacks() -> None:
    sensor = Sensor()
    encoding = threading.Event()
    release_encoder = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    failures: list[BaseException] = []

    def encode(frame: Any, quality: int) -> bytes:
        del frame, quality
        encoding.set()
        assert release_encoder.wait(2.0)
        return b"jpeg"

    relay = CompressedCameraRelay(sensor, jpeg_encoder=encode)
    relay.listen()
    sensor.callback(image())
    assert encoding.wait(1.0)
    sensor.is_alive = False

    def drain(seconds: float) -> None:
        assert 0.0 < seconds <= 0.4
        assert relay._encoder_thread is not None
        assert not relay._encoder_thread.is_alive()
        sensor.callback(image(2))

    def close() -> None:
        close_started.set()
        try:
            relay.close()
            close_finished.set()
        except BaseException as error:
            failures.append(error)

    closer = threading.Thread(target=close)
    with mock.patch("carla_vision.native.world_worker.time.sleep", side_effect=drain) as sleep:
        closer.start()
        try:
            assert close_started.wait(1.0)
            assert not close_finished.wait(0.02)
        finally:
            release_encoder.set()
            closer.join(timeout=2.0)

    assert not closer.is_alive()
    assert not failures
    assert close_finished.is_set()
    assert relay._close_complete
    assert sensor.stop_calls == 0
    sleep.assert_called_once()
    assert relay.snapshot()["telemetry"]["frames_received"] == 1


@pytest.mark.parametrize("state_after_stop", [False, True, None, "unavailable"])
def test_stop_error_requires_confirmed_dead_sensor_before_drain(state_after_stop: Any) -> None:
    class StateSensor(Sensor):
        state: Any = True

        @property
        def is_alive(self) -> Any:
            if self.state == "unavailable":
                raise RuntimeError("sensor state lookup failed")
            return self.state

        def stop(self) -> None:
            self.stop_calls += 1
            self.state = state_after_stop
            raise RuntimeError("sensor disappeared during stop")

    sensor = StateSensor()
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay.listen()
    try:
        with mock.patch("carla_vision.native.world_worker.time.sleep") as sleep:
            if state_after_stop is False:
                relay.close()
                assert relay._close_complete
                sleep.assert_called_once()
            else:
                with pytest.raises(RuntimeError, match="sensor disappeared during stop"):
                    relay.close()
                assert not relay._close_complete
                sleep.assert_not_called()
        assert sensor.stop_calls == 1
    finally:
        sensor.state = False
        relay._STOP_DRAIN_SECONDS = 0.0
        relay.close()

    assert relay._encoder_thread is not None
    assert not relay._encoder_thread.is_alive()


def test_close_serializes_with_pending_listener_registration() -> None:
    sensor = Sensor()
    listen_started = threading.Event()
    release_listen = threading.Event()
    close_started = threading.Event()
    failures: list[BaseException] = []
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay._STOP_DRAIN_SECONDS = 0.0

    def listen(callback: Any) -> None:
        listen_started.set()
        assert release_listen.wait(2.0)
        sensor.callback = callback

    def activate() -> None:
        try:
            relay.listen()
        except BaseException as error:
            failures.append(error)

    def close() -> None:
        close_started.set()
        try:
            relay.close()
        except BaseException as error:
            failures.append(error)

    starter = threading.Thread(target=activate)
    closer = threading.Thread(target=close)
    with mock.patch.object(sensor, "listen", side_effect=listen):
        starter.start()
        try:
            assert listen_started.wait(1.0)
            closer.start()
            assert close_started.wait(1.0)
            assert not sensor.stopped.wait(0.02)
        finally:
            release_listen.set()
            starter.join(timeout=2.0)
            if closer.ident is not None:
                closer.join(timeout=2.0)

    assert not starter.is_alive()
    assert not closer.is_alive()
    assert not failures
    assert relay._close_complete
    sensor.callback(image())
    assert relay.snapshot()["telemetry"]["frames_received"] == 0


def test_rapid_camera_replacements_leave_no_encoder_threads_or_late_frames() -> None:
    stale_callbacks: list[Any] = []
    with mock.patch.object(CompressedCameraRelay, "_STOP_DRAIN_SECONDS", 0.0):
        for frame in range(20):
            sensor = Sensor()
            relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
            relay.listen()
            sensor.callback(image(frame))
            stale_callbacks.append(sensor.callback)
            relay.close()
            relay.close()
            for callback in stale_callbacks:
                callback(image(999))
            assert relay.snapshot()["telemetry"]["frames_received"] == 1
            assert relay._pending_image is None
            assert relay._encoder_thread is not None
            assert not relay._encoder_thread.is_alive()
            assert sensor.stop_calls == 1
            with pytest.raises(RuntimeError, match="closed camera relay"):
                relay.listen()


def test_listener_failure_drains_encoder_before_rethrowing_original_error() -> None:
    sensor = Sensor()
    relay = CompressedCameraRelay(sensor, jpeg_encoder=lambda image, quality: b"jpeg")
    relay._STOP_DRAIN_SECONDS = 0.0

    def fail_after_callback(callback: Any) -> None:
        sensor.callback = callback
        callback(image())
        raise RuntimeError("registration failed")

    with mock.patch.object(sensor, "listen", side_effect=fail_after_callback):
        with pytest.raises(RuntimeError, match="registration failed"):
            relay.listen()

    assert relay._close_complete
    assert relay._encoder_thread is not None
    assert not relay._encoder_thread.is_alive()
    sensor.callback(image(2))
    assert relay.snapshot()["telemetry"]["frames_received"] == 1
    relay.close()
    assert sensor.stop_calls == 1


class WorkerCameraSensor(FakeActor):
    def __init__(
        self,
        world: FakeWorld,
        actor_id: int,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
    ) -> None:
        super().__init__(world, actor_id, blueprint, transform)
        self.callback: Any = None
        self.fail_listen = False
        self.fail_stop = False
        self.stop_attempts = 0

    def listen(self, callback: Any) -> None:
        self.callback = callback
        if self.fail_listen:
            raise RuntimeError("camera registration failed")

    def stop(self) -> None:
        self.stop_attempts += 1
        if self.fail_stop:
            raise RuntimeError("camera detach failed")
        super().stop()

    def destroy(self) -> None:
        assert self.stopped, "camera must be detached before actor destruction"
        super().destroy()


class WorkerCameraWorld(FakeWorld):
    def __init__(self, library: FakeBlueprintLibrary) -> None:
        super().__init__("Town10HD_Opt", library)
        self.fail_camera_listen = False
        self.fail_camera_stop = False
        self.cameras: list[WorkerCameraSensor] = []

    def spawn_actor(
        self,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
        attach_to: FakeActor | None = None,
    ) -> WorkerCameraSensor:
        del attach_to
        sensor = WorkerCameraSensor(self, self.next_actor_id, blueprint, transform)
        self.next_actor_id += 1
        sensor.fail_listen = self.fail_camera_listen
        sensor.fail_stop = self.fail_camera_stop
        self.cameras.append(sensor)
        self.actors[sensor.id] = sensor
        return sensor


@pytest.fixture
def worker_camera(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    library = FakeBlueprintLibrary()
    library.blueprints["sensor.camera.rgb"] = FakeBlueprint(
        "sensor.camera.rgb", {"role_name": FakeAttribute("")}
    )
    world = WorkerCameraWorld(library)
    traffic_manager = FakeTrafficManager(8000)
    client = FakeClient(world, library, traffic_manager)
    carla = FakeCarla(client)
    worker = WorldWorker(carla_loader=lambda: carla, start_monitor=False)
    monkeypatch.setattr(CompressedCameraRelay, "_STOP_DRAIN_SECONDS", 0.0)
    monkeypatch.setattr(
        "carla_vision.native.world_worker._load_in_memory_jpeg_encoder",
        lambda: lambda image, quality: b"jpeg",
    )
    prepared = worker.prepare({})
    scene = worker._scene
    assert scene is not None
    payload = {
        "lease_token": prepared["scene"]["lease_token"],
        "mode": "drive",
        "width": 320,
        "height": 180,
        "fps": 30.0,
        "fov": 90.0,
    }
    try:
        yield SimpleNamespace(worker=worker, world=world, scene=scene, payload=payload)
    finally:
        for sensor in world.cameras:
            sensor.fail_stop = False
        worker.close()


def test_activation_failure_retains_failed_relay_until_scene_stop_retries(
    worker_camera: SimpleNamespace,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    world.fail_camera_listen = True
    world.fail_camera_stop = True
    with pytest.raises(RuntimeError, match="camera registration failed") as raised:
        worker.camera(scene.scene_id, worker_camera.payload)

    sensor = world.cameras[0]
    relay = scene.camera_relay
    assert relay is not None
    assert relay.sensor is sensor
    assert scene.camera_config is not None
    assert worker._scene is scene
    assert any(owned.actor is sensor for owned in scene.owned_actors)
    assert world.get_actor(sensor.id) is sensor
    assert not sensor.destroyed
    assert any("camera detach failed" in note for note in raised.value.__notes__)
    stop_attempts = sensor.stop_attempts

    sensor.fail_stop = False
    stopped = worker.stop(scene.scene_id, {"lease_token": scene.lease_token})

    assert stopped["status"] == "stopped"
    assert sensor.stop_attempts == stop_attempts + 1
    assert sensor.destroyed
    assert relay._close_complete
    assert relay._encoder_thread is not None
    assert not relay._encoder_thread.is_alive()
    assert scene.camera_relay is None
    assert scene.camera_config is None
    assert worker._scene is None
    assert world.actors == {}


def test_scene_stop_retains_camera_and_all_owned_actors_until_detach_retry(
    worker_camera: SimpleNamespace,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    worker.camera(scene.scene_id, worker_camera.payload)
    sensor = world.cameras[0]
    relay = scene.camera_relay
    owned_ids = set(world.actors)
    sensor.fail_stop = True

    with pytest.raises(RuntimeError, match="camera detach failed"):
        worker.stop(scene.scene_id, {"lease_token": scene.lease_token})

    assert worker._scene is scene
    assert scene.status == "stopping"
    assert scene.camera_relay is relay
    assert scene.camera_config is not None
    assert any(owned.actor is sensor for owned in scene.owned_actors)
    assert set(world.actors) == owned_ids
    assert not sensor.destroyed

    sensor.fail_stop = False
    stopped = worker.stop(scene.scene_id, {"lease_token": scene.lease_token})

    assert stopped["status"] == "stopped"
    assert sensor.stop_attempts == 2
    assert sensor.destroyed
    assert relay is not None and relay._close_complete
    assert scene.camera_relay is None
    assert scene.camera_config is None
    assert worker._scene is None
    assert world.actors == {}


@pytest.mark.parametrize("control_mode", ["manual", "autopilot"])
def test_failed_camera_stop_brakes_ego_and_disables_autopilot_before_detach(
    worker_camera: SimpleNamespace,
    control_mode: str,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    lease = {"lease_token": scene.lease_token}
    worker.camera(scene.scene_id, worker_camera.payload)
    worker.mode(scene.scene_id, {**lease, "control_mode": control_mode})
    worker.start(scene.scene_id, lease)
    if control_mode == "manual":
        worker.control(
            scene.scene_id,
            {
                **lease,
                "sequence": 1,
                "throttle": 0.8,
                "steer": 0.2,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
            },
        )
    assert not scene.deadman_active
    assert scene.ego.autopilot is (control_mode == "autopilot")
    controls_before_stop = len(scene.ego.controls)
    owned_ids = set(world.actors)
    sensor = world.cameras[0]

    def fail_detach_after_safety_stop() -> None:
        assert scene.cleanup_guard_passed
        assert not scene.ego.autopilot
        assert len(scene.ego.controls) == controls_before_stop + 1
        assert scene.ego.controls[-1].throttle == 0.0
        assert scene.ego.controls[-1].brake == 1.0
        assert scene.deadman_active
        assert scene.route["enforced"] is False
        raise RuntimeError("camera detach failed")

    with mock.patch.object(sensor, "stop", side_effect=fail_detach_after_safety_stop):
        with pytest.raises(RuntimeError, match="camera detach failed"):
            worker.stop(scene.scene_id, lease)

    assert worker._scene is scene
    assert scene.status == "stopping"
    assert scene.camera_relay is not None
    assert set(world.actors) == owned_ids
    assert not sensor.destroyed
    assert not scene.ego.destroyed

    stopped = worker.stop(scene.scene_id, lease)
    assert stopped["status"] == "stopped"
    assert world.actors == {}


def test_autopilot_disable_failure_still_attempts_brake_before_camera_stop(
    worker_camera: SimpleNamespace,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    lease = {"lease_token": scene.lease_token}
    worker.camera(scene.scene_id, worker_camera.payload)
    worker.mode(scene.scene_id, {**lease, "control_mode": "autopilot"})
    worker.start(scene.scene_id, lease)
    controls_before_stop = len(scene.ego.controls)
    sensor = world.cameras[0]
    sensor.fail_stop = True

    with mock.patch.object(
        scene.ego, "set_autopilot", side_effect=RuntimeError("autopilot RPC failed")
    ):
        with pytest.raises(RuntimeError, match="camera detach failed"):
            worker.stop(scene.scene_id, lease)

    assert len(scene.ego.controls) == controls_before_stop + 1
    assert scene.ego.controls[-1].brake == 1.0
    assert scene.deadman_active
    assert any("ego autopilot disable failed" in error for error in scene.cleanup_errors)
    assert not scene.ego.destroyed
    assert not sensor.destroyed


@pytest.mark.parametrize("guard_failure", ["episode", "identity"])
def test_camera_stop_failure_does_not_bypass_ego_episode_or_identity_guard(
    worker_camera: SimpleNamespace,
    guard_failure: str,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    worker.camera(scene.scene_id, worker_camera.payload)
    sensor = world.cameras[0]
    sensor.fail_stop = True
    ego = scene.ego
    original_world_id = world.id
    original_role = ego.attributes["role_name"]
    if guard_failure == "episode":
        world.id += 1
    else:
        ego.attributes["role_name"] = "not-the-owned-ego"
    try:
        with (
            mock.patch.object(ego, "set_autopilot") as autopilot,
            mock.patch.object(ego, "apply_control") as brake,
            pytest.raises(RuntimeError, match="camera detach failed"),
        ):
            worker.stop(scene.scene_id, {"lease_token": scene.lease_token})

        autopilot.assert_not_called()
        brake.assert_not_called()
        assert not ego.destroyed
        assert not sensor.destroyed
        assert scene.camera_relay is not None
    finally:
        world.id = original_world_id
        ego.attributes["role_name"] = original_role


def test_worker_close_retries_retained_camera_without_reopening_worker(
    worker_camera: SimpleNamespace,
) -> None:
    worker, world, scene = worker_camera.worker, worker_camera.world, worker_camera.scene
    worker.camera(scene.scene_id, worker_camera.payload)
    sensor = world.cameras[0]
    relay = scene.camera_relay
    sensor.fail_stop = True

    with pytest.raises(RuntimeError, match="camera detach failed"):
        worker.close()

    assert worker._closed
    assert worker._scene is scene
    assert scene.camera_relay is relay
    assert not sensor.destroyed
    with pytest.raises(WorkerError, match="worker is closed"):
        worker.prepare({})

    sensor.fail_stop = False
    worker.close()

    assert worker._closed
    assert worker._scene is None
    assert sensor.stop_attempts == 2
    assert sensor.destroyed
    assert relay is not None and relay._close_complete
    assert world.actors == {}
