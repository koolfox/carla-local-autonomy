from __future__ import annotations

from typing import Any

import pytest

from carla_vision.operator.garage_acceptance import (
    GarageAcceptanceRunner,
    _catalog_identifier,
    _parse_args,
    _validate_args,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeWorker:
    def __init__(self, clock: FakeClock, *, version: str = "0.9.16") -> None:
        self.clock = clock
        self.version = version
        self.active = False
        self.paused = False
        self.pause_at: float | None = None
        self.events: list[str] = []
        self.mode = "manual"
        self.traffic = 0
        self.walkers = 0
        self.start_spawn_index: int | None = None
        self.destination_spawn_index: int | None = None
        self.route_mode = "free"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "ready": True,
            "carla": {
                "connected": True,
                "client_version": self.version,
                "server_version": self.version,
                "current_map": "Town10HD_Opt",
            },
            "capabilities": {
                "camera_pause_resume": True,
                "spawn_point_selection": True,
                "selected_route": True,
            },
        }

    def catalog(self) -> dict[str, Any]:
        return {
            "spawn_point_map": "Town10HD_Opt",
            "spawn_count": 4,
            "spawn_points": [
                {"index": index, "label": f"Spawn {index}", "transform": {}}
                for index in range(4)
            ],
            "capabilities": {
                "spawn_point_selection": True,
                "selected_route": True,
            },
        }

    def current_scene(self) -> dict[str, Any]:
        if not self.active:
            return {"schema_version": "1.0", "status": "idle", "scene": None}
        return {
            "schema_version": "1.0",
            "status": "running",
            "scene": {
                "scene_id": "scene-live-acceptance-01",
                "lease_token": "lease-live-acceptance-01",
                "status": "running",
                "episode_id": 10,
                "ego_actor_id": 101,
                "map_name": "Town10HD_Opt",
                "spawn_index": (
                    self.start_spawn_index if self.start_spawn_index is not None else 1
                ),
                "route_mode": self.route_mode,
                "route": (
                    {"planned": True, "waypoint_count": 12, "enforced": False}
                    if self.route_mode == "selected_destination"
                    else {"planned": False, "waypoint_count": 0, "enforced": False}
                ),
                "destination": (
                    {"spawn_index": self.destination_spawn_index, "transform": {}}
                    if self.destination_spawn_index is not None
                    else None
                ),
                "control_mode": "manual",
                "traffic_count": self.traffic,
                "walker_count": self.walkers,
                "prop_actor_ids": [],
                "lease_expires_in_seconds": 4.0,
                "cleanup_guard_passed": None,
                "cleanup_errors": [],
                "capabilities": {"camera_pause_resume": True},
            },
        }

    def pause_camera(self, scene: Any) -> Any:
        assert scene.scene_id == "scene-live-acceptance-01"
        assert self.active
        self.paused = True
        self.pause_at = self.clock()
        self.events.append("pause")
        return scene

    def resume_camera(self, scene: Any) -> Any:
        assert scene.scene_id == "scene-live-acceptance-01"
        assert self.active
        self.paused = False
        self.events.append("resume")
        return scene


class FakeManager:
    def __init__(
        self,
        worker: FakeWorker,
        clock: FakeClock,
        *,
        behavior_fails: bool = False,
    ) -> None:
        self.worker = worker
        self.clock = clock
        self.behavior_fails = behavior_fails
        self.active = False
        self.mode = "manual"
        self.session_id = ""
        self.controls_written = 0
        self.speed = 0.0
        self.last_control_at = clock()
        self.emergency = False
        self.frame_sequence = 0
        self.stop_count = 0
        self.start_count = 0
        self.shutdown_called = False

    def catalog(self) -> dict[str, Any]:
        return {
            "connected": True,
            "server_version": self.worker.version,
            "maps": ["Town01", {"id": "Town10HD_Opt", "label": "Town10HD Opt"}],
            "vehicles": [
                "vehicle.audi.tt",
                {"id": "vehicle.tesla.model3", "label": "Tesla Model3"},
            ],
            "world_worker": {"connected": True, "status": "ready"},
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.mode = str(payload["control_mode"])
        if self.mode == "behavior" and self.behavior_fails:
            raise ImportError("agents.navigation.behavior_agent is unavailable on Operator host")
        self.active = True
        self.worker.active = True
        self.worker.mode = self.mode
        self.worker.traffic = int(payload["traffic_count"])
        self.worker.walkers = int(payload["walker_count"])
        self.worker.start_spawn_index = payload.get("start_spawn_index")
        self.worker.destination_spawn_index = payload.get("destination_spawn_index")
        self.worker.route_mode = str(payload.get("route_mode", "free"))
        self.worker.paused = False
        self.worker.pause_at = None
        self.session_id = f"session-{self.start_count + 1}"
        self.start_count += 1
        self.controls_written = 0
        self.speed = 0.0
        self.last_control_at = self.clock()
        self.emergency = False
        return self.state()

    def state(self) -> dict[str, Any]:
        if not self.active:
            return {"status": "idle", "session_id": None}
        source = "browser_manual"
        deadman = False
        stale = False
        if self.mode == "behavior":
            source = "behavior_agent"
        elif self.emergency:
            source = "emergency_stop"
            deadman = True
        elif self.worker.paused and self.worker.pause_at is not None:
            if self.clock() - self.worker.pause_at > 1.55:
                source = "camera_deadman"
                deadman = True
                stale = True
        elif self.clock() - self.last_control_at > 1.05:
            source = "browser_deadman"
            deadman = True
        state: dict[str, Any] = {
            "schema_version": "1.0",
            "status": "running",
            "session_id": self.session_id,
            "run_id": "fake-run",
            "vehicle_id": 101,
            "camera_id": 202,
            "traffic_count_actual": self.worker.traffic,
            "walker_count_actual": self.worker.walkers,
            "controls_written": self.controls_written,
            "control_source": source,
            "deadman_active": deadman,
            "emergency_stop": self.emergency,
            "telemetry": {"speed": self.speed},
            "stream": {"stale": stale, "source_fps": 30.0, "frame_age_seconds": 0.02},
        }
        if self.mode == "behavior":
            state["autonomy"] = {
                "policy_ready": True,
                "commands": 3,
                "detail": {
                    "destination_index": self.worker.destination_spawn_index,
                    "navigation_intent": {
                        "schema_version": "1.0",
                        "source_frame": {"kind": "carla_world_frame", "id": 77, "exact": True},
                        "command": "left",
                        "privileged": True,
                    }
                },
            }
        return state

    def control(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert raw["session_id"] == self.session_id
        self.controls_written += 1
        self.speed = 1.0
        self.last_control_at = self.clock()
        return self.state()

    def emergency_stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert raw == {"session_id": self.session_id}
        self.emergency = True
        return self.state()

    def frame(self, view: str) -> tuple[int, bytes]:
        assert view == "raw"
        return self.frame_sequence, b"\xff\xd8" + b"x" * 150 + b"\xff\xd9"

    def wait_for_frame(
        self,
        view: str,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> tuple[int, bytes]:
        del timeout
        assert view == "raw"
        if self.worker.paused:
            raise AssertionError("runner must resume the real camera before waiting for recovery")
        self.frame_sequence = max(self.frame_sequence + 1, after_sequence + 1)
        return self.frame_sequence, b"\xff\xd8" + b"x" * 150 + b"\xff\xd9"

    def stop(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert raw == {"session_id": self.session_id}
        self.active = False
        self.worker.active = False
        self.worker.paused = False
        self.stop_count += 1
        return {
            "status": "success",
            "session_id": self.session_id,
            "cleanup_errors": [],
        }

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.active = False
        self.worker.active = False


def runner(
    *,
    repeat: int = 2,
    version: str = "0.9.16",
    behavior_fails: bool = False,
    start_spawn_index: int | None = None,
    destination_spawn_index: int | None = None,
) -> tuple[GarageAcceptanceRunner, FakeManager, FakeWorker]:
    clock = FakeClock()
    worker = FakeWorker(clock, version=version)
    manager = FakeManager(worker, clock, behavior_fails=behavior_fails)
    return (
        GarageAcceptanceRunner(
            manager,
            worker,
            host="192.168.1.20",
            port=2000,
            map_name="Town10HD_Opt",
            vehicle="vehicle.tesla.model3",
            seed=20260809,
            traffic_count=5,
            walker_count=4,
            camera_fps=30.0,
            repeat=repeat,
            start_spawn_index=start_spawn_index,
            destination_spawn_index=destination_spawn_index,
            state_timeout=10.0,
            clock=clock,
            sleep=clock.sleep,
        ),
        manager,
        worker,
    )


def test_catalog_identifier_handles_existing_string_and_mapping_shapes() -> None:
    assert _catalog_identifier("Town10HD_Opt") == "Town10HD_Opt"
    assert _catalog_identifier({"id": "Town10HD_Opt", "label": "Town"}) == "Town10HD_Opt"
    assert _catalog_identifier({"name": "Town10HD_Opt"}) == "Town10HD_Opt"


def test_repeatable_gate_runs_manual_and_behavior_twice_with_real_fault_order() -> None:
    acceptance, manager, worker = runner(repeat=2)
    report = acceptance.run()

    assert report["status"] == "pass"
    assert [session["mode"] for session in report["sessions"]] == [
        "manual",
        "behavior",
        "manual",
        "behavior",
    ]
    assert all(session["status"] == "pass" for session in report["sessions"])
    assert manager.stop_count == 4
    assert manager.start_count == 4
    assert manager.shutdown_called
    assert worker.events == ["pause", "resume", "pause", "resume"]

    manual = [item for item in report["sessions"] if item["mode"] == "manual"]
    for session in manual:
        passed = {check["check_id"]: check["passed"] for check in session["checks"]}
        assert passed["manual_control_effect"]
        assert passed["control_stale_fail_safe"]
        assert passed["camera_stale_fail_safe"]
        assert passed["camera_recovered"]
        assert passed["emergency_brake"]
        assert passed["worker_scene_released"]



def test_selected_map_start_and_destination_are_machine_readable_live_evidence() -> None:
    acceptance, _, _ = runner(
        repeat=1,
        start_spawn_index=1,
        destination_spawn_index=2,
    )
    report = acceptance.run()

    assert report["status"] == "pass"
    assert report["target"]["route_selection"] == {
        "start_spawn_index": 1,
        "destination_spawn_index": 2,
        "behavior_route_mode": "selected_destination",
    }
    preflight = {check["check_id"]: check for check in report["checks"]}
    assert preflight["spawn_point_selection_capability"]["passed"]
    assert preflight["spawn_catalog_matches_target_map"]["passed"]
    assert preflight["start_spawn_index_available"]["passed"]
    assert preflight["selected_route_capability"]["passed"]
    assert preflight["destination_spawn_index_available"]["passed"]

    manual, behavior = report["sessions"]
    assert manual["requested"]["route_mode"] == "free"
    assert manual["requested"]["start_spawn_index"] == 1
    assert manual["requested"]["destination_spawn_index"] is None
    assert behavior["requested"]["route_mode"] == "selected_destination"
    assert behavior["requested"]["start_spawn_index"] == 1
    assert behavior["requested"]["destination_spawn_index"] == 2
    behavior_checks = {check["check_id"]: check for check in behavior["checks"]}
    assert behavior_checks["selected_map_applied"]["passed"]
    assert behavior_checks["selected_start_spawn_applied"]["passed"]
    assert behavior_checks["selected_destination_planned"]["passed"]
    assert behavior_checks["behavior_destination_matches_selected"]["passed"]


def test_wrong_carla_version_fails_preflight_without_spawning() -> None:
    acceptance, manager, _ = runner(version="0.9.15", repeat=1)
    report = acceptance.run()
    assert report["status"] == "fail"
    assert manager.start_count == 0
    checks = {check["check_id"]: check for check in report["checks"]}
    assert checks["carla_server_version"]["passed"] is False
    assert checks["worker_pythonapi_version"]["passed"] is False


def test_missing_operator_behavior_pythonapi_is_machine_readable_failure() -> None:
    acceptance, manager, _ = runner(repeat=1, behavior_fails=True)
    report = acceptance.run()
    assert report["status"] == "fail"
    assert len(report["sessions"]) == 2
    assert report["sessions"][0]["mode"] == "manual"
    behavior = report["sessions"][1]
    assert behavior["mode"] == "behavior"
    assert behavior["status"] == "fail"
    assert behavior["error"]["type"] == "ImportError"
    assert "BehaviorAgent" not in behavior["error"]["message"] or behavior["error"]["message"]
    assert manager.stop_count == 1


def test_cli_uses_worker_token_environment_name_not_plaintext_secret() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://192.168.1.20:8766",
            "--world-worker-token-env",
            "MY_WORKER_TOKEN",
        ]
    )
    assert args.world_worker_token_env == "MY_WORKER_TOKEN"
    assert not hasattr(args, "world_worker_token")
    _validate_args(args)


def test_cli_accepts_exact_spawn_selection_and_rejects_same_endpoint() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--start-spawn-index",
            "1",
            "--destination-spawn-index",
            "2",
        ]
    )
    assert args.start_spawn_index == 1
    assert args.destination_spawn_index == 2
    _validate_args(args)

    same = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--start-spawn-index",
            "2",
            "--destination-spawn-index",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="must differ"):
        _validate_args(same)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--repeat", "0"),
        ("--traffic", "251"),
        ("--walkers", "251"),
        ("--camera-fps", "61"),
        ("--state-timeout", "2"),
    ],
)
def test_cli_rejects_unbounded_live_gate_values(flag: str, value: str) -> None:
    args = _parse_args(["--world-worker-url", "http://127.0.0.1:8766", flag, value])
    with pytest.raises(ValueError):
        _validate_args(args)
