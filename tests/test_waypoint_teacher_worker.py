from __future__ import annotations

import http.client
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from test_world_worker import (
    FakeBlueprintLibrary,
    FakeCarla,
    FakeClient,
    FakeClock,
    FakeLocation,
    FakePlanner,
    FakeTrafficManager,
    FakeTransform,
    FakeWaypoint,
    FakeWorld,
)

from carla_vision.native.world_worker import WorkerError, WorldWorker, create_server
from carla_vision.operator.world_worker_client import (
    WorldWorkerClient,
    WorldWorkerError,
    WorldWorkerScene,
)


@pytest.fixture
def native() -> Any:
    clock = FakeClock()
    library = FakeBlueprintLibrary()
    world = FakeWorld("Town10HD_Opt", library)
    tm = FakeTrafficManager(8000)
    client = FakeClient(world, library, tm)
    worker = WorldWorker(
        carla_loader=lambda: FakeCarla(client),
        route_planner_loader=lambda: FakePlanner,
        clock=clock,
        lease_seconds=5,
        start_monitor=False,
    )
    response = worker.prepare({"route_mode": "random_destination"})
    scene = worker._scene
    assert scene is not None
    scene.route_locations = [FakeLocation(float(x), 1.0, 0.0) for x in range(0, 300, 2)]
    try:
        yield SimpleNamespace(
            worker=worker,
            world=world,
            client=client,
            tm=tm,
            clock=clock,
            scene=scene,
            response=response,
        )
    finally:
        worker.close()


def read(native: Any, **fields: Any) -> dict[str, Any]:
    return native.worker.waypoints(
        native.scene.scene_id, {"lease_token": native.scene.lease_token, **fields}
    )


def test_fixed_route_uses_image_location_and_never_changes_world(native: Any) -> None:
    actor_ids = set(native.world.actors)
    lease_deadline = native.scene.lease_deadline
    native.scene.status = "running"
    native.scene.control_mode = "autopilot"
    with (
        mock.patch.object(native.tm, "set_path", side_effect=AssertionError("route mutation")),
        mock.patch.object(
            native.scene.ego, "get_location", side_effect=AssertionError("live pose")
        ),
        mock.patch.object(native.world, "get_map", side_effect=AssertionError("map reload")),
        mock.patch.object(
            native.tm, "get_all_actions", create=True, side_effect=AssertionError("TM")
        ),
    ):
        result = read(native, camera_location={"x": 40, "y": 1, "z": 2}, source_frame=99)
    assert result["source"] == "planned_route"
    assert result["points"][0] == {"x": 38.0, "y": 1.0, "z": 0.0}
    assert result["points"][-1]["x"] == 138.0
    assert len(result["points"]) == 51
    assert result["source_frame"] == 99
    assert result["teacher_only"] is True
    assert result["model_input"] is False
    assert result["controls_vehicle"] is False
    assert result["route_frame_matched"] is False
    assert result["coordinate_frame"] == "carla_world_metres"
    assert result["scene_id"] == native.scene.scene_id
    assert result["episode_id"] == native.scene.episode_id
    assert "lease_token" not in result
    assert native.scene.lease_deadline == lease_deadline
    assert set(native.world.actors) == actor_ids


def test_free_autopilot_uses_official_tm_upcoming_actions(native: Any) -> None:
    native.scene.route_locations = []
    native.scene.status = "running"
    native.scene.control_mode = "autopilot"
    native.world.get_snapshot = lambda: SimpleNamespace(
        frame=102, timestamp=SimpleNamespace(elapsed_seconds=10.2)
    )
    actions = [["Left", FakeWaypoint(FakeLocation(x, 1, 0))] for x in (2, 4, 6)]
    with mock.patch.object(
        native.tm, "get_all_actions", create=True, return_value=actions
    ) as get_actions:
        result = read(native, camera_location={"x": 0, "y": 1, "z": 2}, source_frame=99)
    get_actions.assert_called_once_with(native.scene.ego)
    assert result["source"] == "traffic_manager"
    assert len(result["points"]) == 3
    assert result["sampled_frame"] == 102
    assert result["sampled_timestamp"] == 10.2
    assert result["source_frame"] == 99
    assert result["route_frame_matched"] is False


class LaneWaypoint:
    def __init__(self, x: float, branch: bool = False) -> None:
        self.transform = FakeTransform(FakeLocation(x, 1.0, 0.0))
        self.branch = branch

    def next(self, distance: float) -> list[LaneWaypoint]:
        assert distance == 2.0
        if self.branch:
            return [LaneWaypoint(99), LaneWaypoint(-99)]
        return [LaneWaypoint(self.transform.location.x + 2.0, branch=True)]


def test_lane_fallback_stops_before_ambiguous_branch_and_caches_map(native: Any) -> None:
    native.scene.route_locations = []
    captured_origins = []

    def get_waypoint(origin: Any) -> LaneWaypoint:
        captured_origins.append(origin)
        return LaneWaypoint(origin.x)

    native.world.map.get_waypoint = get_waypoint
    with mock.patch.object(native.world, "get_map", wraps=native.world.get_map) as get_map:
        first = read(native, camera_location={"x": 10, "y": 1, "z": 2})
        second = read(native, camera_location={"x": 20, "y": 1, "z": 2})
    assert get_map.call_count == 1
    assert first["source"] == second["source"] == "lane_centerline"
    assert [point["x"] for point in first["points"]] == [10, 12]
    assert [point["x"] for point in second["points"]] == [20, 22]
    assert [origin.x for origin in captured_origins] == [10, 20]


def test_missing_tm_and_lane_api_reports_optional_overlay_unavailable(native: Any) -> None:
    native.scene.route_locations = []
    with pytest.raises(WorkerError) as caught:
        read(native)
    assert caught.value.code == "waypoint_teacher_unavailable"
    assert native.scene.status == "prepared"
    assert native.scene.ego.is_alive


@pytest.mark.parametrize(
    "guard,code",
    [
        ("lease", "lease_mismatch"),
        ("expired", "lease_expired"),
        ("episode", "episode_changed"),
        ("identity", "scene_identity_changed"),
        ("dead", "scene_identity_changed"),
        ("stopping", "scene_inactive"),
    ],
)
def test_read_requires_live_owned_scene(native: Any, guard: str, code: str) -> None:
    raw = {"lease_token": native.scene.lease_token}
    if guard == "lease":
        raw["lease_token"] = "incorrect"
    elif guard == "expired":
        native.clock.advance(6)
    elif guard == "episode":
        native.world.id += 1
    elif guard == "identity":
        native.scene.ego.attributes["role_name"] = "not-owned"
    elif guard == "dead":
        native.scene.ego.is_alive = False
    else:
        native.scene.status = "stopping"
    with pytest.raises(WorkerError) as caught:
        native.worker.waypoints(native.scene.scene_id, raw)
    assert caught.value.code == code


def test_busy_read_never_waits_behind_scene_configuration(native: Any) -> None:
    entered = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with native.worker._lock:
            entered.set()
            release.wait(2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert entered.wait(1)
    try:
        with pytest.raises(WorkerError) as caught:
            read(native)
        assert caught.value.code == "worker_busy"
    finally:
        release.set()
        holder.join(1)


@pytest.mark.parametrize(
    "fields",
    [
        {"camera_location": 1},
        {"camera_location": {"x": 1, "y": 2}},
        {"camera_location": {"x": True, "y": 0, "z": 0}},
        {"camera_location": {"x": float("nan"), "y": 0, "z": 0}},
        {"camera_location": {"x": 1e8, "y": 0, "z": 0}},
        {"source_frame": -1},
        {"source_frame": True},
        {"source_frame": 1.5},
        {"unknown": "field"},
    ],
)
def test_native_rejects_bad_image_metadata(native: Any, fields: Any) -> None:
    with pytest.raises(WorkerError) as caught:
        read(native, **fields)
    assert caught.value.status == 400


def test_point_bound_and_duplicate_points() -> None:
    points = [FakeLocation(x / 10, 0, 0) for x in range(1000)]
    points.insert(0, FakeLocation(0, 0, 0))
    result = WorldWorker._bounded_waypoint_locations(points, FakeLocation())
    assert len(result) == 64
    assert len({point["x"] for point in result}) == 64


def test_authenticated_http_endpoint_and_client_round_trip(native: Any) -> None:
    server = create_server(
        bind="127.0.0.1", port=0, token="test-only-worker-token", worker=native.worker
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = WorldWorkerClient(
        f"http://127.0.0.1:{server.server_address[1]}",
        bearer_token="test-only-worker-token",
    )
    scene = WorldWorkerScene.from_response(native.response)
    try:
        result = client.waypoints(scene, camera_location={"x": 4, "y": 1, "z": 2}, source_frame=7)
        assert result["source"] == "planned_route"
        assert result["source_frame"] == 7
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "POST",
            f"/v1/scenes/{scene.scene_id}/waypoints",
            body=json.dumps({"lease_token": scene.lease_token}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_old_worker_capability_is_rejected_without_network(native: Any) -> None:
    client = WorldWorkerClient("http://127.0.0.1:8766", bearer_token="test-only-worker-token")
    scene = replace(WorldWorkerScene.from_response(native.response), capabilities={})
    with mock.patch.object(client, "_request", side_effect=AssertionError("network")):
        with pytest.raises(WorldWorkerError) as caught:
            client.waypoints(scene)
    assert caught.value.code == "waypoint_teacher_unavailable"


@pytest.mark.parametrize(
    "change",
    [
        {"scene_id": "other-scene"},
        {"episode_id": -1},
        {"source_frame": 123},
        {"teacher_only": False},
        {"model_input": True},
        {"route_frame_matched": True},
        {"source": "vision_model"},
        {"points": [{"x": float("nan"), "y": 0, "z": 0}]},
        {"points": [{"x": 0, "y": 0, "z": 0}] * 65},
        {"points": [{"x": 0, "y": 0, "z": 0}, {"x": 101, "y": 0, "z": 0}]},
    ],
)
def test_client_validates_teacher_response_and_short_timeout(native: Any, change: Any) -> None:
    client = WorldWorkerClient(
        "http://127.0.0.1:8766", bearer_token="test-only-worker-token", timeout=5
    )
    scene = WorldWorkerScene.from_response(native.response)
    with mock.patch.object(client, "_request", return_value={**read(native), **change}) as request:
        with pytest.raises(WorldWorkerError):
            client.waypoints(scene)
    assert request.call_args.kwargs["timeout"] == 1
