from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from test_world_worker import (
    FakeBlueprintLibrary,
    FakeCarla,
    FakeClient,
    FakeClock,
    FakePlanner,
    FakeTrafficManager,
    FakeWorld,
)
from test_world_worker_population import BatchClient, make_population_harness
from test_world_worker_spawn_failures import install_command_cleanup

from carla_vision.native.observable_world_worker import ObservableWorldWorker
from carla_vision.native.world_worker import (
    CompressedCameraConfig,
    SceneConfig,
    WorkerError,
    WorldWorker,
    create_server,
)


@pytest.fixture
def native() -> Iterator[SimpleNamespace]:
    clock = FakeClock()
    library = FakeBlueprintLibrary()
    world = FakeWorld("Town10HD_Opt", library)
    tm = FakeTrafficManager(8000)
    client = FakeClient(world, library, tm)
    carla = FakeCarla(client)
    worker = WorldWorker(
        carla_loader=lambda: carla,
        route_planner_loader=lambda: FakePlanner,
        clock=clock,
        lease_seconds=5,
        start_monitor=False,
    )
    try:
        yield SimpleNamespace(
            worker=worker,
            world=world,
            client=client,
            tm=tm,
            clock=clock,
            library=library,
            carla=carla,
        )
    finally:
        worker.close()


def apply(worker: WorldWorker, **changes: Any) -> dict[str, Any]:
    scene = worker._scene
    assert scene is not None
    return worker.configure(
        scene.scene_id,
        {
            **scene.config.as_dict(),
            **changes,
            "lease_token": scene.lease_token,
        },
    )


def ids(actors: list[Any]) -> list[int]:
    return [int(actor.id) for actor in actors]


def test_dynamics_keep_ego_camera_route_and_replace_walkers(native: Any) -> None:
    worker = native.worker
    before = worker.prepare(
        {"traffic_count": 2, "walker_count": 2, "route_mode": "random_destination"}
    )["scene"]
    scene = worker._scene
    existing = set(native.world.actors)
    walker_ids = {actor.id for actor in [*scene.walker_actors, *scene.walker_controllers]}
    route = scene.route
    destinations = [controller.destination for controller in scene.walker_controllers]
    result = apply(
        worker,
        speed_difference_percent=-20,
        following_distance_metres=5,
        pedestrian_crossing_factor=0.8,
        weather_preset="heavy-rain",
    )["scene"]
    assert result["ego_actor_id"] == before["ego_actor_id"]
    assert result["lease_token"] == before["lease_token"]
    assert result["scene_id"] == before["scene_id"]
    assert result["episode_id"] == before["episode_id"]
    assert existing - walker_ids <= set(native.world.actors)
    assert not walker_ids & set(native.world.actors)
    assert len(scene.walker_actors) == 2
    assert scene.route is route
    assert native.tm.speed_difference == -20
    assert native.tm.distance == 5
    assert native.world.cross_factor == 0.8
    assert all(
        controller.destination is not old
        for controller, old in zip(scene.walker_controllers, destinations, strict=True)
    )
    assert result["config"]["weather_preset"] == "heavy-rain"
    assert result["capabilities"]["prepared_scene_reconfigure"]


def test_noop_does_not_respawn_or_replan(native: Any) -> None:
    native.worker.prepare({"route_mode": "random_destination"})
    with (
        mock.patch.object(native.worker, "_spawn_ego", side_effect=AssertionError("spawn")),
        mock.patch.object(native.worker, "_plan_random_route", side_effect=AssertionError("route")),
        mock.patch.object(native.world, "set_weather", side_effect=AssertionError("weather")),
    ):
        assert apply(native.worker)["status"] == "prepared"


def test_population_deltas_preserve_existing_actor_ids_and_close_everything() -> None:
    harness = make_population_harness()
    destroyed = install_command_cleanup(harness)
    worker = harness.worker
    try:
        worker.prepare({"traffic_count": 4, "walker_count": 4})
        scene = worker._scene
        assert scene is not None
        ego = scene.ego.id
        vehicles = ids(scene.vehicle_actors)
        walkers = ids(scene.walker_actors)
        controllers = ids(scene.walker_controllers)
        client = harness.client
        assert isinstance(client, BatchClient)
        client.batches.clear()
        result = apply(worker, traffic_count=6, walker_count=6)["scene"]
        assert result["traffic_count"] == result["traffic_count_requested"] == 6
        assert result["walker_count"] == result["walker_count_requested"] == 6
        assert ids(scene.vehicle_actors)[:4] == vehicles
        assert ids(scene.walker_actors)[:4] == walkers
        assert [len(batch) for batch in client.batches] == [2, 2, 2]
        apply(worker, traffic_count=2, walker_count=2)
        assert ids(scene.vehicle_actors) == vehicles[:2]
        assert ids(scene.walker_actors) == walkers[:2]
        assert ids(scene.walker_controllers) == controllers[:2]
        assert scene.ego.id == ego
        assert harness.world.live_population() == {"traffic": 2, "walker": 2, "controller": 2}
        assert len(scene.owned_actors) == 7
        assert len(destroyed) == 12  # Only four NPCs and four walker/controller pairs.
        assert ego not in destroyed
    finally:
        worker.close()
    assert not harness.world.actors


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"seed": 5}, "scene_reconfigure_unsupported"),
        ({"map_name": "Town05"}, "scene_reconfigure_unsupported"),
        ({"traffic_count": 9}, "scene_population_capacity"),
        ({"vehicle_blueprint": "vehicle.missing"}, "blueprint_unavailable"),
        ({"color": "255,255,255"}, "color_unavailable"),
    ],
)
def test_unsupported_changes_are_rejected_before_mutation(
    native: Any, changes: Any, code: str
) -> None:
    native.worker.prepare({"traffic_count": 1})
    original = set(native.world.actors)
    config = native.worker._scene.config
    with pytest.raises(WorkerError) as caught:
        apply(native.worker, **changes, weather_preset="heavy-rain")
    assert caught.value.code == code
    assert set(native.world.actors) == original
    assert native.worker._scene.config == config


@pytest.mark.parametrize(
    "guard,code",
    [
        ("lease", "lease_mismatch"),
        ("expired", "lease_expired"),
        ("episode", "episode_changed"),
        ("identity", "scene_identity_changed"),
        ("running", "scene_not_prepared"),
    ],
)
def test_scene_guards_do_not_mutate_population(native: Any, guard: str, code: str) -> None:
    native.worker.prepare({"traffic_count": 1})
    scene = native.worker._scene
    original = set(native.world.actors)
    raw = {**scene.config.as_dict(), "traffic_count": 2, "lease_token": scene.lease_token}
    if guard == "lease":
        raw["lease_token"] = "not-my-lease"
    elif guard == "expired":
        native.clock.advance(6)
    elif guard == "episode":
        native.client.world = FakeWorld("Town10HD_Opt", native.library)
    elif guard == "identity":
        scene.ego.attributes["role_name"] = "somebody_else"
    else:
        scene.status = "running"
    with pytest.raises(WorkerError) as caught:
        native.worker.configure(scene.scene_id, raw)
    assert caught.value.code == code
    assert set(native.world.actors) == original
    assert scene.config.traffic_count == 1


def test_growth_shortfall_rolls_back_only_new_actors_and_refreshes_lease(native: Any) -> None:
    native.worker.prepare({"traffic_count": 1, "walker_count": 1})
    old = set(native.world.actors)
    spawn = native.worker._spawn_traffic

    def shortfall(*args: Any) -> list[Any]:
        arguments = list(args)
        arguments[5] -= 1
        result = spawn(*arguments)
        native.clock.advance(30)
        return result

    with mock.patch.object(native.worker, "_spawn_traffic", side_effect=shortfall):
        with pytest.raises(WorkerError) as caught:
            apply(native.worker, traffic_count=3)
    assert caught.value.code == "scene_population_shortfall"
    assert set(native.world.actors) == old
    assert native.worker._scene.config.traffic_count == 1
    assert native.worker._scene.status == "prepared"
    assert native.worker._scene.lease_deadline == native.clock() + 5
    native.worker.enforce_timeouts()
    assert native.worker._scene is not None


def test_spawn_exception_does_not_lose_partial_actor_ownership(native: Any) -> None:
    native.worker.prepare({"traffic_count": 1})
    old = set(native.world.actors)
    spawn = native.worker._spawn_traffic

    def partial_exception(*args: Any) -> list[Any]:
        spawn(*args)
        raise RuntimeError("lost spawn reply")

    with mock.patch.object(native.worker, "_spawn_traffic", side_effect=partial_exception):
        with pytest.raises(WorkerError, match="lost spawn reply"):
            apply(native.worker, traffic_count=2)
    assert set(native.world.actors) == old
    assert {item.actor_id for item in native.worker._scene.owned_actors} == old
    native.worker.close()
    assert not native.world.actors


def test_props_replace_only_props_and_partial_spawn_is_truthful(native: Any) -> None:
    native.worker.prepare({"traffic_count": 1, "walker_count": 1, "prop_preset": "cones"})
    scene = native.worker._scene
    existing = set(native.world.actors) - set(ids(scene.prop_actors))
    old_props = ids(scene.prop_actors)
    result = apply(native.worker, prop_preset="construction")["scene"]
    assert existing <= set(native.world.actors)
    assert not set(old_props) & set(native.world.actors)
    assert result["config"]["prop_preset"] == "construction"
    assert len(result["prop_actor_ids"]) == 3
    spawn = native.world.try_spawn_actor

    def fail_second_prop(blueprint: Any, *args: Any, **kwargs: Any) -> Any:
        if blueprint.id == "static.prop.trafficcone02":
            return None
        return spawn(blueprint, *args, **kwargs)

    with mock.patch.object(native.world, "try_spawn_actor", side_effect=fail_second_prop):
        with pytest.raises(WorkerError, match="could not spawn prop"):
            apply(native.worker, prop_preset="cones")
    assert scene.config.prop_preset == "none"
    assert scene.prop_actors == []
    assert set(native.world.actors) == existing


def test_vehicle_and_color_replace_only_ego_at_same_location(native: Any) -> None:
    native.worker.prepare({"traffic_count": 1, "walker_count": 1})
    scene = native.worker._scene
    ego = scene.ego
    old_population = set(native.world.actors) - {ego.id}
    sensor = SimpleNamespace(id=999, parent=None)
    relay = SimpleNamespace(sensor=sensor, snapshot=lambda: {"actor_id": 999}, close=lambda: None)
    scene.camera_relay = relay
    scene.camera_config = CompressedCameraConfig("garage", 1280, 720, 30, 90)
    result = apply(native.worker, vehicle_blueprint="vehicle.audi.tt", color="0,0,255")["scene"]
    assert result["ego_actor_id"] != ego.id
    assert ego.destroyed
    assert old_population <= set(native.world.actors)
    assert scene.ego.get_transform().location == ego.get_transform().location
    assert scene.ego.controls[-1].brake == 1.0
    assert scene.ego.attributes["color"] == "0,0,255"
    assert scene.camera_relay is relay
    assert result["camera"]["actor_id"] == 999
    assert result["camera_config"] == {
        "mode": "garage",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fov": 90,
        "yaw": 325,
        "pitch": -10,
        "distance": 6.5,
    }
    assert result["config"]["vehicle_blueprint"] == "vehicle.audi.tt"


def test_vehicle_failure_restores_old_ego_without_touching_population(native: Any) -> None:
    native.worker.prepare({"traffic_count": 1, "color": "255,0,0"})
    scene = native.worker._scene
    old_population = ids(scene.vehicle_actors)
    old_ego = scene.ego.id
    spawn = native.world.try_spawn_actor

    def reject_audi(blueprint: Any, *args: Any, **kwargs: Any) -> Any:
        return None if blueprint.id == "vehicle.audi.tt" else spawn(blueprint, *args, **kwargs)

    with mock.patch.object(native.world, "try_spawn_actor", side_effect=reject_audi):
        with pytest.raises(WorkerError, match="original vehicle restored"):
            apply(native.worker, vehicle_blueprint="vehicle.audi.tt")
    assert scene.ego.id != old_ego
    assert scene.ego.type_id == "vehicle.tesla.model3"
    assert scene.ego.attributes["color"] == "255,0,0"
    assert ids(scene.vehicle_actors) == old_population
    assert scene.config.vehicle_blueprint == "vehicle.tesla.model3"
    assert scene.status == "prepared"


def test_attached_camera_rejects_vehicle_change_before_destroy(native: Any) -> None:
    native.worker.prepare({})
    scene = native.worker._scene
    scene.camera_config = CompressedCameraConfig("drive", 1280, 720, 30, 90)
    scene.camera_relay = SimpleNamespace(snapshot=lambda: {}, close=lambda: None)
    with pytest.raises(WorkerError) as caught:
        apply(native.worker, vehicle_blueprint="vehicle.audi.tt")
    assert caught.value.code == "scene_reconfigure_unsupported"
    assert scene.ego.is_alive


def test_route_change_and_initial_mode_keep_prepared_vehicle_parked(native: Any) -> None:
    native.worker.prepare({})
    scene = native.worker._scene
    apply(native.worker, route_mode="random_destination", initial_control_mode="autopilot")
    assert scene.route["planned"]
    assert scene.control_mode == "autopilot"
    assert not scene.ego.autopilot
    apply(native.worker, route_mode="free")
    assert not scene.route["planned"]
    assert scene.route_locations == []
    assert scene.destination is None
    assert scene.route["provider"] == "TrafficManager"


def test_configure_health_remains_responsive_using_observable_cache(native: Any) -> None:
    worker = ObservableWorldWorker(carla_loader=lambda: native.carla, start_monitor=False)
    worker.prepare({})
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_spawn = worker._spawn_traffic

    def slow_spawn(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(2)
        return real_spawn(*args, **kwargs)

    def configure() -> None:
        try:
            apply(worker, traffic_count=2)
        except BaseException as error:
            errors.append(error)

    try:
        with mock.patch.object(worker, "_spawn_traffic", side_effect=slow_spawn):
            thread = threading.Thread(target=configure)
            thread.start()
            assert started.wait(1)
            before = time.monotonic()
            health, scene = worker.health(), worker.current_scene()
            assert time.monotonic() - before < 0.2
            assert health["busy"]
            assert scene["status"] == "preparing"
            assert scene["preparation"]["stage"] == "configure"
            with pytest.raises(WorkerError, match="already in progress"):
                worker.prepare({})
            release.set()
            thread.join(2)
            assert not thread.is_alive()
        assert not errors
        assert worker.health()["preparation"]["actual"]["traffic"] == 2
    finally:
        release.set()
        worker.close()


def test_configure_accepts_prepare_defaults_and_rejects_unknown_fields(native: Any) -> None:
    native.worker.prepare({})
    scene = native.worker._scene
    assert (
        native.worker.configure(scene.scene_id, {"lease_token": scene.lease_token})["status"]
        == "prepared"
    )
    with pytest.raises(WorkerError, match="unknown"):
        native.worker.configure(
            scene.scene_id,
            {
                **SceneConfig().as_dict(),
                "lease_token": scene.lease_token,
                "shell": "anything",
            },
        )


def test_cross_factor_is_set_before_replacement_spawn_and_noop_keeps_walkers(native: Any) -> None:
    native.worker.prepare({"walker_count": 2})
    scene = native.worker._scene
    original_spawn = native.worker._spawn_walkers

    def spawn(*args, **kwargs):
        assert native.world.cross_factor == 1.0
        return original_spawn(*args, **kwargs)

    with mock.patch.object(native.worker, "_spawn_walkers", side_effect=spawn) as replacement:
        apply(native.worker, pedestrian_crossing_factor=1.0)
        replacement.assert_called_once()
        ids = [actor.id for actor in scene.walker_actors]
        apply(native.worker, pedestrian_crossing_factor=1.0)
        replacement.assert_called_once()
        assert [actor.id for actor in scene.walker_actors] == ids


def test_cross_factor_failure_rolls_back_confirmed_factor_and_retry_refreshes(native: Any) -> None:
    native.worker.prepare({"walker_count": 1})
    scene = native.worker._scene
    controller = scene.walker_controllers[0]
    with mock.patch.object(native.world, "set_pedestrians_cross_factor", side_effect=[RuntimeError("factor failed"), None]):
        with pytest.raises(WorkerError, match="factor failed"):
            apply(native.worker, pedestrian_crossing_factor=0.9)
    assert scene.config.pedestrian_crossing_factor == 0.2
    assert native.world.cross_factor == 0.2
    with mock.patch.object(
        controller, "go_to_location", wraps=controller.go_to_location
    ) as reroute:
        apply(native.worker, pedestrian_crossing_factor=0.9)
        reroute.assert_not_called()
    assert scene.walker_controllers[0] is not controller
    assert scene.config.pedestrian_crossing_factor == 0.9


def test_slow_success_response_has_a_fresh_lease(native: Any) -> None:
    native.worker.prepare({})
    with mock.patch.object(
        native.worker, "_apply_weather", side_effect=lambda *args: native.clock.advance(30)
    ):
        response = apply(native.worker, weather_preset="heavy-rain")
    assert response["scene"]["lease_expires_in_seconds"] == 5


def test_unobserved_batch_growth_keeps_ids_for_rollback_and_close() -> None:
    harness = make_population_harness()
    worker = harness.worker
    destroyed = install_command_cleanup(harness)
    try:
        worker.prepare({"traffic_count": 2})
        original = set(harness.world.actors)
        harness.world.delay_batch_visibility = True
        harness.world.wait_for_tick = lambda seconds: None
        with pytest.raises(WorkerError, match="not visible"):
            apply(worker, traffic_count=4)
        assert set(harness.world.actors) == original
        assert len(destroyed) == 2
        assert worker._scene.config.traffic_count == 2
        assert worker._scene.status == "prepared"
    finally:
        worker.close()
    assert not harness.world.actors


def test_configure_http_endpoint_is_authenticated_and_returns_scene(native: Any) -> None:
    native.worker.prepare({})
    scene = native.worker._scene
    token = "test-world-worker-auth-token"
    server = create_server(worker=native.worker, bind="127.0.0.1", port=0, token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address)
    try:
        payload = {
            **scene.config.as_dict(),
            "lease_token": scene.lease_token,
            "speed_difference_percent": -10,
        }
        connection.request(
            "POST",
            f"/v1/scenes/{scene.scene_id}/configure",
            json.dumps(payload),
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        result = connection.getresponse()
        body = json.loads(result.read())
        assert result.status == 200
        assert body["scene"]["config"]["speed_difference_percent"] == -10
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
