from __future__ import annotations

import fnmatch
import http.client
import json
import os
import subprocess
import threading
import unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

from carla_vision.native.world_worker import (
    CompressedCameraConfig,
    CompressedCameraRelay,
    SceneConfig,
    WorkerError,
    WorldWorker,
    create_server,
)
from carla_vision.operator.situations import WEATHER_PRESETS


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class CompressedCameraRelayTests(unittest.TestCase):
    def test_relay_encodes_in_memory_and_retains_only_the_latest_jpeg(self) -> None:
        class Sensor:
            id = 71
            callback: Any = None
            stopped = False

            def listen(self, callback: Any) -> None:
                self.callback = callback

            def stop(self) -> None:
                self.stopped = True

        class Image:
            width = 1920
            height = 1080
            fov = 65.0
            raw_data = b"unused-by-injected-test-encoder"
            transform = FakeTransform(
                FakeLocation(1.0, 2.0, 3.0),
                FakeRotation(-4.0, 5.0, 6.0),
            )

            def __init__(self, frame: int) -> None:
                self.frame = frame
                self.timestamp = frame / 10.0

            def save_to_disk(self, path: str) -> None:
                raise AssertionError(f"disk encoding must not be used: {path}")

        clock = FakeClock()
        first_started = threading.Event()
        release_first = threading.Event()

        def encode(image: Image, quality: int) -> bytes:
            self.assertEqual(quality, 90)
            if image.frame == 19:
                first_started.set()
                self.assertTrue(release_first.wait(1.0))
            clock.advance(0.02)
            return b"\xff\xd8frame-" + str(image.frame).encode("ascii") + b"\xff\xd9"

        sensor = Sensor()
        relay = CompressedCameraRelay(sensor, jpeg_encoder=encode, clock=clock)
        relay.listen()
        assert sensor.callback is not None
        sensor.callback(Image(19))
        self.assertTrue(first_started.wait(1.0))
        sensor.callback(Image(20))
        sensor.callback(Image(21))
        release_first.set()

        sequence, payload, metadata = relay.wait(0, 1.0)
        self.assertEqual(sequence, 1)
        self.assertEqual(payload, b"\xff\xd8frame-21\xff\xd9")
        self.assertEqual(metadata["frame"], 21)
        self.assertEqual(metadata["width"], 1920)
        self.assertEqual(metadata["transform"]["location"]["x"], 1.0)
        telemetry = relay.snapshot()["telemetry"]
        self.assertEqual(telemetry["frames_received"], 3)
        self.assertEqual(telemetry["frames_encoded"], 2)
        self.assertEqual(telemetry["frames_dropped_pending"], 1)
        self.assertEqual(telemetry["frames_replaced"], 1)
        self.assertEqual(telemetry["average_encode_ms"], 20.0)
        self.assertEqual(telemetry["actual_fps_5s"], 50.0)
        self.assertEqual(
            telemetry["encoded_bytes_total"],
            len(b"\xff\xd8frame-19\xff\xd9") + len(b"\xff\xd8frame-21\xff\xd9"),
        )

        relay.close()
        self.assertTrue(sensor.stopped)

    def test_compressed_camera_accepts_sixty_fps_and_rejects_more(self) -> None:
        payload = {
            "lease_token": "lease",
            "mode": "drive",
            "width": 1280,
            "height": 720,
            "fps": 60.0,
            "fov": 90.0,
        }
        self.assertEqual(CompressedCameraConfig.from_mapping(payload).fps, 60.0)
        with self.assertRaisesRegex(WorkerError, r"\[1.0, 60.0\]"):
            CompressedCameraConfig.from_mapping({**payload, "fps": 60.1})

    def test_listen_failure_stops_sensor_and_closes_relay(self) -> None:
        class Sensor:
            id = 72
            stopped = False

            def listen(self, callback: Any) -> None:
                del callback
                raise RuntimeError("sensor callback registration failed")

            def stop(self) -> None:
                self.stopped = True

        sensor = Sensor()
        relay = CompressedCameraRelay(
            sensor,
            jpeg_encoder=lambda image, quality: b"",
        )

        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            relay.listen()

        self.assertTrue(sensor.stopped)
        relay.close()


class FakeAttribute:
    def __init__(self, value: str, recommended: list[str] | None = None) -> None:
        self.value = value
        self.recommended_values = list(recommended or [])

    def __str__(self) -> str:
        return self.value


class FakeBlueprint:
    def __init__(self, identifier: str, attributes: dict[str, FakeAttribute] | None = None) -> None:
        self.id = identifier
        self.attributes = attributes or {}

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def get_attribute(self, name: str) -> FakeAttribute:
        return self.attributes[name]

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name].value = str(value)


class FakeBlueprintLibrary:
    def __init__(self) -> None:
        colors = ["255,0,0", "0,0,255"]
        self.blueprints = {
            "vehicle.tesla.model3": FakeBlueprint(
                "vehicle.tesla.model3",
                {
                    "role_name": FakeAttribute(""),
                    "color": FakeAttribute(colors[0], colors),
                    "driver_id": FakeAttribute("0", ["0", "1"]),
                    "base_type": FakeAttribute("car"),
                },
            ),
            "vehicle.audi.tt": FakeBlueprint(
                "vehicle.audi.tt",
                {
                    "role_name": FakeAttribute(""),
                    "color": FakeAttribute(colors[1], colors),
                    "driver_id": FakeAttribute("0", ["0", "1"]),
                    "base_type": FakeAttribute("car"),
                },
            ),
            "walker.pedestrian.0001": FakeBlueprint(
                "walker.pedestrian.0001",
                {
                    "role_name": FakeAttribute(""),
                    "is_invincible": FakeAttribute("true"),
                    "speed": FakeAttribute("1.4", ["0.0", "1.4", "2.8"]),
                },
            ),
            "controller.ai.walker": FakeBlueprint("controller.ai.walker"),
            "static.prop.trafficcone01": FakeBlueprint("static.prop.trafficcone01"),
            "static.prop.trafficcone02": FakeBlueprint("static.prop.trafficcone02"),
            "static.prop.warningconstruction": FakeBlueprint("static.prop.warningconstruction"),
            "static.prop.streetbarrier": FakeBlueprint("static.prop.streetbarrier"),
            "static.prop.warningaccident": FakeBlueprint("static.prop.warningaccident"),
            "static.prop.dirtdebris01": FakeBlueprint("static.prop.dirtdebris01"),
        }

    def find(self, identifier: str) -> FakeBlueprint:
        if identifier not in self.blueprints:
            raise KeyError(identifier)
        return self.blueprints[identifier]

    def filter(self, pattern: str) -> list[FakeBlueprint]:
        return [
            blueprint
            for identifier, blueprint in self.blueprints.items()
            if fnmatch.fnmatch(identifier, pattern)
        ]


@dataclass
class FakeLocation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance(self, other: Any) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


@dataclass
class FakeRotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class FakeTransform:
    def __init__(
        self,
        location: FakeLocation | None = None,
        rotation: FakeRotation | None = None,
    ) -> None:
        self.location = location or FakeLocation()
        self.rotation = rotation or FakeRotation()


class FakeVehicleControl:
    def __init__(
        self,
        *,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
    ) -> None:
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse


class FakeWeather:
    def __init__(self, marker: str = "initial") -> None:
        self.marker = marker
        for preset in WEATHER_PRESETS.values():
            for name, value in preset.items():
                if name != "light":
                    setattr(self, name, float(value))


class FakeActor:
    def __init__(
        self,
        world: FakeWorld,
        actor_id: int,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
    ) -> None:
        self.world = world
        self.id = actor_id
        self.type_id = blueprint.id
        self.transform = transform
        self.attributes = {name: str(attribute) for name, attribute in blueprint.attributes.items()}
        self.is_alive = True
        self.destroyed = False
        self.controls: list[FakeVehicleControl] = []
        self.autopilot = False
        self.autopilot_port: int | None = None
        self.started = False
        self.stopped = False
        self.destination: FakeLocation | None = None
        self.maximum_speed: float | None = None

    def apply_control(self, control: FakeVehicleControl) -> None:
        self.controls.append(control)

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot = enabled
        self.autopilot_port = port

    def get_location(self) -> FakeLocation:
        return self.transform.location

    def get_transform(self) -> FakeTransform:
        return self.transform

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def go_to_location(self, location: FakeLocation) -> None:
        self.destination = location

    def set_max_speed(self, speed: float) -> None:
        self.maximum_speed = speed

    def destroy(self) -> None:
        self.destroyed = True
        self.is_alive = False
        self.world.actors.pop(self.id, None)


class FakeMap:
    def __init__(self, name: str) -> None:
        self.name = f"/Game/Carla/Maps/{name}/{name}"
        self.spawn_points = [
            FakeTransform(FakeLocation(float(index * 50), float(index * 3), 0.5))
            for index in range(8)
        ]

    def get_spawn_points(self) -> list[FakeTransform]:
        return list(self.spawn_points)


class FakeSettings:
    def __init__(self, synchronous: bool = False) -> None:
        self.synchronous_mode = synchronous


class FakeWorld:
    _world_ids = 1

    def __init__(
        self,
        name: str,
        library: FakeBlueprintLibrary,
        *,
        synchronous: bool = False,
    ) -> None:
        self.id = FakeWorld._world_ids
        FakeWorld._world_ids += 1
        self.map = FakeMap(name)
        self.library = library
        self.settings = FakeSettings(synchronous)
        self.weather = FakeWeather()
        self.actors: dict[int, FakeActor] = {}
        self.next_actor_id = self.id * 1000
        self.spectator = type("Spectator", (), {"id": self.id * 100 + 1})()
        self.walker_seed: int | None = None
        self.cross_factor: float | None = None
        self.wait_count = 0

    def get_map(self) -> FakeMap:
        return self.map

    def get_blueprint_library(self) -> FakeBlueprintLibrary:
        return self.library

    def get_settings(self) -> FakeSettings:
        return self.settings

    def get_weather(self) -> FakeWeather:
        return self.weather

    def set_weather(self, weather: FakeWeather) -> None:
        self.weather = weather

    def get_spectator(self) -> Any:
        return self.spectator

    def set_pedestrians_seed(self, seed: int) -> None:
        self.walker_seed = seed

    def set_pedestrians_cross_factor(self, factor: float) -> None:
        self.cross_factor = factor

    def try_spawn_actor(
        self,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
        attach_to: FakeActor | None = None,
    ) -> FakeActor:
        del attach_to
        actor = FakeActor(self, self.next_actor_id, blueprint, transform)
        self.next_actor_id += 1
        self.actors[actor.id] = actor
        return actor

    def get_actor(self, actor_id: int) -> FakeActor | None:
        return self.actors.get(actor_id)

    def get_actors(self, actor_ids: list[int]) -> list[FakeActor]:
        return [self.actors[actor_id] for actor_id in actor_ids if actor_id in self.actors]

    def get_random_location_from_navigation(self) -> FakeLocation:
        return FakeLocation(float(self.next_actor_id % 100), 7.0, 0.5)

    def wait_for_tick(self, seconds: float) -> None:
        del seconds
        self.wait_count += 1


class FakeTrafficManager:
    def __init__(self, port: int) -> None:
        self.port = port
        self.synchronous = False
        self.seed: int | None = None
        self.paths: list[tuple[int, list[FakeLocation]]] = []
        self.lights: list[int] = []
        self.shutdown = False

    def get_port(self) -> int:
        return self.port

    def set_synchronous_mode(self, enabled: bool) -> None:
        self.synchronous = enabled

    def set_random_device_seed(self, seed: int) -> None:
        self.seed = seed

    def set_global_distance_to_leading_vehicle(self, distance: float) -> None:
        self.distance = distance

    def global_percentage_speed_difference(self, difference: float) -> None:
        self.speed_difference = difference

    def update_vehicle_lights(self, actor: FakeActor, enabled: bool) -> None:
        if enabled:
            self.lights.append(actor.id)

    def set_path(self, actor: FakeActor, locations: list[FakeLocation]) -> None:
        self.paths.append((actor.id, list(locations)))

    def shut_down(self) -> None:
        self.shutdown = True


class FakeClient:
    def __init__(
        self,
        world: FakeWorld,
        library: FakeBlueprintLibrary,
        traffic_manager: FakeTrafficManager,
    ) -> None:
        self.world = world
        self.library = library
        self.traffic_manager = traffic_manager
        self.timeout: float | None = None
        self.loaded_maps: list[str] = []
        self.traffic_manager_calls = 0

    def set_timeout(self, timeout: float) -> None:
        self.timeout = timeout

    def get_client_version(self) -> str:
        return "0.9.16"

    def get_server_version(self) -> str:
        return "0.9.16"

    def get_world(self) -> FakeWorld:
        return self.world

    def get_available_maps(self) -> list[str]:
        return [
            "/Game/Carla/Maps/Town10HD_Opt/Town10HD_Opt",
            "/Game/Carla/Maps/Town05/Town05",
        ]

    def load_world(self, name: str) -> FakeWorld:
        short_name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        self.loaded_maps.append(short_name)
        self.world = FakeWorld(short_name, self.library)
        return self.world

    def get_trafficmanager(self, port: int) -> FakeTrafficManager:
        self.traffic_manager_calls += 1
        if port != self.traffic_manager.port:
            raise ValueError("wrong Traffic Manager port")
        return self.traffic_manager


class FakeCarla:
    Location = FakeLocation
    Rotation = FakeRotation
    Transform = FakeTransform
    VehicleControl = FakeVehicleControl
    WeatherParameters = FakeWeather

    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.client_calls = 0

    def Client(self, host: str, port: int) -> FakeClient:  # noqa: N802
        self.client_calls += 1
        self.last_endpoint = (host, port)
        return self.client


class FakeWaypoint:
    def __init__(self, location: FakeLocation) -> None:
        self.transform = FakeTransform(location)


class FakePlanner:
    def __init__(self, map_object: FakeMap) -> None:
        self.map_object = map_object

    def trace_route(
        self,
        origin: FakeLocation,
        destination: FakeLocation,
    ) -> list[tuple[FakeWaypoint, str]]:
        middle = FakeLocation(
            (origin.x + destination.x) / 2.0,
            (origin.y + destination.y) / 2.0,
            (origin.z + destination.z) / 2.0,
        )
        return [
            (FakeWaypoint(origin), "LANEFOLLOW"),
            (FakeWaypoint(middle), "LANEFOLLOW"),
            (FakeWaypoint(destination), "LANEFOLLOW"),
        ]


class WorldWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.library = FakeBlueprintLibrary()
        self.world = FakeWorld("Town10HD_Opt", self.library)
        self.traffic_manager = FakeTrafficManager(8000)
        self.client = FakeClient(self.world, self.library, self.traffic_manager)
        self.carla = FakeCarla(self.client)
        self.loader_calls = 0

        def load_carla() -> FakeCarla:
            self.loader_calls += 1
            return self.carla

        self.worker = WorldWorker(
            carla_loader=load_carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            lease_seconds=5.0,
            control_timeout=0.5,
            start_monitor=False,
        )

    def tearDown(self) -> None:
        self.worker.close()

    @staticmethod
    def lease(response: dict[str, Any]) -> tuple[str, str]:
        scene = response["scene"]
        return str(scene["scene_id"]), str(scene["lease_token"])

    def test_scene_config_defaults_and_strict_allow_list(self) -> None:
        config = SceneConfig.from_mapping({})
        self.assertEqual(config.map_name, "current")
        self.assertEqual(config.route_mode, "free")
        self.assertEqual(config.initial_control_mode, "manual")
        self.assertEqual(config.pedestrian_crossing_factor, 0.2)
        self.assertEqual(config.speed_difference_percent, 12.0)
        self.assertEqual(config.following_distance_metres, 2.0)
        tuned = SceneConfig.from_mapping(
            {
                "pedestrian_crossing_factor": 0.85,
                "speed_difference_percent": -20.0,
                "following_distance_metres": 8.5,
            }
        )
        self.assertEqual(tuned.pedestrian_crossing_factor, 0.85)
        self.assertEqual(tuned.speed_difference_percent, -20.0)
        self.assertEqual(tuned.following_distance_metres, 8.5)
        with self.assertRaisesRegex(WorkerError, "unknown fields"):
            SceneConfig.from_mapping({"shell": "rm"})
        with self.assertRaisesRegex(WorkerError, "route_mode"):
            SceneConfig.from_mapping({"route_mode": "wander"})

    def test_health_and_catalog_load_carla_lazily(self) -> None:
        self.assertEqual(self.loader_calls, 0)
        health = self.worker.health()
        self.assertTrue(health["ready"])
        self.assertEqual(self.loader_calls, 1)
        self.assertNotIn("token", json.dumps(health))
        catalog = self.worker.catalog()
        self.assertEqual(catalog["maps"][0].keys(), {"id", "label"})
        self.assertEqual(catalog["vehicles"][0].keys(), {"id", "label", "colors"})
        self.assertTrue(catalog["capabilities"]["random_route"])
        self.assertTrue(catalog["capabilities"]["asynchronous_world"])
        self.assertTrue(catalog["capabilities"]["world_dynamics_controls"])
        self.assertTrue(catalog["capabilities"]["garage_camera_presets"])
        self.assertEqual(catalog["worker_api_revision"], 7)
        self.assertTrue(catalog["capabilities"]["spawn_point_selection"])
        self.assertTrue(catalog["capabilities"]["selected_route"])
        self.assertEqual(catalog["spawn_point_map"], "Town10HD_Opt")
        self.assertEqual(catalog["spawn_count"], len(self.world.map.spawn_points))
        self.assertEqual(catalog["spawn_points"][0]["index"], 0)
        self.assertTrue(catalog["capabilities"]["batch_scene_cleanup"])
        self.assertEqual(catalog["carla"]["current_map"], "Town10HD_Opt")

    def test_selected_start_and_destination_use_exact_official_spawn_points(self) -> None:
        prepared = self.worker.prepare(
            {
                "start_spawn_index": 2,
                "route_mode": "selected_destination",
                "destination_spawn_index": 6,
                "initial_control_mode": "autopilot",
            }
        )
        scene = prepared["scene"]
        self.assertEqual(scene["spawn_index"], 2)
        self.assertEqual(scene["destination"]["spawn_index"], 6)
        self.assertEqual(scene["route"]["mode"], "selected_destination")
        self.assertTrue(scene["route"]["planned"])
        scene_id, lease_token = self.lease(prepared)
        started = self.worker.start(scene_id, {"lease_token": lease_token})
        self.assertTrue(started["scene"]["route"]["enforced"])
        self.assertEqual(self.traffic_manager.paths[-1][0], scene["ego_actor_id"])
        self.assertEqual(
            self.traffic_manager.paths[-1][1][-1],
            self.world.map.spawn_points[6].location,
        )

    def test_selected_start_never_silently_falls_back_when_occupied(self) -> None:
        original_try_spawn = self.world.try_spawn_actor

        def occupied(blueprint: Any, transform: Any, *args: Any, **kwargs: Any) -> Any:
            if transform is self.world.map.spawn_points[3] and str(blueprint.id).startswith("vehicle."):
                return None
            return original_try_spawn(blueprint, transform, *args, **kwargs)

        self.world.try_spawn_actor = occupied  # type: ignore[method-assign]
        with self.assertRaisesRegex(WorkerError, "no fallback was used") as raised:
            self.worker.prepare({"start_spawn_index": 3})
        self.assertEqual(raised.exception.code, "ego_spawn_unavailable")
        self.assertEqual(self.world.actors, {})

    def test_selected_route_contract_rejects_missing_or_same_destination(self) -> None:
        with self.assertRaisesRegex(WorkerError, "requires destination_spawn_index"):
            SceneConfig.from_mapping({"route_mode": "selected_destination"})
        with self.assertRaisesRegex(WorkerError, "must differ"):
            SceneConfig.from_mapping(
                {
                    "route_mode": "selected_destination",
                    "start_spawn_index": 4,
                    "destination_spawn_index": 4,
                }
            )

    def test_health_reports_busy_without_waiting_for_world_lock(self) -> None:
        acquired = threading.Event()
        release = threading.Event()

        def hold_world_lock() -> None:
            with self.worker._lock:
                acquired.set()
                release.wait(timeout=2.0)

        thread = threading.Thread(target=hold_world_lock)
        thread.start()
        self.assertTrue(acquired.wait(timeout=1.0))
        try:
            health = self.worker.health()
        finally:
            release.set()
            thread.join(timeout=2.0)

        self.assertEqual(health["status"], "busy")
        self.assertTrue(health["ready"])
        self.assertTrue(health["capabilities"]["nonblocking_health"])

    def test_population_capacity_fails_before_returning_a_partial_scene(self) -> None:
        with self.assertRaisesRegex(WorkerError, "maximum of 7") as raised:
            self.worker.prepare({"traffic_count": 8})

        self.assertEqual(raised.exception.code, "scene_population_capacity")
        self.assertIsNone(self.worker.current_scene()["scene"])
        self.assertEqual(self.world.actors, {})

    def test_dense_walker_activation_retries_a_lost_registry_pair(self) -> None:
        original_start = FakeActor.start
        failed_once = False

        def flaky_start(actor: FakeActor) -> None:
            nonlocal failed_once
            if actor.type_id == "controller.ai.walker" and not failed_once:
                failed_once = True
                raise RuntimeError("Actor could not be found in the registry")
            original_start(actor)

        with mock.patch.object(FakeActor, "start", flaky_start):
            prepared = self.worker.prepare({"walker_count": 1})

        self.assertTrue(failed_once)
        self.assertEqual(prepared["scene"]["walker_count"], 1)
        controllers = [
            actor for actor in self.world.actors.values() if actor.type_id == "controller.ai.walker"
        ]
        self.assertEqual(len(controllers), 1)
        self.assertTrue(controllers[0].started)

    def test_120_vehicle_and_120_walker_population_is_exact(self) -> None:
        self.world.map.spawn_points = [
            FakeTransform(FakeLocation(float(index * 8), float(index % 7) * 3.0, 0.5))
            for index in range(130)
        ]
        prepared = self.worker.prepare({"traffic_count": 120, "walker_count": 120})

        self.assertEqual(prepared["scene"]["traffic_count_requested"], 120)
        self.assertEqual(prepared["scene"]["traffic_count"], 120)
        self.assertEqual(prepared["scene"]["walker_count_requested"], 120)
        self.assertEqual(prepared["scene"]["walker_count"], 120)
        self.assertEqual(self.world.wait_count, 5)

    def test_failed_walker_tick_barrier_cleans_every_pair_and_allows_retry(self) -> None:
        original_wait = self.world.wait_for_tick

        def fail_wait(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("tick timeout")

        self.world.wait_for_tick = fail_wait  # type: ignore[method-assign]
        with self.assertRaisesRegex(WorkerError, "walkers 0/1") as raised:
            self.worker.prepare({"walker_count": 1})

        self.assertEqual(raised.exception.code, "scene_population_shortfall")
        self.assertIsNone(self.worker.current_scene()["scene"])
        self.assertEqual(self.world.actors, {})

        self.world.wait_for_tick = original_wait  # type: ignore[method-assign]
        prepared = self.worker.prepare({"walker_count": 1})
        self.assertEqual(prepared["scene"]["walker_count"], 1)

    def test_failed_walker_activation_retains_actors_until_cleanup_confirms_destroy(
        self,
    ) -> None:
        original_destroy = FakeActor.destroy
        destroy_attempts: dict[int, int] = {}
        first_failure_modes: dict[int, str] = {}

        def reject_controller_start(actor: FakeActor) -> None:
            if actor.type_id == "controller.ai.walker":
                raise RuntimeError("controller activation failed")
            actor.started = True

        def flaky_destroy(actor: FakeActor) -> None | bool:
            if actor.type_id != "controller.ai.walker":
                return original_destroy(actor)
            attempt = destroy_attempts.get(actor.id, 0) + 1
            destroy_attempts[actor.id] = attempt
            if attempt == 1:
                mode = "false" if len(first_failure_modes) % 2 == 0 else "exception"
                first_failure_modes[actor.id] = mode
                if mode == "false":
                    return False
                raise RuntimeError("transient destroy failure")
            return original_destroy(actor)

        with (
            mock.patch.object(FakeActor, "start", reject_controller_start),
            mock.patch.object(FakeActor, "destroy", flaky_destroy),
            self.assertRaisesRegex(WorkerError, "rollback was not confirmed") as raised,
        ):
            self.worker.prepare({"walker_count": 1})

        self.assertEqual(raised.exception.code, "scene_prepare_failed")
        # Do not replenish a failed pair while its controller still exists.
        self.assertEqual(set(first_failure_modes.values()), {"false"})
        self.assertTrue(all(destroy_attempts[actor_id] >= 2 for actor_id in first_failure_modes))
        self.assertIsNone(self.worker.current_scene()["scene"])
        self.assertEqual(self.world.actors, {})

    def test_stop_reports_actor_destroy_that_remains_unconfirmed(self) -> None:
        prepared = self.worker.prepare({"walker_count": 1})
        scene_id, lease_token = self.lease(prepared)
        controller = next(
            actor for actor in self.world.actors.values() if actor.type_id == "controller.ai.walker"
        )
        original_destroy = controller.destroy
        controller.destroy = lambda: False  # type: ignore[method-assign]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertIn(controller.id, self.world.actors)
        self.assertTrue(
            any(
                f"destroy walker_controller {controller.id}" in value
                and "remains registered" in value
                for value in stopped["scene"]["cleanup_errors"]
            )
        )
        original_destroy()

    def test_stop_is_idempotent_after_a_lost_response(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 1, "walker_count": 1})
        scene_id, lease_token = self.lease(prepared)

        first = self.worker.stop(scene_id, {"lease_token": lease_token})
        second = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertEqual(second, first)
        self.assertEqual(second["status"], "stopped")

    def test_dense_stop_batches_unique_live_owned_actors_once(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 7, "walker_count": 7})
        scene_id, lease_token = self.lease(prepared)
        scene = self.worker._scene
        assert scene is not None
        scene.owned_actors.append(scene.owned_actors[-1])
        absent = next(owned for owned in scene.owned_actors if owned.kind == "traffic")
        absent.actor.destroy()
        batches: list[tuple[list[int], bool]] = []
        individual_lookups: list[int] = []

        class Command:
            @staticmethod
            def DestroyActor(actor_id: int) -> int:  # noqa: N802
                return actor_id

        class Response:
            error = ""

        def apply_batch_sync(commands: list[int], due_tick_cue: bool) -> list[Response]:
            batches.append((list(commands), due_tick_cue))
            for actor_id in commands:
                actor = self.world.actors.get(actor_id)
                self.assertIsNotNone(actor)
                assert actor is not None
                self.assertTrue(actor.is_alive)
                actor.destroy()
            return [Response() for _ in commands]

        def reject_individual_lookup(actor_id: int) -> FakeActor | None:
            individual_lookups.append(actor_id)
            raise AssertionError("batch cleanup must use World.get_actors(ids)")

        self.carla.command = Command  # type: ignore[attr-defined]
        self.client.apply_batch_sync = apply_batch_sync  # type: ignore[attr-defined]
        self.world.get_actor = reject_individual_lookup  # type: ignore[method-assign]

        first = self.worker.stop(scene_id, {"lease_token": lease_token})
        second = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertEqual(second, first)
        self.assertEqual(len(batches), 1)
        destroyed_ids, due_tick_cue = batches[0]
        self.assertFalse(due_tick_cue)
        self.assertEqual(len(destroyed_ids), len(set(destroyed_ids)))
        self.assertNotIn(absent.actor_id, destroyed_ids)
        self.assertEqual(individual_lookups, [])
        self.assertEqual(self.world.actors, {})
        self.assertEqual(first["scene"]["cleanup_errors"], [])

    def test_batch_destroy_waits_for_async_snapshot_before_confirmation(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 2, "walker_count": 2})
        scene_id, lease_token = self.lease(prepared)
        pending_destroy: list[int] = []
        barrier_calls: list[float] = []

        class Command:
            @staticmethod
            def DestroyActor(actor_id: int) -> int:  # noqa: N802
                return actor_id

        class Response:
            error = ""

        def apply_batch_sync(commands: list[int], due_tick_cue: bool) -> list[Response]:
            self.assertFalse(due_tick_cue)
            pending_destroy.extend(commands)
            return [Response() for _ in commands]

        def wait_for_tick(seconds: float) -> None:
            barrier_calls.append(seconds)
            for actor_id in pending_destroy:
                actor = self.world.actors.get(actor_id)
                if actor is not None:
                    actor.destroy()

        self.carla.command = Command  # type: ignore[attr-defined]
        self.client.apply_batch_sync = apply_batch_sync  # type: ignore[attr-defined]
        self.world.wait_for_tick = wait_for_tick  # type: ignore[method-assign]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertEqual(len(barrier_calls), 1)
        self.assertLessEqual(barrier_calls[0], 5.0)
        self.assertEqual(self.world.actors, {})
        self.assertEqual(stopped["scene"]["cleanup_errors"], [])

    def test_successful_batch_is_not_rejected_when_snapshot_barrier_times_out(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 2})
        scene_id, lease_token = self.lease(prepared)

        class Command:
            @staticmethod
            def DestroyActor(actor_id: int) -> int:  # noqa: N802
                return actor_id

        class Response:
            error = ""

        def apply_batch_sync(commands: list[int], due_tick_cue: bool) -> list[Response]:
            self.assertFalse(due_tick_cue)
            # The authoritative server response succeeded, but this fake
            # deliberately retains the old client-side actor snapshot.
            return [Response() for _ in commands]

        def wait_for_tick(seconds: float) -> None:
            self.assertLessEqual(seconds, 5.0)
            raise TimeoutError("next episode snapshot did not arrive")

        self.carla.command = Command  # type: ignore[attr-defined]
        self.client.apply_batch_sync = apply_batch_sync  # type: ignore[attr-defined]
        self.world.wait_for_tick = wait_for_tick  # type: ignore[method-assign]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertEqual(len(stopped["scene"]["cleanup_errors"]), 1)
        self.assertIn("snapshot barrier failed", stopped["scene"]["cleanup_errors"][0])

    def test_stop_uses_owned_proxies_when_actor_lookup_state_is_unknown(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 1})
        scene_id, lease_token = self.lease(prepared)

        def fail_bulk_lookup(actor_ids: list[int]) -> list[FakeActor]:
            del actor_ids
            raise TimeoutError("bulk actor lookup stalled")

        def fail_actor_lookup(actor_id: int) -> FakeActor | None:
            del actor_id
            raise TimeoutError("actor lookup stalled")

        self.world.get_actors = fail_bulk_lookup  # type: ignore[method-assign]
        self.world.get_actor = fail_actor_lookup  # type: ignore[method-assign]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertEqual(self.world.actors, {})
        self.assertTrue(
            any(
                "ego stop lookup failed" in value
                and "bulk actor lookup stalled" in value
                for value in stopped["scene"]["cleanup_errors"]
            )
        )

    def test_batch_destroy_reports_only_errors_for_confirmed_survivors(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 1})
        scene_id, lease_token = self.lease(prepared)
        scene = self.worker._scene
        assert scene is not None
        survivor = next(owned for owned in scene.owned_actors if owned.kind == "traffic")

        class Command:
            @staticmethod
            def DestroyActor(actor_id: int) -> int:  # noqa: N802
                return actor_id

        class Response:
            def __init__(self, error: str) -> None:
                self.error = error

        def apply_batch_sync(commands: list[int], due_tick_cue: bool) -> list[Response]:
            self.assertFalse(due_tick_cue)
            responses: list[Response] = []
            for actor_id in commands:
                if actor_id == survivor.actor_id:
                    responses.append(Response("actor remained registered"))
                    continue
                actor = self.world.actors[actor_id]
                actor.destroy()
                responses.append(Response("stale server warning"))
            return responses

        self.carla.command = Command  # type: ignore[attr-defined]
        self.client.apply_batch_sync = apply_batch_sync  # type: ignore[attr-defined]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertIn(survivor.actor_id, self.world.actors)
        matching = [
            value
            for value in stopped["scene"]["cleanup_errors"]
            if value.startswith("destroy ")
        ]
        self.assertEqual(
            matching,
            [
                f"destroy traffic {survivor.actor_id}: actor remained registered",
            ],
        )
        survivor.actor.destroy()

    def test_batch_destroy_accepts_authoritative_absence_with_stale_snapshot(self) -> None:
        from types import SimpleNamespace

        prepared = self.worker.prepare({"traffic_count": 1})
        scene_id, lease_token = self.lease(prepared)
        self.carla.command = SimpleNamespace(DestroyActor=lambda actor_id: actor_id)
        self.client.apply_batch_sync = lambda commands, tick: [
            SimpleNamespace(error="unable to destroy actor: not found") for _ in commands
        ]
        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})
        self.assertEqual(stopped["scene"]["cleanup_errors"], [])
        # The fake client snapshot deliberately lags the authoritative response.
        for actor in list(self.world.actors.values()):
            actor.destroy()

    def test_batch_destroy_refuses_actor_whose_owned_identity_changed(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 1})
        scene_id, lease_token = self.lease(prepared)
        scene = self.worker._scene
        assert scene is not None
        replaced = next(owned for owned in scene.owned_actors if owned.kind == "traffic")
        replaced.actor.type_id = "vehicle.replaced"
        destroyed_ids: list[int] = []

        class Command:
            @staticmethod
            def DestroyActor(actor_id: int) -> int:  # noqa: N802
                return actor_id

        class Response:
            error = ""

        def apply_batch_sync(commands: list[int], due_tick_cue: bool) -> list[Response]:
            self.assertFalse(due_tick_cue)
            destroyed_ids.extend(commands)
            for actor_id in commands:
                self.world.actors[actor_id].destroy()
            return [Response() for _ in commands]

        self.carla.command = Command  # type: ignore[attr-defined]
        self.client.apply_batch_sync = apply_batch_sync  # type: ignore[attr-defined]

        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})

        self.assertNotIn(replaced.actor_id, destroyed_ids)
        self.assertIn(replaced.actor_id, self.world.actors)
        self.assertTrue(
            any(
                f"destroy traffic {replaced.actor_id}" in value
                and "type changed" in value
                for value in stopped["scene"]["cleanup_errors"]
            )
        )
        replaced.actor.destroy()

    def test_garage_autoframing_scales_with_vehicle_bounds(self) -> None:
        prepared = self.worker.prepare({})
        ego = self.world.get_actor(prepared["scene"]["ego_actor_id"])
        assert ego is not None

        class Bounds:
            location = FakeLocation(0.0, 0.0, 1.0)
            extent = FakeLocation(2.0, 1.0, 1.0)

        ego.bounding_box = Bounds()
        compact = self.worker._garage_camera_transform(
            ego,
            yaw=0.0,
            pitch=-8.0,
            distance=6.0,
            width=1280,
            height=720,
            fov=65.0,
            preset="front",
        )
        ego.bounding_box.extent = FakeLocation(6.0, 1.4, 1.8)
        long_vehicle = self.worker._garage_camera_transform(
            ego,
            yaw=0.0,
            pitch=-8.0,
            distance=6.0,
            width=1280,
            height=720,
            fov=65.0,
            preset="front",
        )

        self.assertGreater(long_vehicle.location.x, compact.location.x)
        self.assertEqual(long_vehicle.rotation.yaw, 180.0)
        top = self.worker._garage_camera_transform(
            ego,
            yaw=0.0,
            pitch=-25.0,
            distance=8.0,
            width=1280,
            height=720,
            fov=65.0,
            preset="top",
        )
        self.assertEqual(top.rotation.pitch, -70.0)

    def test_camera_orbit_rejects_an_attached_drive_camera(self) -> None:
        prepared = self.worker.prepare({})
        scene_id, lease_token = self.lease(prepared)

        class Sensor:
            def set_transform(self, transform: Any) -> None:
                del transform
                raise AssertionError("drive camera transform must remain untouched")

        class Relay:
            sensor = Sensor()

            def close(self) -> None:
                pass

        assert self.worker._scene is not None
        self.worker._scene.camera_relay = Relay()  # type: ignore[assignment]
        self.worker._scene.camera_config = CompressedCameraConfig(
            mode="drive",
            width=1280,
            height=720,
            fps=30.0,
            fov=90.0,
        )

        with self.assertRaisesRegex(WorkerError, "active Garage camera") as raised:
            self.worker.camera_orbit(
                scene_id,
                {
                    "lease_token": lease_token,
                    "yaw": 0.0,
                    "pitch": -8.0,
                    "distance": 6.0,
                    "preset": "front",
                },
            )

        self.assertEqual(raised.exception.code, "camera_mode_conflict")

    def test_camera_activation_failure_rolls_back_owned_sensor(self) -> None:
        self.library.blueprints["sensor.camera.rgb"] = FakeBlueprint(
            "sensor.camera.rgb",
            {"role_name": FakeAttribute("")},
        )

        class FailingSensor(FakeActor):
            def listen(self, callback: Any) -> None:
                del callback
                raise RuntimeError("listen failed")

        def spawn_actor(
            blueprint: FakeBlueprint,
            transform: FakeTransform,
            attach_to: FakeActor | None = None,
        ) -> FakeActor:
            del attach_to
            sensor = FailingSensor(
                self.world,
                self.world.next_actor_id,
                blueprint,
                transform,
            )
            self.world.next_actor_id += 1
            self.world.actors[sensor.id] = sensor
            return sensor

        self.world.spawn_actor = spawn_actor  # type: ignore[attr-defined]
        prepared = self.worker.prepare({})
        scene_id, lease_token = self.lease(prepared)
        before_ids = set(self.world.actors)
        with (
            mock.patch(
                "carla_vision.native.world_worker._load_in_memory_jpeg_encoder",
                return_value=lambda image, quality: b"",
            ),
            self.assertRaisesRegex(RuntimeError, "listen failed"),
        ):
            self.worker.camera(
                scene_id,
                {
                    "lease_token": lease_token,
                    "mode": "drive",
                    "width": 1280,
                    "height": 720,
                    "fps": 30.0,
                    "fov": 90.0,
                },
            )

        self.assertEqual(set(self.world.actors), before_ids)
        self.assertIsNone(self.worker.current_scene()["scene"]["camera"])

    def test_manual_deadman_lease_cleanup_and_owned_props(self) -> None:
        original_weather = self.world.weather
        prepared = self.worker.prepare(
            {
                "weather_preset": "heavy-rain",
                "color": "255,0,0",
                "seed": 42,
                "traffic_count": 2,
                "walker_count": 2,
                "prop_preset": "cones",
                "pedestrian_crossing_factor": 0.75,
                "speed_difference_percent": -15.0,
                "following_distance_metres": 6.5,
            }
        )
        scene_id, lease_token = self.lease(prepared)
        scene = prepared["scene"]
        self.assertEqual(scene["traffic_count"], 2)
        self.assertEqual(scene["traffic_count_requested"], 2)
        self.assertEqual(scene["walker_count"], 2)
        self.assertEqual(scene["walker_count_requested"], 2)
        self.assertEqual(scene["pedestrian_crossing_factor"], 0.75)
        self.assertEqual(scene["speed_difference_percent"], -15.0)
        self.assertEqual(scene["following_distance_metres"], 6.5)
        self.assertEqual(len(scene["prop_actor_ids"]), 2)
        self.assertEqual(scene["map_name"], "Town10HD_Opt")
        self.assertIsInstance(scene["episode_id"], int)
        self.assertEqual(self.client.loaded_maps, [])
        self.assertIsNot(self.world.weather, original_weather)
        self.assertEqual(self.world.cross_factor, 0.75)
        self.assertEqual(self.traffic_manager.speed_difference, -15.0)
        self.assertEqual(self.traffic_manager.distance, 6.5)

        started = self.worker.start(scene_id, {"lease_token": lease_token})
        ego = self.world.get_actor(started["scene"]["ego_actor_id"])
        assert ego is not None
        self.assertEqual(ego.controls[-1].brake, 1.0)
        controlled = self.worker.control(
            scene_id,
            {
                "lease_token": lease_token,
                "sequence": 1,
                "throttle": 0.4,
                "steer": -0.2,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
            },
        )
        self.assertFalse(controlled["scene"]["deadman_active"])
        self.assertEqual(ego.controls[-1].throttle, 0.4)

        self.clock.advance(0.6)
        self.worker.heartbeat(scene_id, {"lease_token": lease_token})
        self.worker.enforce_timeouts()
        self.assertEqual(ego.controls[-1].brake, 1.0)
        self.assertTrue(self.worker.current_scene()["scene"]["deadman_active"])

        all_owned = list(self.world.actors.values())
        self.clock.advance(5.1)
        self.worker.enforce_timeouts()
        self.assertIsNone(self.worker.current_scene()["scene"])
        self.assertTrue(all(actor.destroyed for actor in all_owned))
        self.assertIs(self.world.weather, original_weather)
        self.assertFalse(self.traffic_manager.synchronous)
        self.assertFalse(
            self.traffic_manager.shutdown,
            "scene cleanup must not call CARLA's unbounded Traffic Manager shutdown",
        )

    def test_traffic_manager_connection_is_reused_across_scene_leases(self) -> None:
        first = self.worker.prepare({"traffic_count": 1, "walker_count": 1})
        first_id, first_token = self.lease(first)
        self.worker.stop(first_id, {"lease_token": first_token})

        second = self.worker.prepare({"traffic_count": 1, "walker_count": 1})
        second_id, second_token = self.lease(second)
        self.worker.stop(second_id, {"lease_token": second_token})

        self.assertEqual(self.client.traffic_manager_calls, 2)
        self.assertFalse(self.traffic_manager.synchronous)
        self.assertFalse(self.traffic_manager.shutdown)
        self.assertIsNone(self.worker.current_scene()["scene"])

    def test_current_map_never_starts_isolated_loader(self) -> None:
        runner_calls: list[list[str]] = []

        def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            del kwargs
            runner_calls.append(command)
            raise AssertionError("the current map must not start a child")

        worker = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            start_monitor=False,
            map_process_runner=runner,
        )
        self.addCleanup(worker.close)

        prepared = worker.prepare({"map_name": "Town10HD_Opt"})

        self.assertEqual(prepared["scene"]["map_name"], "Town10HD_Opt")
        self.assertEqual(runner_calls, [])
        self.assertEqual(self.client.loaded_maps, [])

    def test_map_change_runs_in_child_without_bearer_token_and_reconnects(self) -> None:
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append((command, kwargs))
            self.assertEqual(self.client.traffic_manager_calls, 0)
            self.client.world = FakeWorld("Town05", self.library)
            return subprocess.CompletedProcess(command, 0, "", "")

        worker = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            start_monitor=False,
            map_process_runner=runner,
        )
        self.addCleanup(worker.close)
        previous_token = os.environ.get("CARLA_WORLD_WORKER_TOKEN")
        os.environ["CARLA_WORLD_WORKER_TOKEN"] = "not-for-map-child"
        self.addCleanup(
            lambda: (
                os.environ.pop("CARLA_WORLD_WORKER_TOKEN", None)
                if previous_token is None
                else os.environ.__setitem__("CARLA_WORLD_WORKER_TOKEN", previous_token)
            )
        )

        prepared = worker.prepare({"map_name": "Town05"})

        self.assertEqual(prepared["scene"]["map_name"], "Town05")
        self.assertEqual(self.client.loaded_maps, [])
        self.assertEqual(self.client.traffic_manager_calls, 1)
        self.assertGreaterEqual(self.carla.client_calls, 2)
        self.assertEqual(len(calls), 1)
        command, options = calls[0]
        self.assertIn("--internal-map-load", command)
        self.assertEqual(
            command[command.index("--target-map") + 1], "/Game/Carla/Maps/Town05/Town05"
        )
        self.assertNotIn("CARLA_WORLD_WORKER_TOKEN", options["env"])
        self.assertTrue(options["check"] is False)

    def test_child_crash_is_accepted_when_carla_reached_target_map(self) -> None:
        def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            del kwargs
            self.client.world = FakeWorld("Town05", self.library)
            return subprocess.CompletedProcess(command, -1073741819, "", "native crash")

        worker = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            start_monitor=False,
            map_process_runner=runner,
        )
        self.addCleanup(worker.close)

        prepared = worker.prepare({"map_name": "Town05"})

        self.assertEqual(prepared["scene"]["map_name"], "Town05")
        self.assertEqual(self.client.loaded_maps, [])

    def test_child_timeout_is_accepted_when_carla_reached_target_map(self) -> None:
        def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            self.client.world = FakeWorld("Town05", self.library)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        worker = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            start_monitor=False,
            map_process_runner=runner,
        )
        self.addCleanup(worker.close)

        prepared = worker.prepare({"map_name": "Town05"})

        self.assertEqual(prepared["scene"]["map_name"], "Town05")

    def test_child_failure_without_target_map_reports_map_load_failed(self) -> None:
        def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            del kwargs
            return subprocess.CompletedProcess(command, 1, "", "load failed")

        worker = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
            clock=self.clock,
            start_monitor=False,
            map_reconnect_seconds=1.0,
            map_process_runner=runner,
        )
        self.addCleanup(worker.close)

        with self.assertRaises(WorkerError) as caught:
            worker.prepare({"map_name": "Town05"})

        self.assertEqual(caught.exception.code, "map_load_failed")
        self.assertEqual(self.client.loaded_maps, [])

    def test_random_destination_is_enforced_and_mode_switches(self) -> None:
        prepared = self.worker.prepare(
            {
                "route_mode": "random_destination",
                "initial_control_mode": "autopilot",
                "seed": 99,
            }
        )
        scene_id, lease_token = self.lease(prepared)
        self.assertTrue(prepared["scene"]["route"]["planned"])
        self.assertFalse(prepared["scene"]["route"]["enforced"])
        self.assertIsNotNone(prepared["scene"]["destination"])

        started = self.worker.start(scene_id, {"lease_token": lease_token})
        ego = self.world.get_actor(started["scene"]["ego_actor_id"])
        assert ego is not None
        self.assertTrue(ego.autopilot)
        self.assertTrue(started["scene"]["route"]["enforced"])
        self.assertEqual(len(self.traffic_manager.paths), 1)

        manual = self.worker.mode(
            scene_id,
            {"lease_token": lease_token, "control_mode": "manual"},
        )
        self.assertFalse(ego.autopilot)
        self.assertTrue(manual["scene"]["deadman_active"])
        self.assertFalse(manual["scene"]["route"]["enforced"])
        autopilot = self.worker.mode(
            scene_id,
            {"lease_token": lease_token, "control_mode": "autopilot"},
        )
        self.assertTrue(ego.autopilot)
        self.assertTrue(autopilot["scene"]["route"]["enforced"])
        self.assertEqual(len(self.traffic_manager.paths), 2)
        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})
        self.assertEqual(stopped["status"], "stopped")
        self.assertTrue(stopped["scene"]["cleanup_guard_passed"])

    def test_random_destination_rejected_when_route_planner_is_missing(self) -> None:
        unavailable = WorldWorker(
            carla_loader=lambda: self.carla,
            route_planner_loader=lambda: None,
            clock=self.clock,
            start_monitor=False,
        )
        self.addCleanup(unavailable.close)
        catalog = unavailable.catalog()
        self.assertFalse(catalog["capabilities"]["random_route"])
        with self.assertRaisesRegex(WorkerError, "GlobalRoutePlanner"):
            unavailable.prepare({"route_mode": "random_destination"})
        self.assertFalse(self.traffic_manager.synchronous)
        self.assertFalse(self.traffic_manager.shutdown)

    def test_large_research_seed_is_bounded_for_carla_seed_apis(self) -> None:
        prepared = self.worker.prepare({"seed": 2**63 - 1})
        scene_id, lease_token = self.lease(prepared)

        self.assertGreaterEqual(self.traffic_manager.seed, 0)
        self.assertLess(self.traffic_manager.seed, 2**31 - 1)
        self.assertGreaterEqual(self.world.walker_seed, 0)
        self.assertLess(self.world.walker_seed, 2**31 - 1)

        self.worker.stop(scene_id, {"lease_token": lease_token})

    def test_stale_controls_wrong_lease_and_one_active_scene_are_rejected(self) -> None:
        prepared = self.worker.prepare({})
        scene_id, lease_token = self.lease(prepared)
        with self.assertRaisesRegex(WorkerError, "another leased scene"):
            self.worker.prepare({})
        with self.assertRaisesRegex(WorkerError, "lease does not match"):
            self.worker.start(scene_id, {"lease_token": "x" * 32})
        self.worker.start(scene_id, {"lease_token": lease_token})
        payload = {
            "lease_token": lease_token,
            "sequence": 4,
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 1.0,
            "hand_brake": False,
            "reverse": False,
        }
        self.worker.control(scene_id, payload)
        with self.assertRaisesRegex(WorkerError, "newer"):
            self.worker.control(scene_id, payload)
        with self.assertRaisesRegex(WorkerError, "unknown fields"):
            self.worker.heartbeat(scene_id, {"lease_token": lease_token, "rpc": "anything"})

    def test_episode_guard_refuses_to_destroy_replaced_world(self) -> None:
        prepared = self.worker.prepare({"traffic_count": 1})
        scene_id, lease_token = self.lease(prepared)
        old_actors = list(self.world.actors.values())
        self.client.world = FakeWorld("Town10HD_Opt", self.library)
        stopped = self.worker.stop(scene_id, {"lease_token": lease_token})
        self.assertFalse(stopped["scene"]["cleanup_guard_passed"])
        self.assertTrue(
            any("episode changed" in value for value in stopped["scene"]["cleanup_errors"])
        )
        self.assertTrue(all(not actor.destroyed for actor in old_actors))

    def test_synchronous_world_is_rejected_without_mutation(self) -> None:
        self.world.settings.synchronous_mode = True
        with self.assertRaisesRegex(WorkerError, "asynchronous CARLA world"):
            self.worker.prepare({})
        self.assertEqual(self.world.actors, {})

    def test_http_requires_bearer_and_returns_strict_error_envelope(self) -> None:
        token = "test-token-that-is-long-enough"
        server = create_server(
            bind="127.0.0.1",
            port=0,
            token=token,
            worker=self.worker,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host, port = server.server_address[:2]

        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request("GET", "/v1/health")
        response = connection.getresponse()
        unauthorized = json.loads(response.read())
        self.assertEqual(response.status, 401)
        self.assertEqual(unauthorized["error"]["code"], "unauthorized")
        connection.close()

        connection = http.client.HTTPConnection(host, port, timeout=3)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        connection.request("GET", "/v1/catalog", headers=headers)
        response = connection.getresponse()
        catalog = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertIn("maps", catalog)
        connection.request(
            "POST",
            "/v1/scenes/prepare",
            body=json.dumps({"command": "shell"}),
            headers=headers,
        )
        response = connection.getresponse()
        invalid = json.loads(response.read())
        self.assertEqual(response.status, 400)
        self.assertEqual(invalid["error"]["code"], "unknown_fields")
        connection.close()

    def test_http_mjpeg_stream_is_authenticated_persistent_and_newest_only(self) -> None:
        class StreamingWorker:
            def __init__(self) -> None:
                self.after_sequences: list[int] = []

            def camera_frame(
                self,
                scene_id: str,
                lease_token: str,
                *,
                after_sequence: int,
                timeout: float,
            ) -> tuple[int, bytes, dict[str, Any]]:
                self.assert_contract(scene_id, lease_token, timeout)
                self.after_sequences.append(after_sequence)
                sequence = len(self.after_sequences) - 1
                if sequence >= 2:
                    raise WorkerError(409, "camera_inactive", "test stream complete")
                return (
                    sequence,
                    b"\xff\xd8" + str(sequence).encode("ascii") + b"\xff\xd9",
                    {
                        "frame": 100 + sequence,
                        "timestamp": 1.25 + sequence,
                        "width": 1280,
                        "height": 720,
                        "fov": 90.0,
                        "transform": {"location": {"x": float(sequence)}},
                    },
                )

            @staticmethod
            def assert_contract(scene_id: str, lease_token: str, timeout: float) -> None:
                if scene_id != "scene_1234567890" or lease_token != "lease-secret":
                    raise AssertionError("handler did not forward scene credentials")
                if timeout != 5.0:
                    raise AssertionError("handler used an unexpected stream wait timeout")

        token = "test-token-that-is-long-enough"
        worker = StreamingWorker()
        server = create_server(
            bind="127.0.0.1",
            port=0,
            token=token,
            worker=worker,  # type: ignore[arg-type]
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host, port = server.server_address[:2]

        connection = http.client.HTTPConnection(host, port, timeout=3)
        connection.request(
            "GET",
            "/v1/scenes/scene_1234567890/camera/stream.mjpg",
            headers={
                "Authorization": f"Bearer {token}",
                "X-Scene-Lease": "lease-secret",
            },
        )
        response = connection.getresponse()
        body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.getheader("Content-Type"),
            "multipart/x-mixed-replace; boundary=carla-frame",
        )
        self.assertEqual(body.count(b"--carla-frame\r\n"), 2)
        self.assertIn(b"X-CARLA-Sequence: 0\r\n", body)
        self.assertIn(b'X-Camera-Transform: {"location":{"x":1.0}}\r\n', body)
        self.assertIn(b"\xff\xd80\xff\xd9", body)
        self.assertIn(b"\xff\xd81\xff\xd9", body)
        self.assertEqual(worker.after_sequences, [-1, 0, 1])
        connection.close()

    def test_non_loopback_server_requires_explicit_flag_and_strong_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow-lan"):
            create_server(
                bind="0.0.0.0",
                port=0,
                token="x" * 40,
                worker=self.worker,
            )
        with self.assertRaisesRegex(ValueError, "at least 32"):
            create_server(
                bind="0.0.0.0",
                port=0,
                token="x" * 20,
                worker=self.worker,
                allow_lan=True,
            )


if __name__ == "__main__":
    unittest.main()
