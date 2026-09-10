"""Exercise the real teacher collection loop with a deterministic fake CARLA world."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carla_vision.dataset.writer import DatasetWriter
from carla_vision.native.behavior_teacher import BehaviorTeacherSession, collect_behavior_teacher
from carla_vision.native.behavior_teacher import parse_args as teacher_args
from carla_vision.native.synchronization import SensorFrameError
from carla_vision.native.teacher_verify import verify_teacher_dataset
from carla_vision.navigation_intent import NavigationCommand, NavigationIntent
from carla_vision.scenarios.contracts import load_scenario_suite
from carla_vision.scenarios.planner import expand_scenario_suite, plan_scenarios
from carla_vision.scenarios.splits import load_split_plan

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/scenarios/thesis_pilot_v1.json"
SPLITS = ROOT / "configs/scenarios/split_plan_thesis_pilot_v1.json"


def test_camera_rig_dry_run_has_no_carla_import(tmp_path: Path, monkeypatch) -> None:
    plan = plan_scenarios(
        suite_path=SUITE,
        split_plan_path=SPLITS,
        runs_root=tmp_path,
        run_id="plan",
        repository_root=ROOT,
    )
    monkeypatch.setattr(
        "carla_vision.native.behavior_teacher._load_carla_module",
        lambda *_: pytest.fail("dry run must not import CARLA"),
    )
    result = collect_behavior_teacher(
        teacher_args(
            [
                "--scenario-plan",
                str(plan["run_dir"]),
                "--dataset-id",
                "dry",
                "--camera-rig",
                "front-three",
                "--max-episodes",
                "1",
                "--dry-run",
            ]
        )
    )
    assert len(result["camera_rigs"]) == 1
    assert set(next(iter(result["camera_rigs"].values()))) == {"front", "front_left", "front_right"}


@pytest.mark.parametrize("missing_view,cancelled", [(False, False), (True, False), (False, True)])
def test_capture_loop_alignment_and_sensor_cleanup(
    tmp_path: Path, monkeypatch, missing_view, cancelled
) -> None:
    episode = expand_scenario_suite(load_scenario_suite(SUITE), load_split_plan(SPLITS))[0]
    primary = replace(episode.recipe.camera, width=320, height=180, sensor_tick_seconds=0.05)
    episode = replace(
        episode,
        recipe=replace(
            episode.recipe,
            camera=primary,
            fixed_delta_seconds=0.05,
            capture=replace(
                episode.recipe.capture, warmup_ticks=0, duration_ticks=5, capture_every_ticks=2
            ),
        ),
    )
    sensor_actors = {}
    removed = []
    stopped = []
    state = {"frame": 0}

    def transform(recipe):
        yaw = np.radians(recipe.yaw)
        matrix = np.eye(4)
        matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        matrix[:3, 3] = [recipe.x, recipe.y, recipe.z]
        return SimpleNamespace(
            location=SimpleNamespace(x=recipe.x, y=recipe.y, z=recipe.z),
            rotation=SimpleNamespace(pitch=recipe.pitch, yaw=recipe.yaw, roll=recipe.roll),
            get_matrix=lambda: matrix,
        )

    def blueprint(_library, sensor_type, camera_episode, *, role_name):
        return SimpleNamespace(
            name=role_name,
            sensor_type=sensor_type,
            recipe=camera_episode.recipe.camera,
            has_attribute=lambda _: True,
            set_attribute=lambda *_: None,
        )

    def apply_batch(commands, tick):
        replies = []
        for command in commands:
            if command[0] == "destroy":
                removed.append(command[1])
                replies.append(SimpleNamespace(error=None))
                continue
            _, bp, mount, _parent = command
            sensor_id = 10 + len(sensor_actors)
            sensor = SimpleNamespace(id=sensor_id, bp=bp, mount=mount, callback=None)
            sensor.listen = lambda callback, sensor=sensor: setattr(sensor, "callback", callback)
            sensor.stop = lambda sensor_id=sensor_id: stopped.append(sensor_id)
            sensor_actors[sensor_id] = sensor
            replies.append(SimpleNamespace(error=None, actor_id=sensor_id))
        return replies

    def tick():
        state["frame"] += 1
        number = state["frame"]
        for sensor in sensor_actors.values():
            if missing_view and sensor.bp.name == "front_left" and number == 2:
                continue
            pixels = np.zeros((180, 320, 4), np.uint8)
            if sensor.bp.sensor_type == "sensor.camera.rgb":
                pixels[:, :, :3] = number * 10 + sensor.id
            sensor.callback(
                SimpleNamespace(
                    frame=number,
                    timestamp=number * 0.05,
                    width=320,
                    height=180,
                    raw_data=pixels.tobytes(),
                    transform=sensor.mount,
                )
            )
        return number

    world = SimpleNamespace(
        get_blueprint_library=lambda: None,
        get_map=lambda: SimpleNamespace(
            name=episode.recipe.map_name, get_spawn_points=lambda: [None]
        ),
        get_actor=sensor_actors.get,
        tick=tick,
        get_snapshot=lambda: SimpleNamespace(frame=state["frame"]),
    )
    ego = SimpleNamespace(get_transform=lambda: transform(primary.mount))
    session = object.__new__(BehaviorTeacherSession)
    session.world = world
    session.carla = SimpleNamespace(
        command=SimpleNamespace(
            SpawnActor=lambda *args: ("spawn", *args),
            DestroyActor=lambda actor_id: ("destroy", actor_id),
        )
    )
    session.client = SimpleNamespace(apply_batch_sync=apply_batch)
    session.client_version = session.server_version = "0.9.16"
    session.sensor_timeout = 0.02
    session.behavior = "normal"
    session.target_speed_kmh = 20
    session.minimum_route_distance_m = 10
    session.camera_rig = "front-three"
    session.camera_rig_config = None
    session.cancel_check = lambda: cancelled and state["frame"] >= 2
    session._configure_world = lambda _: (world, None)
    session._spawn_ego = lambda _episode, actors, *_: (setattr(actors, "ego_id", 1), ego)[1]
    session._spawn_props = lambda *_: []
    session._spawn_traffic = lambda *_: []
    session._spawn_walkers = lambda *_: []
    session._actor_inventory = lambda _: []
    session._camera_blueprint = blueprint
    monkeypatch.setattr(
        "carla_vision.native.camera_rig._carla_transform", lambda _carla, raw: transform(raw)
    )
    monkeypatch.setattr(
        "carla_vision.native.behavior_teacher._privileged_ego_state",
        lambda _: {"velocity_mps": {"speed": 2.0}},
    )
    control = SimpleNamespace(throttle=0.2, steer=0.0, brake=0.0)
    controller = SimpleNamespace(
        apply_before_tick=lambda **_: control,
        history=[],
        sample_route_context=lambda **_: {
            "route_id": "route",
            "destination_spawn_index": 0,
            "destination_world_transform": primary.mount.as_dict(),
        },
        sample_navigation_intent=lambda carla_frame: NavigationIntent(
            source_frame_id=carla_frame,
            command=NavigationCommand.FOLLOW_LANE,
            direction=(1.0, 0.0),
            target_point_m=(10.0, 0.0),
            distance_to_maneuver_m=10.0,
            route_polyline_m=((0.0, 0.0), (10.0, 0.0)),
            source="carla_global_route_planner_via_behavior_agent",
            confidence=1.0,
            privileged=True,
            route_id="route",
        ).as_dict(),
    )
    monkeypatch.setattr(
        "carla_vision.native.behavior_teacher.BehaviorRouteController",
        lambda *_args, **_kwargs: controller,
    )
    writer = DatasetWriter(
        tmp_path,
        dataset_id="capture",
        carla_endpoint={},
        carla_version="0.9.16",
        carla_map="fixture",
        repository_root=ROOT,
    )
    if cancelled:
        with pytest.raises(InterruptedError) as failure, writer:
            session.run_episode(episode, writer)
        assert failure.value.native_cleanup_confirmed
        assert not (writer.dataset_dir / "dataset.json").exists()
    elif missing_view:
        with pytest.raises(SensorFrameError, match="missing front_left"), writer:
            session.run_episode(episode, writer)
        assert list((writer.dataset_dir / "episodes").rglob("capture_failure.json"))
        assert not (writer.dataset_dir / "dataset.json").exists()
        manifest = json.loads((writer.dataset_dir / "manifest.json").read_text())
        assert manifest["status"] == "failed"
    else:
        with writer:
            result = session.run_episode(episode, writer)
            writer.set_release_metadata(
                {"control_mode": "behavior_agent_teacher", "episode_ids": [episode.episode_id]}
            )
        assert result["samples"]["carla_frames"] == [2, 4, 6]
        report = verify_teacher_dataset(writer.dataset_dir)
        assert report["status"] == "passed", report["errors"]
    assert set(stopped) == {10, 11, 12, 13}
    assert set(removed) == {1, 10, 11, 12, 13}
