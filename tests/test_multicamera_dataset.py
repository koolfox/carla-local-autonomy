from __future__ import annotations

import copy
import json
from dataclasses import make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from carla_vision.bridge import CarlaImageFrame
from carla_vision.dataset.camera_rig import calibration, resolve_camera_rig, validate_bundle
from carla_vision.dataset.camera_views import verify_camera_views
from carla_vision.dataset.episode_replay import replay_teacher_episode
from carla_vision.dataset.sync import SynchronizedFramePair
from carla_vision.dataset.verified import load_verified_dataset
from carla_vision.dataset.writer import DatasetWriter
from carla_vision.native.camera_rig import TeacherCameraRig, read_frame_bundle
from carla_vision.native.synchronization import NativeSensorQueue, SensorFrameError
from carla_vision.native.teacher_verify import verify_teacher_dataset
from carla_vision.native.worker import EpisodeActors
from carla_vision.operator.artifacts import ArtifactStore
from carla_vision.scenarios.contracts import CameraRecipe, TransformRecipe


def camera() -> CameraRecipe:
    return CameraRecipe(320, 180, 90, 0.05, 2.2, False, TransformRecipe(1.5, 0, 2, 0, 0, 0))


def frame(number: int, value: int = 100) -> CarlaImageFrame:
    pixels = np.full((180, 320, 4), value, np.uint8)
    pixels[:, :, 3] = 255
    return CarlaImageFrame(
        sequence=number,
        sensor_type=0,
        frame=number,
        timestamp=number * 0.05,
        transform=(1.5, 0, 2, 0, 0, 0),
        width=320,
        height=180,
        fov=90,
        bgra=pixels.tobytes(),
        received_monotonic=number * 0.05,
    )


def make_dataset(root: Path, *, multi: bool = True) -> Path:
    recipes = resolve_camera_rig(camera(), preset="front-three")
    specs = {name: calibration(recipe, np.eye(4)) for name, recipe in recipes.items()}
    with DatasetWriter(
        root,
        dataset_id="teacher",
        carla_endpoint={"host": "fixture", "port": 2000},
        carla_version="0.9.16",
        carla_map="fixture",
        repository_root=root,
    ) as writer:
        for number in (100, 102, 104):
            rgb = frame(number, number)
            views = {
                name: (rgb if name == "front" else frame(number, 30 + index * 50))
                for index, name in enumerate(recipes)
            }
            teacher = replace(rgb, sensor_type=1, bgra=bytes(180 * 320 * 4))
            writer.add_pair(
                SynchronizedFramePair(rgb, teacher, 0, 0),
                split="train",
                scenario_id="scenario",
                episode_id="episode",
                rgb_views=views if multi else None,
                camera_calibrations=specs if multi else None,
                context={
                    "control_mode": "behavior_agent_teacher",
                    "route": {
                        "route_id": "route",
                        "destination_spawn_index": 1,
                        "destination_world_transform": {
                            "x": 10,
                            "y": 0,
                            "z": 0,
                            "pitch": 0,
                            "yaw": 0,
                            "roll": 0,
                        },
                    },
                    "privileged_teacher_control": {
                        "source": "BehaviorAgent.run_step",
                        "carla_frame": number,
                        "throttle": 0.2,
                        "steer": 0.0,
                        "brake": 0.0,
                    },
                    "privileged_evaluation": {"velocity_mps": {"speed": 2.0}},
                },
            )
        writer.set_release_metadata(
            {"control_mode": "behavior_agent_teacher", "episode_ids": ["episode"]}
        )
    return root / "teacher"


def test_rig_preserves_primary_and_inherits_sampling() -> None:
    primary = camera()
    assert resolve_camera_rig(primary) == {"front": primary}
    rig = resolve_camera_rig(primary, preset="front-three")
    assert rig["front"] is primary
    assert rig["front_left"].mount.yaw == -60
    assert rig["front_right"].mount.yaw == 60
    assert all(item.sensor_tick_seconds == 0.05 for item in rig.values())
    config = {
        "schema_version": "1.0",
        "additional_views": [
            {"id": "rear", "mount": replace(primary.mount, yaw=180).as_dict(), "fov_degrees": 100}
        ],
    }
    assert resolve_camera_rig(primary, config=config)["rear"].fov_degrees == 100
    for invalid_id in ("front", "front_teacher", "../left", "left/right"):
        config["additional_views"][0]["id"] = invalid_id
        with pytest.raises(ValueError, match="IDs"):
            resolve_camera_rig(primary, config=config)


def test_calibration_projection_and_mismatched_frames() -> None:
    spec = calibration(camera(), np.eye(4))
    # A point ten metres ahead in CARLA camera axes projects to image centre.
    optical = np.array(spec["optical_from_camera"]) @ np.array([10, 0, 0])
    uvw = np.array(spec["intrinsics"]) @ optical
    assert uvw[:2] / uvw[2] == pytest.approx([160, 90])
    rgb = frame(100)
    validate_bundle(rgb, {"front": rgb, "left": rgb}, {"front": spec, "left": spec})
    with pytest.raises(ValueError, match="exact CARLA frame"):
        validate_bundle(rgb, {"front": rgb, "left": frame(101)}, {"front": spec, "left": spec})
    with pytest.raises(ValueError, match="timestamps"):
        validate_bundle(
            rgb, {"front": rgb, "left": replace(rgb, timestamp=6)}, {"front": spec, "left": spec}
        )
    with pytest.raises(ValueError, match="sensor period"):
        validate_bundle(
            rgb,
            {"front": rgb, "left": rgb},
            {"front": spec, "left": {**spec, "sensor_tick_seconds": 0.1}},
        )


def test_bundle_discards_old_callbacks_and_reports_missing_view() -> None:
    queues = {name: NativeSensorQueue(name, max_frames=2) for name in ("front", "left")}
    for item in queues.values():
        for number in (98, 99, 100):
            item.callback(SimpleNamespace(frame=number))
    bundle = read_frame_bundle(queues, 100, 0.1)
    assert all(item.image.frame == 100 for item in bundle.values())
    assert queues["front"].discarded == 2
    queues["front"].callback(SimpleNamespace(frame=101))
    queues["left"].callback(SimpleNamespace(frame=102))
    with pytest.raises(SensorFrameError, match="missing left"):
        read_frame_bundle(queues, 101, 0.1)


def test_bundle_uses_one_deadline_not_one_timeout_per_camera() -> None:
    queue = SimpleNamespace(get_exact=lambda *_: SimpleNamespace(image=SimpleNamespace(frame=1)))
    with patch("carla_vision.native.camera_rig.time.monotonic", side_effect=[0, 0.01, 0.2]):
        with pytest.raises(SensorFrameError, match="missing left"):
            read_frame_bundle({"front": queue, "left": queue}, 1, 0.1)


@pytest.mark.parametrize("lookup_failure", [False, True])
def test_spawn_records_partial_successes_for_cleanup(lookup_failure: bool) -> None:
    recipes = resolve_camera_rig(camera(), preset="front-three")
    rig = TeacherCameraRig(recipes)
    destroyed_ids = []
    sensors = []
    attributes = []
    blueprint = SimpleNamespace(
        has_attribute=lambda _: True,
        set_attribute=lambda name, value: attributes.append((name, value)),
    )
    responses = [
        SimpleNamespace(error="failed", actor_id=0)
        if index == 2
        else SimpleNamespace(error=None, actor_id=10 + index)
        for index in range(4)
    ]
    spawned = {
        item.actor_id: SimpleNamespace(id=item.actor_id, stop=lambda: None)
        for item in responses
        if not item.error
    }
    session = SimpleNamespace(
        carla=SimpleNamespace(command=SimpleNamespace(SpawnActor=lambda *args: args)),
        world=SimpleNamespace(get_blueprint_library=lambda: None, get_actor=spawned.get),
        client=SimpleNamespace(apply_batch_sync=lambda commands, tick: responses),
        _camera_blueprint=lambda *args, **kwargs: blueprint,
    )
    # Real dataclasses are used for recipe replacement; only CARLA is substituted.
    # Spawning only reads/replaces the episode's recipe.camera field.
    Recipe = make_dataclass("Recipe", [("camera", CameraRecipe)])
    Episode = make_dataclass("Episode", [("recipe", Recipe)])
    actors = EpisodeActors(ego_id=1)
    if lookup_failure:

        def unavailable_actor(_actor_id):
            raise RuntimeError("actor lookup timed out")

        session.world.get_actor = unavailable_actor
    with patch(
        "carla_vision.native.camera_rig._carla_transform",
        return_value=SimpleNamespace(get_matrix=lambda: np.eye(4)),
    ):
        with pytest.raises(RuntimeError, match="timed out" if lookup_failure else "front_left"):
            rig.spawn(session, Episode(Recipe(camera())), actors, sensors)
    destroyed_ids.extend(actors.destruction_order())
    assert set(destroyed_ids) == {1, 10, 11, 13}
    assert len(sensors) == (0 if lookup_failure else 3)
    assert ("lens_k", "0.0") in attributes


@pytest.mark.parametrize("multi", [False, True])
def test_dataset_round_trip_and_existing_front_consumer(tmp_path: Path, multi: bool) -> None:
    root = make_dataset(tmp_path, multi=multi)
    report = verify_teacher_dataset(root)
    assert report["status"] == "passed", report["errors"]
    assert len(report["rgb_camera_ids"]) == (3 if multi else 1)
    verified = load_verified_dataset(root)
    assert verified is not None
    metadata = json.loads((root / "metadata/00000001.json").read_text())
    assert ("rgb_views" in metadata) is multi
    assert (root / "images/train/00000001.png").is_file()
    if multi:
        assert metadata["rgb_views"]["front_left"]["image"]["path"].startswith(
            "cameras/front_left/"
        )


def test_verifier_catches_bundle_tampering_even_with_valid_front(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    dataset = json.loads((root / "dataset.json").read_text())
    sample = dataset["samples"][0]
    metadata = json.loads((root / sample["metadata"]["path"]).read_text())
    checksums = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (root / "checksums.sha256").read_text().splitlines()
    }
    for name, value in (("carla_frame", 999), ("timestamp_seconds", 9.0)):
        bad = copy.deepcopy(metadata)
        bad["rgb_views"]["front_left"][name] = value
        assert verify_camera_views(root, sample, bad, dataset["rgb_camera_ids"], checksums)
    bad = copy.deepcopy(metadata)
    bad["rgb_views"]["front_left"]["calibration"]["camera_to_ego"][0][0] = 3
    assert verify_camera_views(root, sample, bad, dataset["rgb_camera_ids"], checksums)
    (root / "cameras/front_left/train/00000001.png").unlink()
    report = verify_teacher_dataset(root)
    assert report["status"] == "failed"
    assert any("front_left" in message for message in report["errors"])


@pytest.mark.parametrize("multi", [False, True])
def test_replay_produces_playable_indexed_artifact(tmp_path: Path, multi: bool) -> None:
    root = make_dataset(tmp_path / "datasets", multi=multi)
    result = replay_teacher_episode(
        root, episode_id=None, runs_root=tmp_path / "runs", run_id="replay", tile_width=320
    )
    video = cv2.VideoCapture(str(result / "camera-rig.mp4"))
    try:
        assert video.isOpened()
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert video.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        assert video.read()[0]
    finally:
        video.release()
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert "teacher_multicamera_video" in {item["role"] for item in manifest["artifacts"]}
    # The existing recording browser understands the producer's manifest;
    # no new HTTP route, CARLA connection or player implementation is needed.
    store = ArtifactStore(tmp_path)
    inspected = store.inspect_research_object("runs/replay")
    recording = next(item for item in inspected["artifacts"] if item["path"] == "camera-rig.mp4")
    assert recording["preview_kind"] == "video" and recording["available"]
    assert store.artifact_path("runs/replay", "camera-rig.mp4") == (
        result / "camera-rig.mp4",
        False,
    )
    index = json.loads((result / "frames.json").read_text())
    assert index["frames"][0]["carla_frame"] == 100
    assert len(index["frames"][0]["rgb_views"]) == (3 if multi else 1)
