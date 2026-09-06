from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from carla_vision.contracts import Detection
from carla_vision.model_registry import discover_model_packages
from carla_vision.scene_perception import (
    CAPABILITY_DRIVABLE_AREA,
    CAPABILITY_OBJECTS,
    CAPABILITY_TRAFFIC_LIGHTS,
    CAPABILITY_TRAFFIC_LIGHT_STATE,
    DrivableAreaObservation,
    LaneMarkingObservation,
    RoadUserObservation,
    ScenePerception,
    ScenePerceptionObservation,
    ScenePerceptionResult,
    TrafficLightObservation,
)


def _detection(label: str = "vehicle") -> Detection:
    return Detection(
        class_id=1,
        label=label,
        confidence=0.9,
        xyxy=(10.0, 20.0, 30.0, 50.0),
    )


def _observation(image: np.ndarray | None = None) -> ScenePerceptionObservation:
    if image is None:
        image = np.zeros((24, 32, 3), dtype=np.uint8)
    return ScenePerceptionObservation(
        sequence=7,
        carla_frame=101,
        source_timestamp=4.25,
        source_received_monotonic=20.0,
        image_bgr=image,
    )


def test_scene_observation_owns_read_only_frame_copy() -> None:
    source = np.zeros((24, 32, 3), dtype=np.uint8)
    observation = _observation(source)

    source[0, 0, 0] = 255

    assert observation.image_bgr[0, 0, 0] == 0
    assert observation.image_bgr.flags.c_contiguous is True
    assert observation.image_bgr.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        observation.image_bgr[0, 0, 0] = 1


def test_scene_perception_reuses_detection_and_normalizes_road_user_category() -> None:
    road_user = RoadUserObservation(detection=_detection(), category=" Pedestrian ")

    perception = ScenePerception(
        capabilities=(CAPABILITY_OBJECTS,),
        road_users=(road_user,),
    )

    assert perception.road_users[0].detection.label == "vehicle"
    assert perception.road_users[0].category == "pedestrian"


def test_traffic_light_state_is_never_inferred_without_declared_capability() -> None:
    light = TrafficLightObservation(
        detection=_detection("traffic light"),
        state="red",
        state_confidence=0.8,
    )

    with pytest.raises(ValueError, match="traffic_light_state capability"):
        ScenePerception(
            capabilities=(CAPABILITY_TRAFFIC_LIGHTS,),
            traffic_lights=(light,),
        )


def test_traffic_light_detection_without_state_is_valid_for_detection_only_model() -> None:
    light = TrafficLightObservation(detection=_detection("traffic light"))

    perception = ScenePerception(
        capabilities=(CAPABILITY_TRAFFIC_LIGHTS,),
        traffic_lights=(light,),
    )

    assert perception.traffic_lights[0].state is None


def test_traffic_light_state_capability_requires_traffic_light_detection_capability() -> None:
    with pytest.raises(ValueError, match="requires traffic_lights"):
        ScenePerception(capabilities=(CAPABILITY_TRAFFIC_LIGHT_STATE,))


def test_semantics_cannot_be_populated_when_model_did_not_declare_them() -> None:
    road_user = RoadUserObservation(detection=_detection(), category="vehicle")

    with pytest.raises(ValueError, match="objects capability"):
        ScenePerception(
            capabilities=(CAPABILITY_DRIVABLE_AREA,),
            road_users=(road_user,),
        )


def test_lane_and_drivable_geometry_are_validated_in_image_space() -> None:
    lane = LaneMarkingObservation(
        label="lane_boundary",
        confidence=0.7,
        points=((10.0, 30.0), (12.0, 20.0)),
    )
    drivable = DrivableAreaObservation(
        confidence=0.75,
        polygon=((0.0, 23.0), (31.0, 23.0), (16.0, 8.0)),
    )

    assert lane.points[1] == (12.0, 20.0)
    assert drivable.label == "drivable"

    with pytest.raises(ValueError, match="at least two points"):
        LaneMarkingObservation(label="lane", confidence=0.5, points=((1.0, 2.0),))
    with pytest.raises(ValueError, match="at least three"):
        DrivableAreaObservation(
            confidence=0.5,
            polygon=((0.0, 0.0), (1.0, 1.0)),
        )


def test_scene_result_preserves_exact_frame_identity_and_latency() -> None:
    result = ScenePerceptionResult(
        observation=_observation(),
        perception=ScenePerception(capabilities=(CAPABILITY_OBJECTS,)),
        inference_started_monotonic=20.1,
        completed_monotonic=20.4,
        model_name="scene-policy",
    )

    assert result.carla_frame == 101
    assert result.model_inference_seconds == pytest.approx(0.3)
    assert result.end_to_end_seconds == pytest.approx(0.4)
    assert result.age_seconds(20.7) == pytest.approx(0.7)

    with pytest.raises(ValueError, match="follow source frame receipt"):
        ScenePerceptionResult(
            observation=_observation(),
            perception=ScenePerception(capabilities=(CAPABILITY_OBJECTS,)),
            inference_started_monotonic=19.9,
            completed_monotonic=20.1,
        )


def _write_scene_package(
    root: Path,
    *,
    package_id: str = "scene-policy",
    inputs: object | None = None,
    outputs: object | None = None,
    runtime: str = "python_factory",
) -> None:
    artifact = b"scene-perception-checkpoint"
    directory = root / "models" / package_id
    directory.mkdir(parents=True)
    (directory / "weights.pth").write_bytes(artifact)
    payload = {
        "schema_version": "1.0",
        "object_type": "runtime_model_package",
        "id": package_id,
        "name": "Scene Policy",
        "version": "1.0.0",
        "role": "scene_perception",
        "runtime": runtime,
        "artifact": "weights.pth",
        "sha256": hashlib.sha256(artifact).hexdigest(),
        "factory": "research_models.scene:create_model" if runtime == "python_factory" else None,
        "devices": ["cpu"],
        "inputs": inputs
        if inputs is not None
        else {
            "kind": "scene_perception_observation_v1",
            "history": 3,
            "options": {"resize": 640},
        },
        "outputs": outputs
        if outputs is not None
        else {
            "kind": "scene_perception_v1",
            "capabilities": [
                "objects",
                "traffic_lights",
                "traffic_light_state",
                "drivable_area",
            ],
        },
    }
    if payload["factory"] is None:
        payload.pop("factory")
    (directory / "model.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_registry_advertises_normalized_scene_perception_contract(tmp_path: Path) -> None:
    _write_scene_package(tmp_path)

    registry = discover_model_packages(tmp_path)

    assert registry["invalid"] == []
    package = registry["packages"][0]
    assert package["role"] == "scene_perception"
    assert package["runtime"] == "python_factory"
    assert package["requiresTrustedCode"] is True
    assert package["inputs"] == {
        "kind": "scene_perception_observation_v1",
        "history": 3,
        "options": {"resize": 640},
    }
    assert package["outputs"]["kind"] == "scene_perception_v1"
    assert package["outputs"]["capabilities"] == [
        "objects",
        "traffic_lights",
        "traffic_light_state",
        "drivable_area",
    ]


def test_registry_rejects_scene_package_with_unsupported_semantic_claim(tmp_path: Path) -> None:
    _write_scene_package(
        tmp_path,
        outputs={
            "kind": "scene_perception_v1",
            "capabilities": ["objects", "weather_prediction"],
        },
    )

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "unsupported values: weather_prediction" in registry["invalid"][0]["message"]


def test_registry_rejects_scene_package_that_claims_light_state_without_lights(
    tmp_path: Path,
) -> None:
    _write_scene_package(
        tmp_path,
        outputs={
            "kind": "scene_perception_v1",
            "capabilities": ["traffic_light_state"],
        },
    )

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "traffic_light_state capability requires traffic_lights" in registry["invalid"][0][
        "message"
    ]


def test_registry_rejects_scene_package_with_unversioned_input_shape(tmp_path: Path) -> None:
    _write_scene_package(
        tmp_path,
        inputs={"image": "front_rgb", "history": 1},
    )

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "inputs has unknown fields: image" in registry["invalid"][0]["message"]


def test_registry_does_not_advertise_scene_role_on_unimplemented_runtime(tmp_path: Path) -> None:
    _write_scene_package(tmp_path, runtime="torchscript_control_v1")

    registry = discover_model_packages(tmp_path)

    assert registry["packages"] == []
    assert "torchscript_control_v1 runtime requires role=driving_policy" in registry["invalid"][0][
        "message"
    ]
