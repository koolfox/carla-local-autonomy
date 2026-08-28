from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_vision.operator.configuration import (
    EXPERIMENT_PRESETS,
    build_configuration_contract,
    build_legacy_drive_request,
    session_defaults,
)
from carla_vision.operator.drive_contracts import EXPERIMENT_PRESETS as DRIVE_EXPERIMENT_PRESETS
from carla_vision.operator.situations import WEATHER_PRESETS


def _session(**overrides: object) -> dict[str, object]:
    payload = session_defaults(detector_enabled=False)
    payload["identity"]["runId"] = "unified-test"
    payload["vehicle"]["blueprint"] = "vehicle.tesla.model3"
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        payload[section][field] = value
    return payload


def test_configuration_contract_is_secret_free_and_matches_runtime_feature_gate() -> None:
    secret = "never-expose-this-worker-token"
    worker = SimpleNamespace(base_url="http://192.168.1.108:8766", _bearer_token=secret)
    application = SimpleNamespace(
        workspace=Path("/tmp/carla-workspace"),
        carla_host="192.168.1.108",
        carla_port=2000,
        world_worker=worker,
        experimental_enabled=True,
    )

    payload = build_configuration_contract(application, detector_enabled=False)

    assert payload["schema_version"] == "1.0"
    assert payload["system"]["carlaHost"] == "192.168.1.108"
    assert payload["system"]["carlaPort"] == 2000
    assert payload["system"]["worldWorker"] == {
        "configured": True,
        "url": "http://192.168.1.108:8766",
    }
    assert payload["system"]["experimentalEnabled"] is True
    assert payload["sessionDefaults"]["perception"]["enabled"] is False
    assert secret not in json.dumps(payload, sort_keys=True)


def test_configuration_presets_cover_exact_drive_contract_ids() -> None:
    ids = {str(preset["id"]) for preset in EXPERIMENT_PRESETS}
    assert ids == DRIVE_EXPERIMENT_PRESETS


def test_adverse_weather_preset_references_a_real_weather_preset() -> None:
    adverse = next(preset for preset in EXPERIMENT_PRESETS if preset["id"] == "adverse_weather")
    weather = adverse["patch"]["scene"]["weatherPresetIfKeep"]
    assert weather in WEATHER_PRESETS


def test_session_defaults_have_one_scene_drive_perception_recording_shape() -> None:
    defaults = session_defaults(detector_enabled=True)

    assert defaults["scene"] == {
        "mapName": "current",
        "weatherPreset": "keep",
        "propPreset": "none",
        "trafficCount": 0,
        "walkerCount": 0,
        "pedestrianCrossingFactor": 0.2,
        "speedDifferencePercent": 12.0,
        "followingDistanceMetres": 2.0,
    }
    assert defaults["control"] == {"mode": "manual"}
    assert defaults["route"] == {"mode": "free"}
    assert defaults["camera"]["resolution"] == "1280x720"
    assert defaults["camera"]["fps"] == 30.0
    assert defaults["perception"]["detector"] == "rtdetr"
    assert defaults["recording"] == {"video": True}
    assert defaults["policy"]["acknowledgeAutonomy"] is False


def test_connected_worker_is_the_only_population_owner() -> None:
    request = build_legacy_drive_request(
        _session(scene__trafficCount=12, scene__walkerCount=7),
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=True,
        capabilities={"autopilot": True},
    )

    assert request["traffic_count"] == 12
    assert request["walker_count"] == 7
    assert request["traffic_vehicles"] == 0
    assert request["walkers"] == 0


def test_pythonapi_population_is_the_fallback_without_worker() -> None:
    request = build_legacy_drive_request(
        _session(scene__trafficCount=12, scene__walkerCount=7),
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=False,
        capabilities={
            "garage_traffic_population": True,
            "garage_walker_population": True,
        },
    )

    assert request["traffic_count"] == 0
    assert request["walker_count"] == 0
    assert request["traffic_vehicles"] == 12
    assert request["walkers"] == 7


def test_autopilot_is_one_user_mode_but_maps_to_worker_control_ownership() -> None:
    request = build_legacy_drive_request(
        _session(control__mode="autopilot"),
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=True,
        capabilities={"autopilot": True},
    )

    assert request["control_mode"] == "manual"
    assert request["initial_control_mode"] == "autopilot"


def test_autopilot_without_worker_fails_in_the_canonical_adapter() -> None:
    with pytest.raises(RuntimeError, match="connected capable World Worker"):
        build_legacy_drive_request(
            _session(control__mode="autopilot"),
            carla_host="192.168.1.108",
            carla_port=2000,
            worker_connected=False,
            capabilities={"autopilot": False},
        )


def test_experimental_mode_maps_to_garage_policy_without_worker_autopilot() -> None:
    request = build_legacy_drive_request(
        _session(
            control__mode="imitation",
            policy__acknowledgeAutonomy=True,
            policy__checkpoint="models/imitation/best.pt",
            policy__device="mps",
        ),
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=True,
        capabilities={},
    )

    assert request["control_mode"] == "imitation"
    assert request["initial_control_mode"] == "manual"
    assert request["acknowledge_autonomy"] is True
    assert request["policy_checkpoint"] == "models/imitation/best.pt"
    assert request["policy_device"] == "mps"


def test_worker_only_scene_features_fail_cleanly_when_worker_is_offline() -> None:
    with pytest.raises(RuntimeError, match="map selection"):
        build_legacy_drive_request(
            _session(scene__mapName="Town10HD_Opt"),
            carla_host="192.168.1.108",
            carla_port=2000,
            worker_connected=False,
            capabilities={
                "garage_traffic_population": True,
                "garage_walker_population": True,
            },
        )
