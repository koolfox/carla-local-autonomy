from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_vision.operator.configuration import (
    EXPERIMENT_PRESETS,
    build_configuration_contract,
    build_configuration_evidence,
    build_garage_preview_request,
    build_legacy_drive_request,
    build_situation_request,
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
    assert defaults["route"] == {
        "mode": "free",
        "startSpawnIndex": None,
        "destinationSpawnIndex": None,
    }
    assert defaults["camera"]["resolution"] == "1280x720"
    assert defaults["camera"]["fps"] == 30.0
    assert defaults["perception"]["detector"] == "rtdetr"
    assert defaults["recording"] == {"video": True}
    assert defaults["policy"]["acknowledgeAutonomy"] is False
    assert defaults["policy"]["acknowledgeTrustedCode"] is False
    assert defaults["policy"]["modelId"] == ""


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
    assert "model_package_id" not in request
    assert "model_trusted_code_acknowledged" not in request


def test_manual_and_autopilot_requests_do_not_leak_registered_model_fields() -> None:
    for mode in ("manual", "autopilot"):
        request = build_legacy_drive_request(
            _session(control__mode=mode),
            carla_host="192.168.1.108",
            carla_port=2000,
            worker_connected=True,
            capabilities={"autopilot": True},
        )

        assert "model_package_id" not in request
        assert "model_trusted_code_acknowledged" not in request


def test_registered_model_mode_uses_package_identity_not_raw_checkpoint() -> None:
    request = build_legacy_drive_request(
        _session(
            control__mode="model",
            policy__acknowledgeAutonomy=True,
            policy__acknowledgeTrustedCode=True,
            policy__modelId="road-policy",
            policy__device="cuda",
        ),
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=True,
        capabilities={},
    )

    assert request["control_mode"] == "model"
    assert request["model_package_id"] == "road-policy"
    assert request["model_trusted_code_acknowledged"] is True
    assert request["policy_checkpoint"] == ""
    assert request["policy_device"] == "cuda"


def test_registered_model_mode_requires_model_id() -> None:
    with pytest.raises(ValueError, match="modelId"):
        build_legacy_drive_request(
            _session(control__mode="model", policy__acknowledgeAutonomy=True),
            carla_host="192.168.1.108",
            carla_port=2000,
            worker_connected=True,
            capabilities={},
        )


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


def test_garage_preview_maps_every_shared_scene_and_camera_value() -> None:
    request = build_garage_preview_request(
        _session(
            scene__mapName="Town03",
            scene__weatherPreset="fog-night",
            scene__propPreset="accident",
            scene__trafficCount=80,
            scene__walkerCount=55,
            scene__pedestrianCrossingFactor=0.85,
            scene__speedDifferencePercent=-20.0,
            scene__followingDistanceMetres=7.5,
            vehicle__color="20,40,60",
            camera__resolution="1920x1080",
            camera__fps=60.0,
            camera__fov=104.0,
            camera__spectatorFollow=True,
            route__mode="random_destination",
        )
    )

    assert request == {
        "map_name": "Town03",
        "weather_preset": "fog-night",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "20,40,60",
        "seed": 7,
        "traffic_count": 80,
        "walker_count": 55,
        "prop_preset": "accident",
        "route_mode": "random_destination",
        "start_spawn_index": None,
        "destination_spawn_index": None,
        "pedestrian_crossing_factor": 0.85,
        "speed_difference_percent": -20.0,
        "following_distance_metres": 7.5,
        "spectator_mirror": True,
        "profile": "detail",
        "fov": 104.0,
    }


def test_full_carla_map_paths_are_normalized_across_session_adapters() -> None:
    session = _session(
        scene__mapName="/Game/Carla/Maps/Town03",
        scene__weatherPreset="clear-day",
    )

    preview = build_garage_preview_request(session)
    drive = build_legacy_drive_request(
        session,
        carla_host="192.168.1.108",
        carla_port=2000,
        worker_connected=True,
        capabilities={"autopilot": True},
    )
    situation = build_situation_request(
        session,
        {
            "situationId": "normalized-map",
            "egoSpawnIndex": 0,
            "durationSeconds": 30,
            "captureFps": 5,
            "repetitions": 1,
        },
        current_map=None,
    )

    assert preview["map_name"] == "Town03"
    assert drive["map_name"] == "Town03"
    assert situation["map_name"] == "Town03"


def test_situation_uses_shared_session_and_only_recipe_specific_fields() -> None:
    request = build_situation_request(
        _session(
            scene__mapName="current",
            scene__weatherPreset="wet-day",
            scene__propPreset="construction",
            scene__trafficCount=250,
            scene__walkerCount=250,
            scene__pedestrianCrossingFactor=1.0,
            scene__speedDifferencePercent=-100.0,
            scene__followingDistanceMetres=20.0,
            camera__resolution="1920x1080",
            camera__fov=110.0,
        ),
        {
            "situationId": "shared-scene",
            "egoSpawnIndex": 14,
            "durationSeconds": 120,
            "captureFps": 10,
            "repetitions": 3,
        },
        current_map="/Game/Carla/Maps/Town10HD_Opt",
    )

    assert request["map_name"] == "Town10HD_Opt"
    assert request["vehicle_count"] == 250
    assert request["walker_count"] == 250
    assert request["pedestrian_crossing_factor"] == 1.0
    assert request["speed_difference_percent"] == -100.0
    assert request["following_distance_metres"] == 20.0
    assert request["ego_blueprint"] == "vehicle.tesla.model3"
    assert (request["camera_width"], request["camera_height"]) == (1920, 1080)
    assert request["camera_fov"] == 110.0
    assert request["capture_fps"] == 10


def test_situation_requires_named_weather_for_reproducibility() -> None:
    with pytest.raises(ValueError, match="explicit weather"):
        build_situation_request(
            _session(scene__weatherPreset="keep"),
            {
                "situationId": "shared-scene",
                "egoSpawnIndex": 0,
                "durationSeconds": 30,
                "captureFps": 5,
                "repetitions": 1,
            },
            current_map="Town10HD_Opt",
        )


def test_configuration_evidence_does_not_alias_mutable_inputs() -> None:
    requested = {"scene": {"trafficCount": 20}}
    evidence = build_configuration_evidence(
        requested=requested,
        resolved={"traffic_count": 20},
        applied={"traffic_count": 18},
    )
    requested["scene"]["trafficCount"] = 99

    assert evidence["requested"]["scene"]["trafficCount"] == 20
    assert evidence["resolved"]["traffic_count"] == 20
    assert evidence["applied"]["traffic_count"] == 18
