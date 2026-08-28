from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carla_vision.operator.configuration import (
    EXPERIMENT_PRESETS,
    build_configuration_contract,
    session_defaults,
)
from carla_vision.operator.drive_contracts import EXPERIMENT_PRESETS as DRIVE_EXPERIMENT_PRESETS
from carla_vision.operator.situations import WEATHER_PRESETS


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
