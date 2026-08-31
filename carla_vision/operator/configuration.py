"""Canonical product configuration contract shared by the Operator frontends.

This module deliberately contains no HTTP framework code. The current
``http.server`` application and a future ASGI backend can expose the same JSON
shape without changing frontend semantics. It is also the only adapter that maps
the unified product configuration onto the legacy Drive/Garage execution
contracts while those contracts are being migrated.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

CONFIGURATION_SCHEMA_VERSION = "1.0"

EXPERIMENT_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "free_drive",
        "label": "Free Drive",
        "description": "Clean manual baseline",
        "patch": {
            "control": {"mode": "manual"},
            "scene": {"trafficCount": 0, "walkerCount": 0},
        },
    },
    {
        "id": "manual_handling",
        "label": "Manual Handling",
        "description": "Human control without advisory perception",
        "patch": {
            "control": {"mode": "manual"},
            "perception": {"enabled": False},
        },
    },
    {
        "id": "autopilot_takeover",
        "label": "Autopilot Takeover",
        "description": "Traffic Manager baseline with manual takeover",
        "patch": {
            "control": {"mode": "autopilot"},
            "scene": {"trafficCountMinimum": 8, "walkerCountMinimum": 4},
        },
    },
    {
        "id": "perception_review",
        "label": "Perception Review",
        "description": "Advisory detection with retained review media",
        "patch": {
            "control": {"mode": "manual"},
            "perception": {"enabled": True},
            "recording": {"video": True},
        },
    },
    {
        "id": "traffic_stress",
        "label": "Traffic Stress",
        "description": "Dense vehicle and pedestrian population",
        "patch": {
            "scene": {"trafficCountMinimum": 30, "walkerCountMinimum": 20},
        },
    },
    {
        "id": "adverse_weather",
        "label": "Adverse Weather",
        "description": "Heavy-rain evidence capture",
        "patch": {
            "scene": {"weatherPresetIfKeep": "heavy-rain"},
            "recording": {"video": True},
        },
    },
)

_SESSION_SECTIONS = frozenset(
    {
        "identity",
        "scene",
        "vehicle",
        "route",
        "control",
        "camera",
        "perception",
        "recording",
        "experiment",
        "policy",
    }
)
_SECTION_KEYS = {
    "identity": frozenset({"runId", "seed"}),
    "scene": frozenset(
        {
            "mapName",
            "weatherPreset",
            "propPreset",
            "trafficCount",
            "walkerCount",
            "pedestrianCrossingFactor",
            "speedDifferencePercent",
            "followingDistanceMetres",
        }
    ),
    "vehicle": frozenset({"blueprint", "color"}),
    "route": frozenset({"mode"}),
    "control": frozenset({"mode"}),
    "camera": frozenset({"resolution", "fps", "fov", "spectatorFollow"}),
    "perception": frozenset(
        {"enabled", "detector", "weights", "device", "imageSize", "confidence"}
    ),
    "recording": frozenset({"video"}),
    "experiment": frozenset({"preset"}),
    "policy": frozenset(
        {
            "behavior",
            "acknowledgeAutonomy",
            "acknowledgeTrustedCode",
            "modelId",
            "checkpoint",
            "device",
            "voxelReadinessReport",
            "targetSpeedKmh",
            "maxPolicyErrors",
            "maxModelSpeedKmh",
            "maxSteerRate",
        }
    ),
}


def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be an object")
    return raw


def _strict_keys(raw: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")
    missing = sorted(key for key in allowed if key not in raw)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def _validated_session(raw: Any) -> dict[str, Mapping[str, Any]]:
    session = _mapping(raw, "session configuration")
    _strict_keys(session, _SESSION_SECTIONS, "session configuration")
    result: dict[str, Mapping[str, Any]] = {}
    for section, keys in _SECTION_KEYS.items():
        value = _mapping(session[section], f"session.{section}")
        _strict_keys(value, keys, f"session.{section}")
        result[section] = value
    return result


def _camera_resolution(raw: Any) -> tuple[int, int]:
    value = str(raw).strip().lower()
    try:
        width_text, height_text = value.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as error:
        raise ValueError("session.camera.resolution must look like 1280x720") from error
    if not 320 <= width <= 3840 or not 180 <= height <= 2160:
        raise ValueError("session.camera.resolution must be within 320x180 and 3840x2160")
    return width, height


def _garage_camera_profile(resolution: str, fps: Any) -> str:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError("session.camera.fps must be a number")
    width, height = _camera_resolution(resolution)
    rate = float(fps)
    if (width, height) == (640, 384) and rate <= 10.0:
        return "compatibility"
    if (width, height) == (1280, 720) and rate >= 60.0:
        return "high-refresh"
    if (width, height) == (1920, 1080):
        return "detail"
    return "balanced"


def _short_map_name(raw: Any, *, name: str) -> str:
    value = str(raw or "").strip().rstrip("/")
    if not value:
        raise ValueError(f"{name} is empty")
    return value.rsplit("/", 1)[-1]


def _selected_map_name(raw: Any) -> str:
    value = str(raw or "").strip()
    if value == "current":
        return value
    return _short_map_name(value, name="session.scene.mapName")


def _current_map_name(raw: Any) -> str:
    try:
        return _short_map_name(raw, name="current CARLA map")
    except ValueError as error:
        raise RuntimeError(
            "the current CARLA map is unavailable; choose an explicit map before saving"
        ) from error


def build_configuration_evidence(
    *,
    requested: Any,
    resolved: Any,
    applied: Any,
) -> dict[str, Any]:
    """Return one JSON-safe account of configuration resolution and application."""

    return {
        "schema_version": CONFIGURATION_SCHEMA_VERSION,
        "requested": deepcopy(requested),
        "resolved": deepcopy(resolved),
        "applied": deepcopy(applied),
    }


def build_garage_preview_request(raw: Any) -> dict[str, Any]:
    """Map the canonical SessionConfig onto the bounded Garage preview contract."""

    session = _validated_session(raw)
    identity = session["identity"]
    scene = session["scene"]
    vehicle = session["vehicle"]
    camera = session["camera"]
    return {
        "map_name": _selected_map_name(scene["mapName"]),
        "weather_preset": str(scene["weatherPreset"]).strip(),
        "vehicle_blueprint": str(vehicle["blueprint"]).strip(),
        "color": str(vehicle["color"]).strip(),
        "seed": identity["seed"],
        "traffic_count": scene["trafficCount"],
        "walker_count": scene["walkerCount"],
        "prop_preset": str(scene["propPreset"]).strip(),
        "pedestrian_crossing_factor": scene["pedestrianCrossingFactor"],
        "speed_difference_percent": scene["speedDifferencePercent"],
        "following_distance_metres": scene["followingDistanceMetres"],
        "spectator_mirror": camera["spectatorFollow"],
        "profile": _garage_camera_profile(
            str(camera["resolution"]),
            camera["fps"],
        ),
        "fov": camera["fov"],
    }


def build_situation_request(
    raw: Any,
    situation_raw: Any,
    *,
    current_map: str | None,
) -> dict[str, Any]:
    """Map shared SessionConfig plus recipe-only fields to SituationSpec input."""

    session = _validated_session(raw)
    situation = _mapping(situation_raw, "situation settings")
    situation_fields = frozenset(
        {
            "situationId",
            "egoSpawnIndex",
            "durationSeconds",
            "captureFps",
            "repetitions",
        }
    )
    _strict_keys(situation, situation_fields, "situation settings")

    identity = session["identity"]
    scene = session["scene"]
    vehicle = session["vehicle"]
    camera = session["camera"]
    map_name = str(scene["mapName"]).strip()
    if map_name == "current":
        map_name = _current_map_name(current_map)
    else:
        map_name = _short_map_name(map_name, name="session.scene.mapName")
    weather = str(scene["weatherPreset"]).strip()
    if weather == "keep":
        raise ValueError(
            "saving a reproducible situation requires an explicit weather preset"
        )
    width, height = _camera_resolution(camera["resolution"])
    return {
        "situation_id": str(situation["situationId"]).strip(),
        "map_name": map_name,
        "weather_preset": weather,
        "vehicle_count": scene["trafficCount"],
        "walker_count": scene["walkerCount"],
        "pedestrian_crossing_factor": scene["pedestrianCrossingFactor"],
        "speed_difference_percent": scene["speedDifferencePercent"],
        "following_distance_metres": scene["followingDistanceMetres"],
        "prop_preset": str(scene["propPreset"]).strip(),
        "ego_blueprint": str(vehicle["blueprint"]).strip(),
        "ego_spawn_index": situation["egoSpawnIndex"],
        "duration_seconds": situation["durationSeconds"],
        "capture_fps": situation["captureFps"],
        "repetitions": situation["repetitions"],
        "master_seed": identity["seed"],
        "camera_width": width,
        "camera_height": height,
        "camera_fov": camera["fov"],
    }


def session_defaults(*, detector_enabled: bool) -> dict[str, Any]:
    """Return the single editable session configuration defaults."""

    return {
        "identity": {"seed": 7},
        "scene": {
            "mapName": "current",
            "weatherPreset": "keep",
            "propPreset": "none",
            "trafficCount": 0,
            "walkerCount": 0,
            "pedestrianCrossingFactor": 0.2,
            "speedDifferencePercent": 12.0,
            "followingDistanceMetres": 2.0,
        },
        "vehicle": {"blueprint": "", "color": ""},
        "route": {"mode": "free"},
        "control": {"mode": "manual"},
        "camera": {
            "resolution": "1280x720",
            "fps": 30.0,
            "fov": 90.0,
            "spectatorFollow": True,
        },
        "perception": {
            "enabled": bool(detector_enabled),
            "detector": "rtdetr",
            "weights": "",
            "device": "cpu",
            "imageSize": 640,
            "confidence": 0.5,
        },
        "recording": {"video": True},
        "experiment": {"preset": "free_drive"},
        "policy": {
            "behavior": "normal",
            "acknowledgeAutonomy": False,
            "acknowledgeTrustedCode": False,
            "modelId": "",
            "checkpoint": "",
            "device": "cpu",
            "voxelReadinessReport": "",
            "targetSpeedKmh": 35.0,
            "maxPolicyErrors": 3,
            "maxModelSpeedKmh": 45.0,
            "maxSteerRate": 2.5,
        },
    }


def build_configuration_contract(
    application: Any,
    *,
    detector_enabled: bool,
) -> dict[str, Any]:
    """Build a secret-free configuration contract for one Operator process."""

    worker = getattr(application, "world_worker", None)
    worker_url = getattr(worker, "base_url", None) if worker is not None else None
    return {
        "schema_version": CONFIGURATION_SCHEMA_VERSION,
        "system": {
            "workspace": str(application.workspace),
            "carlaHost": str(application.carla_host),
            "carlaPort": int(application.carla_port),
            "localOnly": True,
            "worldWorker": {
                "configured": worker is not None,
                "url": str(worker_url) if worker_url is not None else None,
            },
            "experimentalEnabled": bool(
                getattr(application, "experimental_enabled", False)
            ),
        },
        "sessionDefaults": session_defaults(detector_enabled=detector_enabled),
        "experimentPresets": deepcopy(EXPERIMENT_PRESETS),
    }


def build_legacy_drive_request(
    raw: Any,
    *,
    carla_host: str,
    carla_port: int,
    worker_connected: bool,
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one unified SessionConfig onto the existing Garage start contract.

    Population has exactly one execution owner. A connected World Worker owns
    traffic/walkers through the base Drive contract. Without the Worker, the
    Garage PythonAPI population lane is used only when its explicit capability
    is available. The adapter never populates both lanes.
    """

    session = _validated_session(raw)
    identity = session["identity"]
    scene = session["scene"]
    vehicle = session["vehicle"]
    route = session["route"]
    control = session["control"]
    camera = session["camera"]
    perception = session["perception"]
    recording = session["recording"]
    experiment = session["experiment"]
    policy = session["policy"]

    user_control_mode = str(control["mode"]).strip().lower()
    valid_modes = {"manual", "autopilot", "behavior", "imitation", "voxel", "model"}
    if user_control_mode not in valid_modes:
        raise ValueError(
            "session.control.mode must be manual, autopilot, behavior, imitation, voxel, or model"
        )

    traffic_count = scene["trafficCount"]
    walker_count = scene["walkerCount"]
    if isinstance(traffic_count, bool) or not isinstance(traffic_count, int):
        raise TypeError("session.scene.trafficCount must be an integer")
    if isinstance(walker_count, bool) or not isinstance(walker_count, int):
        raise TypeError("session.scene.walkerCount must be an integer")
    if not 0 <= traffic_count <= 250 or not 0 <= walker_count <= 250:
        raise ValueError("session population counts must be in [0, 250]")

    if user_control_mode == "autopilot":
        if not worker_connected or not bool(capabilities.get("autopilot")):
            raise RuntimeError("Traffic Manager autopilot requires a connected capable World Worker")
        garage_mode = "manual"
        initial_control_mode = "autopilot"
    else:
        garage_mode = user_control_mode
        initial_control_mode = "manual"

    if worker_connected:
        worker_traffic = traffic_count
        worker_walkers = walker_count
        garage_traffic = 0
        garage_walkers = 0
    else:
        worker_traffic = 0
        worker_walkers = 0
        garage_traffic = traffic_count
        garage_walkers = walker_count
        if garage_traffic and not bool(capabilities.get("garage_traffic_population")):
            raise RuntimeError("traffic population requires the World Worker or CARLA PythonAPI")
        if garage_walkers and not bool(capabilities.get("garage_walker_population")):
            raise RuntimeError("pedestrian population requires the World Worker or CARLA PythonAPI")
        if str(scene["mapName"]).strip() != "current":
            raise RuntimeError("map selection requires a connected World Worker")
        if str(route["mode"]).strip() != "free":
            raise RuntimeError("route selection requires a connected World Worker")
        if (
            float(scene["pedestrianCrossingFactor"]) != 0.2
            or float(scene["speedDifferencePercent"]) != 12.0
            or float(scene["followingDistanceMetres"]) != 2.0
        ):
            raise RuntimeError("custom traffic dynamics require a connected World Worker")

    checkpoint = str(policy["checkpoint"]).strip()
    readiness = str(policy["voxelReadinessReport"]).strip()
    model_id = str(policy["modelId"]).strip()
    if garage_mode not in {"imitation", "voxel"}:
        checkpoint = ""
    if garage_mode != "voxel":
        readiness = ""
    if garage_mode != "model":
        model_id = ""
    elif not model_id:
        raise ValueError("session.policy.modelId is required for control.mode=model")

    return {
        "run_id": str(identity["runId"]).strip(),
        "host": str(carla_host),
        "port": int(carla_port),
        "vehicle_blueprint": str(vehicle["blueprint"]).strip(),
        "color": str(vehicle["color"]).strip(),
        "seed": identity["seed"],
        "weather_preset": str(scene["weatherPreset"]).strip(),
        "prop_preset": str(scene["propPreset"]).strip(),
        "detector_enabled": perception["enabled"],
        "detector": str(perception["detector"]).strip(),
        "weights": str(perception["weights"]).strip(),
        "device": str(perception["device"]).strip(),
        "image_size": perception["imageSize"],
        "confidence": perception["confidence"],
        "resolution": str(camera["resolution"]).strip(),
        "camera_fps": camera["fps"],
        "camera_fov": camera["fov"],
        "record_video": recording["video"],
        "spectator_follow": camera["spectatorFollow"],
        "map_name": _selected_map_name(scene["mapName"]),
        "traffic_count": worker_traffic,
        "walker_count": worker_walkers,
        "route_mode": str(route["mode"]).strip(),
        "initial_control_mode": initial_control_mode,
        "pedestrian_crossing_factor": scene["pedestrianCrossingFactor"],
        "speed_difference_percent": scene["speedDifferencePercent"],
        "following_distance_metres": scene["followingDistanceMetres"],
        "experiment_preset": str(experiment["preset"]).strip(),
        "control_mode": garage_mode,
        "behavior": str(policy["behavior"]).strip(),
        "acknowledge_autonomy": policy["acknowledgeAutonomy"],
        "traffic_vehicles": garage_traffic,
        "walkers": garage_walkers,
        "tm_port": 8000,
        "target_speed_kmh": policy["targetSpeedKmh"],
        "policy_checkpoint": checkpoint,
        "policy_device": str(policy["device"]).strip(),
        "voxel_readiness_report": readiness,
        "max_policy_errors": policy["maxPolicyErrors"],
        "max_model_speed_kmh": policy["maxModelSpeedKmh"],
        "max_steer_rate": policy["maxSteerRate"],
        "model_package_id": model_id,
        "model_trusted_code_acknowledged": policy["acknowledgeTrustedCode"],
    }


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "EXPERIMENT_PRESETS",
    "build_configuration_evidence",
    "build_configuration_contract",
    "build_garage_preview_request",
    "build_legacy_drive_request",
    "build_situation_request",
    "session_defaults",
]
