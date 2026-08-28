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
    valid_modes = {"manual", "autopilot", "behavior", "imitation", "voxel"}
    if user_control_mode not in valid_modes:
        raise ValueError(
            "session.control.mode must be manual, autopilot, behavior, imitation, or voxel"
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
    if garage_mode not in {"imitation", "voxel"}:
        checkpoint = ""
    if garage_mode != "voxel":
        readiness = ""

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
        "map_name": str(scene["mapName"]).strip(),
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
    }


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "EXPERIMENT_PRESETS",
    "build_configuration_contract",
    "build_legacy_drive_request",
    "session_defaults",
]
