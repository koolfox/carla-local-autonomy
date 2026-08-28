"""Canonical product configuration contract shared by the Operator frontends.

This module deliberately contains no HTTP framework code.  The current
``http.server`` application and a future ASGI backend can expose the same JSON
shape without changing frontend semantics.
"""

from __future__ import annotations

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


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "EXPERIMENT_PRESETS",
    "build_configuration_contract",
    "session_defaults",
]
