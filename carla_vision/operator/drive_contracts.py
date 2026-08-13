"""Strict, small contracts for the interactive browser drive console."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..controller import ControlCommand
from .situations import PROP_PRESETS, WEATHER_PRESETS

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESOLUTION = re.compile(r"^(\d{3,4})x(\d{3,4})$")
_MAP_NAME = re.compile(r"^[A-Za-z0-9_./-]{1,160}$")
_CONTROL_MODES = frozenset({"manual", "autopilot"})
_ROUTE_MODES = frozenset({"free", "random_destination"})
EXPERIMENT_PRESETS = frozenset(
    {
        "free_drive",
        "manual_handling",
        "autopilot_takeover",
        "perception_review",
        "traffic_stress",
        "adverse_weather",
    }
)
_WORKER_FIELDS = frozenset(
    {
        "map_name",
        "traffic_count",
        "walker_count",
        "route_mode",
        "initial_control_mode",
    }
)
_WEATHER_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "wetness",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
    "dust_storm",
)


def _strict_keys(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def weather_payload(preset: str) -> list[float]:
    """Serialize one named preset in CARLA 0.9.16's fixed RPC field order."""

    if preset not in WEATHER_PRESETS:
        raise ValueError(f"unknown weather preset {preset!r}")
    values = WEATHER_PRESETS[preset]
    return [float(values[field]) for field in _WEATHER_FIELDS]


@dataclass(frozen=True)
class DriveStartConfig:
    run_id: str
    host: str
    port: int
    vehicle_blueprint: str
    color: str | None
    seed: int
    weather_preset: str
    prop_preset: str
    detector_enabled: bool
    detector: str
    weights: Path | None
    device: str
    image_size: int
    confidence: float
    width: int
    height: int
    camera_fps: float
    camera_fov: float
    record_video: bool
    spectator_follow: bool
    world_worker_enabled: bool = False
    map_name: str = "current"
    traffic_count: int = 0
    walker_count: int = 0
    route_mode: str = "free"
    initial_control_mode: str = "manual"
    experiment_preset: str = "free_drive"
    max_throttle: float = 0.55

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        workspace: Path,
        expected_host: str,
        expected_port: int,
        world_worker_configured: bool = False,
    ) -> Self:
        allowed = {
            "run_id",
            "host",
            "port",
            "vehicle_blueprint",
            "color",
            "seed",
            "weather_preset",
            "prop_preset",
            "detector_enabled",
            "detector",
            "weights",
            "device",
            "image_size",
            "confidence",
            "resolution",
            "camera_fps",
            "camera_fov",
            "record_video",
            "spectator_follow",
            "experiment_preset",
            *_WORKER_FIELDS,
        }
        _strict_keys(raw, allowed, "drive start request")
        required = allowed - {"color", "experiment_preset"} - _WORKER_FIELDS
        missing = sorted(key for key in required if key not in raw)
        if missing:
            raise ValueError(f"drive start request is missing fields: {', '.join(missing)}")

        run_id = str(raw["run_id"]).strip()
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must use 1-128 letters, digits, '.', '_' or '-'")
        host = str(raw["host"]).strip()
        port = _integer(raw["port"], "port", 1, 65535)
        if host != expected_host or port != expected_port:
            raise ValueError("drive endpoint must match the operator's configured CARLA server")
        blueprint = str(raw["vehicle_blueprint"]).strip()
        if not blueprint.startswith("vehicle.") or len(blueprint) > 160:
            raise ValueError("vehicle_blueprint must be an exact vehicle.* identifier")

        color_value = raw.get("color")
        color = None if color_value in {None, ""} else str(color_value).strip()
        if color is not None and (
            len(color) > 32
            or any(character not in "0123456789," for character in color)
            or len(color.split(",")) != 3
            or any(not item or not 0 <= int(item) <= 255 for item in color.split(","))
        ):
            raise ValueError("color must be an RGB triplet supplied by the vehicle catalog")

        weather = str(raw["weather_preset"]).strip()
        if weather != "keep" and weather not in WEATHER_PRESETS:
            raise ValueError(f"unknown weather preset {weather!r}")
        prop_preset = str(raw["prop_preset"]).strip()
        if prop_preset not in PROP_PRESETS:
            raise ValueError(f"unknown prop preset {prop_preset!r}")
        detector_enabled = _boolean(raw["detector_enabled"], "detector_enabled")
        detector = str(raw["detector"]).strip().lower()
        if detector not in {"rtdetr", "yolo"}:
            raise ValueError("detector must be rtdetr or yolo")

        weights: Path | None = None
        weights_value = str(raw["weights"]).strip()
        if detector_enabled:
            if not weights_value:
                raise ValueError("weights are required when detector_enabled is true")
            candidate = (workspace / weights_value).resolve(strict=True)
            candidate.relative_to(workspace)
            if not candidate.is_file() or candidate.suffix.lower() != ".pt":
                raise ValueError("weights must be a workspace-contained .pt file")
            weights = candidate

        resolution = str(raw["resolution"]).strip().lower()
        match = _RESOLUTION.fullmatch(resolution)
        if match is None:
            raise ValueError("resolution must use WIDTHxHEIGHT")
        width, height = (int(value) for value in match.groups())
        if not 320 <= width <= 1920 or not 180 <= height <= 1080:
            raise ValueError("resolution must be between 320x180 and 1920x1080")

        map_name = str(raw.get("map_name", "current")).strip()
        if map_name != "current" and not _MAP_NAME.fullmatch(map_name):
            raise ValueError("map_name must be 'current' or an exact CARLA map identifier")
        traffic_count = _integer(raw.get("traffic_count", 0), "traffic_count", 0, 250)
        walker_count = _integer(raw.get("walker_count", 0), "walker_count", 0, 250)
        route_mode = str(raw.get("route_mode", "free")).strip()
        if route_mode not in _ROUTE_MODES:
            raise ValueError("route_mode must be free or random_destination")
        initial_control_mode = str(raw.get("initial_control_mode", "manual")).strip()
        if initial_control_mode not in _CONTROL_MODES:
            raise ValueError("initial_control_mode must be manual or autopilot")
        experiment_preset = str(raw.get("experiment_preset", "free_drive")).strip()
        if experiment_preset not in EXPERIMENT_PRESETS:
            raise ValueError(
                "experiment_preset must be one of: " + ", ".join(sorted(EXPERIMENT_PRESETS))
            )
        if not world_worker_configured:
            unsupported = []
            if map_name != "current":
                unsupported.append("map_name")
            if traffic_count:
                unsupported.append("traffic_count")
            if walker_count:
                unsupported.append("walker_count")
            if route_mode != "free":
                unsupported.append("route_mode")
            if initial_control_mode != "manual":
                unsupported.append("initial_control_mode")
            if unsupported:
                raise ValueError(
                    "configured World Worker is required for: " + ", ".join(unsupported)
                )

        return cls(
            run_id=run_id,
            host=host,
            port=port,
            vehicle_blueprint=blueprint,
            color=color,
            seed=_integer(raw["seed"], "seed", 0, 2**63 - 1),
            weather_preset=weather,
            prop_preset=prop_preset,
            detector_enabled=detector_enabled,
            detector=detector,
            weights=weights,
            device=str(raw["device"]).strip(),
            image_size=_integer(raw["image_size"], "image_size", 64, 4096),
            confidence=_number(raw["confidence"], "confidence", 0.0, 1.0),
            width=width,
            height=height,
            camera_fps=_number(raw["camera_fps"], "camera_fps", 1.0, 20.0),
            camera_fov=_number(raw["camera_fov"], "camera_fov", 30.0, 150.0),
            record_video=_boolean(raw["record_video"], "record_video"),
            spectator_follow=_boolean(raw["spectator_follow"], "spectator_follow"),
            world_worker_enabled=bool(world_worker_configured),
            map_name=map_name,
            traffic_count=traffic_count,
            walker_count=walker_count,
            route_mode=route_mode,
            initial_control_mode=initial_control_mode,
            experiment_preset=experiment_preset,
        )

    def manifest_config(self) -> dict[str, Any]:
        return {
            "object_type": "interactive_drive_session",
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "control_mode": (
                f"world_worker_{self.initial_control_mode}"
                if self.world_worker_enabled
                else "browser_manual_with_deadman"
            ),
            "world_owner": "world_worker" if self.world_worker_enabled else "raw_bridge_session",
            "model_output_actuated": False,
            "run_id": self.run_id,
            "host": self.host,
            "port": self.port,
            "vehicle_blueprint": self.vehicle_blueprint,
            "color": self.color,
            "seed": self.seed,
            "weather_preset": self.weather_preset,
            "prop_preset": self.prop_preset,
            "detector_enabled": self.detector_enabled,
            "detector": self.detector,
            "weights": str(self.weights) if self.weights is not None else None,
            "device": self.device,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "resolution": [self.width, self.height],
            "camera_fps": self.camera_fps,
            "camera_fov": self.camera_fov,
            "record_video": self.record_video,
            "spectator_follow": self.spectator_follow,
            "world_worker_enabled": self.world_worker_enabled,
            "map_name": self.map_name,
            "traffic_count": self.traffic_count,
            "walker_count": self.walker_count,
            "route_mode": self.route_mode,
            "initial_control_mode": self.initial_control_mode,
            "experiment_preset": self.experiment_preset,
            "max_throttle": self.max_throttle,
        }


@dataclass(frozen=True)
class DriveInput:
    session_id: str
    sequence: int
    throttle: float
    steer: float
    brake: float
    hand_brake: bool
    reverse: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        allowed = {
            "session_id",
            "sequence",
            "throttle",
            "steer",
            "brake",
            "hand_brake",
            "reverse",
        }
        _strict_keys(raw, allowed, "drive control request")
        missing = sorted(key for key in allowed if key not in raw)
        if missing:
            raise ValueError(f"drive control request is missing fields: {', '.join(missing)}")
        session_id = str(raw["session_id"]).strip()
        if not _RUN_ID.fullmatch(session_id):
            raise ValueError("session_id is invalid")
        return cls(
            session_id=session_id,
            sequence=_integer(raw["sequence"], "sequence", 0, 2**63 - 1),
            throttle=_number(raw["throttle"], "throttle", 0.0, 1.0),
            steer=_number(raw["steer"], "steer", -1.0, 1.0),
            brake=_number(raw["brake"], "brake", 0.0, 1.0),
            hand_brake=_boolean(raw["hand_brake"], "hand_brake"),
            reverse=_boolean(raw["reverse"], "reverse"),
        )

    def command(self, *, max_throttle: float) -> ControlCommand:
        throttle = min(float(max_throttle), self.throttle)
        brake = self.brake
        if brake > 0.01 or self.hand_brake:
            throttle = 0.0
        return ControlCommand(
            throttle=throttle,
            steer=self.steer,
            brake=brake,
            hand_brake=self.hand_brake,
            reverse=self.reverse,
        )


__all__ = ["DriveInput", "DriveStartConfig", "weather_payload"]
