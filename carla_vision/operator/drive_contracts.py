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
_ROUTE_MODES = frozenset({"free", "random_destination", "selected_destination"})
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
        "start_spawn_index",
        "destination_spawn_index",
        "initial_control_mode",
        "pedestrian_crossing_factor",
        "speed_difference_percent",
        "following_distance_metres",
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
    start_spawn_index: int | None = None
    destination_spawn_index: int | None = None
    initial_control_mode: str = "manual"
    pedestrian_crossing_factor: float = 0.2
    speed_difference_percent: float = 12.0
    following_distance_metres: float = 2.0
    experiment_preset: str = "free_drive"
    max_throttle: float = 0.55
    voxel_enabled: bool = False
    road_enabled: bool = False
    road_backend: str = "segformer"
    road_checkpoint: Path | None = None
    road_device: str = "cpu"

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
            "voxel_enabled",
            "road_enabled", "road_backend", "road_checkpoint", "road_device",
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
        required = allowed - {"color", "experiment_preset", "voxel_enabled", "road_enabled",
                              "road_backend", "road_checkpoint", "road_device"} - _WORKER_FIELDS
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
        supported_detectors = {
            "rtdetr",
            "yolo",
            "m9-hierarchical",
            "m9-hierarchical-rtdetr",
        }
        if detector not in supported_detectors:
            raise ValueError(
                "detector must be one of: "
                + ", ".join(sorted(supported_detectors))
            )

        weights: Path | None = None
        weights_value = str(raw["weights"]).strip()
        if detector_enabled and (weights_value or detector != "ssdlite"):
            if not weights_value:
                raise ValueError("weights are required when detector_enabled is true")
            candidate = workspace / weights_value
            # Preserve saved selections from before checkpoints moved into models/.
            if not candidate.exists() and Path(weights_value).name == weights_value:
                candidate = workspace / "models" / weights_value
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(workspace)
            suffixes = {".pt", ".pth"} if detector == "ssdlite" else {".pt", ".onnx"}
            if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
                raise ValueError(f"weights must be a workspace-contained {'/'.join(sorted(suffixes))} file")
            weights = candidate

        if detector_enabled and detector.startswith("m9-hierarchical"):
            if weights is None or weights.suffix.lower() != ".pt":
                raise ValueError("M9 requires its certified .pt checkpoint")
            if raw["image_size"] != 800:
                raise ValueError("M9 requires image_size=800; select M9 Hierarchical RT-DETR in Vision")
        elif detector_enabled and weights is not None and weights.name == (
            "hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt"
        ):
            raise ValueError("Select M9 Hierarchical RT-DETR in Vision for this checkpoint")

        road_enabled = _boolean(raw.get("road_enabled", False), "road_enabled")
        road_backend = str(raw.get("road_backend", "segformer")).strip().lower()
        if road_backend not in {"segformer", "yolop", "yolopv2"}:
            raise ValueError("road_backend must be segformer, yolop or yolopv2")
        road_device = str(raw.get("road_device", "cpu")).strip().lower()
        allowed_devices = {"cpu", "cuda"} if road_backend in {"yolop", "yolopv2"} else {"cpu", "cuda", "mps"}
        if road_device not in allowed_devices:
            raise ValueError(f"{road_backend} road_device must be {'/'.join(sorted(allowed_devices))}")
        road_checkpoint = None
        if road_enabled and str(raw.get("road_checkpoint", "")).strip():
            road_checkpoint = (workspace / str(raw["road_checkpoint"]).strip()).resolve(strict=True)
            road_checkpoint.relative_to(workspace)
            if road_backend == "yolopv2" and (not road_checkpoint.is_file() or road_checkpoint.suffix != ".pt"):
                raise ValueError("YOLOPv2 requires the official checksum-verified TorchScript .pt file")
            if road_backend == "yolop" and (not road_checkpoint.is_file() or road_checkpoint.suffix != ".onnx"):
                raise ValueError("YOLOP road_checkpoint must be a workspace-contained ONNX file")
            if road_backend == "segformer" and not road_checkpoint.is_dir():
                raise ValueError("SegFormer road_checkpoint must be a workspace-contained model directory")

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
            raise ValueError("route_mode must be free, random_destination, or selected_destination")
        start_raw = raw.get("start_spawn_index")
        destination_raw = raw.get("destination_spawn_index")
        start_spawn_index = (
            None
            if start_raw is None
            else _integer(start_raw, "start_spawn_index", 0, 1_000_000)
        )
        destination_spawn_index = (
            None
            if destination_raw is None
            else _integer(destination_raw, "destination_spawn_index", 0, 1_000_000)
        )
        if route_mode == "selected_destination" and destination_spawn_index is None:
            raise ValueError("selected_destination requires destination_spawn_index")
        if route_mode != "selected_destination" and destination_spawn_index is not None:
            raise ValueError("destination_spawn_index requires route_mode=selected_destination")
        if start_spawn_index is not None and start_spawn_index == destination_spawn_index:
            raise ValueError("start_spawn_index and destination_spawn_index must differ")
        initial_control_mode = str(raw.get("initial_control_mode", "manual")).strip()
        if initial_control_mode not in _CONTROL_MODES:
            raise ValueError("initial_control_mode must be manual or autopilot")
        pedestrian_crossing_factor = _number(
            raw.get("pedestrian_crossing_factor", 0.2),
            "pedestrian_crossing_factor",
            0.0,
            1.0,
        )
        speed_difference_percent = _number(
            raw.get("speed_difference_percent", 12.0),
            "speed_difference_percent",
            -100.0,
            100.0,
        )
        following_distance_metres = _number(
            raw.get("following_distance_metres", 2.0),
            "following_distance_metres",
            0.1,
            20.0,
        )
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
            if start_spawn_index is not None:
                unsupported.append("start_spawn_index")
            if destination_spawn_index is not None:
                unsupported.append("destination_spawn_index")
            if initial_control_mode != "manual":
                unsupported.append("initial_control_mode")
            if pedestrian_crossing_factor != 0.2:
                unsupported.append("pedestrian_crossing_factor")
            if speed_difference_percent != 12.0:
                unsupported.append("speed_difference_percent")
            if following_distance_metres != 2.0:
                unsupported.append("following_distance_metres")
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
            voxel_enabled=_boolean(raw.get("voxel_enabled", False), "voxel_enabled"),
            road_enabled=road_enabled,
            road_backend=road_backend,
            road_checkpoint=road_checkpoint,
            road_device=road_device,
            detector=detector,
            weights=weights,
            device=str(raw["device"]).strip(),
            image_size=_integer(raw["image_size"], "image_size", 64, 4096),
            confidence=_number(raw["confidence"], "confidence", 0.0, 1.0),
            width=width,
            height=height,
            camera_fps=_number(raw["camera_fps"], "camera_fps", 1.0, 60.0),
            camera_fov=_number(raw["camera_fov"], "camera_fov", 30.0, 150.0),
            record_video=_boolean(raw["record_video"], "record_video"),
            spectator_follow=_boolean(raw["spectator_follow"], "spectator_follow"),
            world_worker_enabled=bool(world_worker_configured),
            map_name=map_name,
            traffic_count=traffic_count,
            walker_count=walker_count,
            route_mode=route_mode,
            start_spawn_index=start_spawn_index,
            destination_spawn_index=destination_spawn_index,
            initial_control_mode=initial_control_mode,
            pedestrian_crossing_factor=pedestrian_crossing_factor,
            speed_difference_percent=speed_difference_percent,
            following_distance_metres=following_distance_metres,
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
            "voxel_enabled": self.voxel_enabled,
            "road_enabled": self.road_enabled,
            "road_backend": self.road_backend,
            "road_checkpoint": str(self.road_checkpoint) if self.road_checkpoint else None,
            "road_device": self.road_device,
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
            "start_spawn_index": self.start_spawn_index,
            "destination_spawn_index": self.destination_spawn_index,
            "initial_control_mode": self.initial_control_mode,
            "pedestrian_crossing_factor": self.pedestrian_crossing_factor,
            "speed_difference_percent": self.speed_difference_percent,
            "following_distance_metres": self.following_distance_metres,
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
