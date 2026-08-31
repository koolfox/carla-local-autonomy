"""Small situation builder that emits the existing strict scenario contract."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..scenarios.contracts import ScenarioSuite

SITUATION_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAP_PATTERN = re.compile(r"^Town(?:0[1-9]|10HD|1[1-5])(?:_Opt)?$")

WEATHER_PRESETS: dict[str, dict[str, Any]] = {
    "clear-day": {
        "light": "day",
        "cloudiness": 5.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_azimuth_angle": 35.0,
        "sun_altitude_angle": 55.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "wetness": 0.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.03,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "cloudy-day": {
        "light": "day",
        "cloudiness": 75.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 20.0,
        "sun_azimuth_angle": 120.0,
        "sun_altitude_angle": 40.0,
        "fog_density": 2.0,
        "fog_distance": 80.0,
        "wetness": 0.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.05,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "soft-rain-sunset": {
        "light": "sunset",
        "cloudiness": 70.0,
        "precipitation": 20.0,
        "precipitation_deposits": 25.0,
        "wind_intensity": 35.0,
        "sun_azimuth_angle": 250.0,
        "sun_altitude_angle": 8.0,
        "fog_density": 4.0,
        "fog_distance": 40.0,
        "wetness": 55.0,
        "fog_falloff": 0.5,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.08,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "heavy-rain": {
        "light": "day",
        "cloudiness": 95.0,
        "precipitation": 80.0,
        "precipitation_deposits": 70.0,
        "wind_intensity": 60.0,
        "sun_azimuth_angle": 180.0,
        "sun_altitude_angle": 25.0,
        "fog_density": 12.0,
        "fog_distance": 30.0,
        "wetness": 90.0,
        "fog_falloff": 0.5,
        "scattering_intensity": 1.2,
        "mie_scattering_scale": 0.1,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "fog-night": {
        "light": "night",
        "cloudiness": 65.0,
        "precipitation": 0.0,
        "precipitation_deposits": 10.0,
        "wind_intensity": 8.0,
        "sun_azimuth_angle": 315.0,
        "sun_altitude_angle": -35.0,
        "fog_density": 45.0,
        "fog_distance": 12.0,
        "wetness": 20.0,
        "fog_falloff": 0.7,
        "scattering_intensity": 1.5,
        "mie_scattering_scale": 0.2,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    "wet-day": {
        "light": "day",
        "cloudiness": 45.0,
        "precipitation": 0.0,
        "precipitation_deposits": 45.0,
        "wind_intensity": 15.0,
        "sun_azimuth_angle": 160.0,
        "sun_altitude_angle": 35.0,
        "fog_density": 3.0,
        "fog_distance": 60.0,
        "wetness": 70.0,
        "fog_falloff": 0.2,
        "scattering_intensity": 1.0,
        "mie_scattering_scale": 0.05,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
}


def _transform(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "z": z,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }


PROP_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "none": (),
    "cones": (
        {
            "blueprint_id": "static.prop.trafficcone01",
            "relative_to": "ego_start",
            "transform": _transform(24.0, 1.8),
        },
        {
            "blueprint_id": "static.prop.trafficcone02",
            "relative_to": "ego_start",
            "transform": _transform(30.0, 1.8),
        },
    ),
    "construction": (
        {
            "blueprint_id": "static.prop.warningconstruction",
            "relative_to": "ego_start",
            "transform": _transform(28.0, 3.5),
        },
        {
            "blueprint_id": "static.prop.streetbarrier",
            "relative_to": "ego_start",
            "transform": _transform(34.0, 2.8),
        },
        {
            "blueprint_id": "static.prop.trafficcone01",
            "relative_to": "ego_start",
            "transform": _transform(25.0, 1.8),
        },
    ),
    "accident": (
        {
            "blueprint_id": "static.prop.warningaccident",
            "relative_to": "ego_start",
            "transform": _transform(30.0, 3.0),
        },
        {
            "blueprint_id": "static.prop.dirtdebris01",
            "relative_to": "ego_start",
            "transform": _transform(36.0, 1.5),
        },
    ),
}


def _strict_keys(raw: Mapping[str, Any], required: set[str], name: str) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class SituationSpec:
    """Operator-facing subset of the full scenario recipe."""

    situation_id: str
    map_name: str
    weather_preset: str
    vehicle_count: int
    walker_count: int
    pedestrian_crossing_factor: float
    speed_difference_percent: float
    following_distance_metres: float
    prop_preset: str
    ego_blueprint: str
    ego_spawn_index: int
    duration_seconds: int
    capture_fps: int
    repetitions: int
    master_seed: int
    camera_width: int
    camera_height: int
    camera_fov: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "situation_id",
            "map_name",
            "weather_preset",
            "vehicle_count",
            "walker_count",
            "pedestrian_crossing_factor",
            "speed_difference_percent",
            "following_distance_metres",
            "prop_preset",
            "ego_blueprint",
            "ego_spawn_index",
            "duration_seconds",
            "capture_fps",
            "repetitions",
            "master_seed",
            "camera_width",
            "camera_height",
            "camera_fov",
        }
        _strict_keys(raw, required, "situation")
        situation_id = str(raw["situation_id"]).strip().lower()
        if not _ID_PATTERN.fullmatch(situation_id):
            raise ValueError("situation_id must use lowercase letters, digits, and hyphens")
        map_name = str(raw["map_name"]).strip()
        if not _MAP_PATTERN.fullmatch(map_name):
            raise ValueError("map_name must be a supported CARLA Town map name")
        weather_preset = str(raw["weather_preset"]).strip()
        if weather_preset not in WEATHER_PRESETS:
            raise ValueError(f"unknown weather preset {weather_preset!r}")
        prop_preset = str(raw["prop_preset"]).strip()
        if prop_preset not in PROP_PRESETS:
            raise ValueError(f"unknown prop preset {prop_preset!r}")
        ego_blueprint = str(raw["ego_blueprint"]).strip()
        if not ego_blueprint.startswith("vehicle."):
            raise ValueError("ego_blueprint must start with vehicle.")
        capture_fps = _integer(raw["capture_fps"], "capture_fps", 1, 10)
        if capture_fps not in {1, 2, 5, 10}:
            raise ValueError("capture_fps must be one of 1, 2, 5, or 10")
        return cls(
            situation_id=situation_id,
            map_name=map_name,
            weather_preset=weather_preset,
            vehicle_count=_integer(raw["vehicle_count"], "vehicle_count", 0, 250),
            walker_count=_integer(raw["walker_count"], "walker_count", 0, 250),
            pedestrian_crossing_factor=_number(
                raw["pedestrian_crossing_factor"],
                "pedestrian_crossing_factor",
                0.0,
                1.0,
            ),
            speed_difference_percent=_number(
                raw["speed_difference_percent"],
                "speed_difference_percent",
                -100.0,
                100.0,
            ),
            following_distance_metres=_number(
                raw["following_distance_metres"],
                "following_distance_metres",
                0.1,
                20.0,
            ),
            prop_preset=prop_preset,
            ego_blueprint=ego_blueprint,
            ego_spawn_index=_integer(raw["ego_spawn_index"], "ego_spawn_index", 0, 10000),
            duration_seconds=_integer(raw["duration_seconds"], "duration_seconds", 5, 3600),
            capture_fps=capture_fps,
            repetitions=_integer(raw["repetitions"], "repetitions", 1, 100),
            master_seed=_integer(raw["master_seed"], "master_seed", 0, 2**63 - 1),
            camera_width=_integer(raw["camera_width"], "camera_width", 320, 3840),
            camera_height=_integer(raw["camera_height"], "camera_height", 180, 2160),
            camera_fov=_number(raw["camera_fov"], "camera_fov", 30.0, 150.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in (
                "situation_id",
                "map_name",
                "weather_preset",
                "vehicle_count",
                "walker_count",
                "pedestrian_crossing_factor",
                "speed_difference_percent",
                "following_distance_metres",
                "prop_preset",
                "ego_blueprint",
                "ego_spawn_index",
                "duration_seconds",
                "capture_fps",
                "repetitions",
                "master_seed",
                "camera_width",
                "camera_height",
                "camera_fov",
            )
        }


def build_scenario_suite(spec: SituationSpec) -> ScenarioSuite:
    fixed_delta = 0.05
    sensor_tick = 0.1
    duration_ticks = spec.duration_seconds * 20
    capture_every_ticks = 20 // spec.capture_fps
    weather = {
        "weather_id": f"operator-{spec.weather_preset}",
        **WEATHER_PRESETS[spec.weather_preset],
    }
    raw = {
        "suite_id": f"suite-operator-{spec.situation_id}-v1",
        "schema_version": "1.0",
        "carla_version": "0.9.16",
        "master_seed": spec.master_seed,
        "traffic_manager_port": 8000,
        "recipes": [
            {
                "recipe_id": f"operator-{spec.situation_id}",
                "map_name": spec.map_name,
                "route_region_id": f"spawn-{spec.ego_spawn_index:03d}-autopilot",
                "ego_blueprint": spec.ego_blueprint,
                "ego_spawn_index": spec.ego_spawn_index,
                "fixed_delta_seconds": fixed_delta,
                "repetitions": spec.repetitions,
                "weather": weather,
                "traffic": {
                    "vehicle_count": spec.vehicle_count,
                    "walker_count": spec.walker_count,
                    "pedestrian_crossing_factor": spec.pedestrian_crossing_factor,
                    "vehicle_filter": "vehicle.*",
                    "walker_filter": "walker.pedestrian.*",
                    "vehicle_generation": "All",
                    "walker_generation": "2",
                    "global_speed_difference_percent": spec.speed_difference_percent,
                    "global_distance_to_leading_vehicle": spec.following_distance_metres,
                    "automatic_vehicle_lights": True,
                },
                "camera": {
                    "width": spec.camera_width,
                    "height": spec.camera_height,
                    "fov_degrees": spec.camera_fov,
                    "sensor_tick_seconds": sensor_tick,
                    "gamma": 2.2,
                    "enable_postprocess_effects": True,
                    "mount": {
                        "x": 1.5,
                        "y": 0.0,
                        "z": 1.7,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "roll": 0.0,
                    },
                },
                "capture": {
                    "warmup_ticks": 100,
                    "duration_ticks": duration_ticks,
                    "capture_every_ticks": capture_every_ticks,
                    "minimum_visible_pixels": 16,
                    "minimum_box_width": 2,
                    "minimum_box_height": 2,
                },
                "props": [dict(prop) for prop in PROP_PRESETS[spec.prop_preset]],
            }
        ],
    }
    return ScenarioSuite.from_mapping(raw)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_situation_suite(
    spec: SituationSpec,
    *,
    workspace: str | Path,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    output = root / "operator_configs" / "situations" / f"{spec.situation_id}.json"
    if output.exists():
        raise FileExistsError(
            f"situation {spec.situation_id!r} already exists; choose a new situation ID"
        )
    suite = build_scenario_suite(spec)
    _atomic_json(output, suite.as_dict())
    return output


__all__ = [
    "PROP_PRESETS",
    "SITUATION_SCHEMA_VERSION",
    "WEATHER_PRESETS",
    "SituationSpec",
    "build_scenario_suite",
    "save_situation_suite",
]
