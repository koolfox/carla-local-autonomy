"""Strict, JSON-serializable scenario recipe contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

SCENARIO_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _validate_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    keys = {str(key) for key in value}
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: Any, name: str, minimum: float, maximum: float) -> float:
    result = _finite(value, name)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _non_negative_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or not 0 <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return result


def _positive_int(value: Any, name: str, maximum: int) -> int:
    result = _non_negative_int(value, name, maximum)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _identifier(value: Any, name: str) -> str:
    result = str(value)
    if not _ID_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must be 1-128 characters using letters, digits, '.', '_' or '-'")
    return result


@dataclass(frozen=True)
class TransformRecipe:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> Self:
        _validate_keys(
            value,
            name=name,
            required={"x", "y", "z", "pitch", "yaw", "roll"},
        )
        return cls(
            x=_finite(value["x"], f"{name}.x"),
            y=_finite(value["y"], f"{name}.y"),
            z=_finite(value["z"], f"{name}.z"),
            pitch=_bounded(value["pitch"], f"{name}.pitch", -360.0, 360.0),
            yaw=_bounded(value["yaw"], f"{name}.yaw", -360.0, 360.0),
            roll=_bounded(value["roll"], f"{name}.roll", -360.0, 360.0),
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class WeatherRecipe:
    weather_id: str
    light: str
    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    fog_distance: float
    wetness: float
    fog_falloff: float
    scattering_intensity: float
    mie_scattering_scale: float
    rayleigh_scattering_scale: float
    dust_storm: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        required = {
            "weather_id",
            "light",
            "cloudiness",
            "precipitation",
            "precipitation_deposits",
            "wind_intensity",
            "sun_azimuth_angle",
            "sun_altitude_angle",
            "fog_density",
            "fog_distance",
            "wetness",
            "fog_falloff",
            "scattering_intensity",
            "mie_scattering_scale",
            "rayleigh_scattering_scale",
            "dust_storm",
        }
        _validate_keys(raw, name="weather", required=required)
        light = str(raw["light"]).strip().lower()
        if light not in {"dawn", "day", "sunset", "night"}:
            raise ValueError("weather.light must be dawn, day, sunset, or night")
        return cls(
            weather_id=_identifier(raw["weather_id"], "weather.weather_id"),
            light=light,
            cloudiness=_bounded(raw["cloudiness"], "weather.cloudiness", 0.0, 100.0),
            precipitation=_bounded(raw["precipitation"], "weather.precipitation", 0.0, 100.0),
            precipitation_deposits=_bounded(
                raw["precipitation_deposits"],
                "weather.precipitation_deposits",
                0.0,
                100.0,
            ),
            wind_intensity=_bounded(raw["wind_intensity"], "weather.wind_intensity", 0.0, 100.0),
            sun_azimuth_angle=_bounded(
                raw["sun_azimuth_angle"], "weather.sun_azimuth_angle", 0.0, 360.0
            ),
            sun_altitude_angle=_bounded(
                raw["sun_altitude_angle"], "weather.sun_altitude_angle", -90.0, 90.0
            ),
            fog_density=_bounded(raw["fog_density"], "weather.fog_density", 0.0, 100.0),
            fog_distance=_bounded(raw["fog_distance"], "weather.fog_distance", 0.0, 100_000.0),
            wetness=_bounded(raw["wetness"], "weather.wetness", 0.0, 100.0),
            fog_falloff=_bounded(raw["fog_falloff"], "weather.fog_falloff", 0.0, 100.0),
            scattering_intensity=_bounded(
                raw["scattering_intensity"],
                "weather.scattering_intensity",
                0.0,
                100.0,
            ),
            mie_scattering_scale=_bounded(
                raw["mie_scattering_scale"],
                "weather.mie_scattering_scale",
                0.0,
                100.0,
            ),
            rayleigh_scattering_scale=_bounded(
                raw["rayleigh_scattering_scale"],
                "weather.rayleigh_scattering_scale",
                0.0,
                100.0,
            ),
            dust_storm=_bounded(raw["dust_storm"], "weather.dust_storm", 0.0, 100.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CameraRecipe:
    width: int
    height: int
    fov_degrees: float
    sensor_tick_seconds: float
    gamma: float
    enable_postprocess_effects: bool
    mount: TransformRecipe

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _validate_keys(
            raw,
            name="camera",
            required={
                "width",
                "height",
                "fov_degrees",
                "sensor_tick_seconds",
                "gamma",
                "enable_postprocess_effects",
                "mount",
            },
        )
        width = _positive_int(raw["width"], "camera.width", 7680)
        height = _positive_int(raw["height"], "camera.height", 4320)
        if width < 320 or height < 180:
            raise ValueError("camera dimensions must be at least 320x180")
        postprocess = raw["enable_postprocess_effects"]
        if not isinstance(postprocess, bool):
            raise TypeError("camera.enable_postprocess_effects must be boolean")
        return cls(
            width=width,
            height=height,
            fov_degrees=_bounded(raw["fov_degrees"], "camera.fov_degrees", 30.0, 150.0),
            sensor_tick_seconds=_bounded(
                raw["sensor_tick_seconds"],
                "camera.sensor_tick_seconds",
                0.001,
                10.0,
            ),
            gamma=_bounded(raw["gamma"], "camera.gamma", 0.1, 5.0),
            enable_postprocess_effects=postprocess,
            mount=TransformRecipe.from_mapping(
                _mapping(raw["mount"], "camera.mount"),
                "camera.mount",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "mount": self.mount.as_dict(),
        }


@dataclass(frozen=True)
class TrafficRecipe:
    vehicle_count: int
    walker_count: int
    pedestrian_crossing_factor: float
    vehicle_filter: str
    walker_filter: str
    vehicle_generation: str
    walker_generation: str
    global_speed_difference_percent: float
    global_distance_to_leading_vehicle: float
    automatic_vehicle_lights: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _validate_keys(
            raw,
            name="traffic",
            required={
                "vehicle_count",
                "walker_count",
                "pedestrian_crossing_factor",
                "vehicle_filter",
                "walker_filter",
                "vehicle_generation",
                "walker_generation",
                "global_speed_difference_percent",
                "global_distance_to_leading_vehicle",
                "automatic_vehicle_lights",
            },
        )
        lights = raw["automatic_vehicle_lights"]
        if not isinstance(lights, bool):
            raise TypeError("traffic.automatic_vehicle_lights must be boolean")
        vehicle_filter = str(raw["vehicle_filter"]).strip()
        walker_filter = str(raw["walker_filter"]).strip()
        if not vehicle_filter or not walker_filter:
            raise ValueError("traffic blueprint filters must not be empty")
        return cls(
            vehicle_count=_non_negative_int(raw["vehicle_count"], "traffic.vehicle_count", 1000),
            walker_count=_non_negative_int(raw["walker_count"], "traffic.walker_count", 1000),
            pedestrian_crossing_factor=_bounded(
                raw["pedestrian_crossing_factor"],
                "traffic.pedestrian_crossing_factor",
                0.0,
                1.0,
            ),
            vehicle_filter=vehicle_filter,
            walker_filter=walker_filter,
            vehicle_generation=str(raw["vehicle_generation"]),
            walker_generation=str(raw["walker_generation"]),
            global_speed_difference_percent=_bounded(
                raw["global_speed_difference_percent"],
                "traffic.global_speed_difference_percent",
                -100.0,
                100.0,
            ),
            global_distance_to_leading_vehicle=_bounded(
                raw["global_distance_to_leading_vehicle"],
                "traffic.global_distance_to_leading_vehicle",
                0.1,
                100.0,
            ),
            automatic_vehicle_lights=lights,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureRecipe:
    warmup_ticks: int
    duration_ticks: int
    capture_every_ticks: int
    minimum_visible_pixels: int
    minimum_box_width: int
    minimum_box_height: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _validate_keys(
            raw,
            name="capture",
            required={
                "warmup_ticks",
                "duration_ticks",
                "capture_every_ticks",
                "minimum_visible_pixels",
                "minimum_box_width",
                "minimum_box_height",
            },
        )
        duration = _positive_int(raw["duration_ticks"], "capture.duration_ticks", 10_000_000)
        cadence = _positive_int(
            raw["capture_every_ticks"], "capture.capture_every_ticks", 1_000_000
        )
        if cadence > duration:
            raise ValueError("capture.capture_every_ticks cannot exceed duration_ticks")
        return cls(
            warmup_ticks=_non_negative_int(raw["warmup_ticks"], "capture.warmup_ticks", 1_000_000),
            duration_ticks=duration,
            capture_every_ticks=cadence,
            minimum_visible_pixels=_positive_int(
                raw["minimum_visible_pixels"],
                "capture.minimum_visible_pixels",
                1_000_000_000,
            ),
            minimum_box_width=_positive_int(
                raw["minimum_box_width"], "capture.minimum_box_width", 100_000
            ),
            minimum_box_height=_positive_int(
                raw["minimum_box_height"], "capture.minimum_box_height", 100_000
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PropRecipe:
    blueprint_id: str
    relative_to: str
    transform: TransformRecipe

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], index: int) -> Self:
        name = f"props[{index}]"
        _validate_keys(
            raw,
            name=name,
            required={"blueprint_id", "relative_to", "transform"},
        )
        blueprint_id = str(raw["blueprint_id"]).strip()
        if not blueprint_id.startswith("static.prop."):
            raise ValueError(f"{name}.blueprint_id must start with static.prop.")
        relative_to = str(raw["relative_to"])
        if relative_to not in {"ego_start", "world"}:
            raise ValueError(f"{name}.relative_to must be ego_start or world")
        return cls(
            blueprint_id=blueprint_id,
            relative_to=relative_to,
            transform=TransformRecipe.from_mapping(
                _mapping(raw["transform"], f"{name}.transform"),
                f"{name}.transform",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "relative_to": self.relative_to,
            "transform": self.transform.as_dict(),
        }


@dataclass(frozen=True)
class ScenarioRecipe:
    recipe_id: str
    map_name: str
    route_region_id: str
    ego_blueprint: str
    ego_spawn_index: int
    fixed_delta_seconds: float
    repetitions: int
    weather: WeatherRecipe
    traffic: TrafficRecipe
    camera: CameraRecipe
    capture: CaptureRecipe
    props: tuple[PropRecipe, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], index: int) -> Self:
        name = f"recipes[{index}]"
        _validate_keys(
            raw,
            name=name,
            required={
                "recipe_id",
                "map_name",
                "route_region_id",
                "ego_blueprint",
                "ego_spawn_index",
                "fixed_delta_seconds",
                "repetitions",
                "weather",
                "traffic",
                "camera",
                "capture",
                "props",
            },
        )
        map_name = str(raw["map_name"]).strip()
        ego_blueprint = str(raw["ego_blueprint"]).strip()
        route_region_id = _identifier(raw["route_region_id"], f"{name}.route_region_id")
        if not map_name:
            raise ValueError(f"{name}.map_name must not be empty")
        if not ego_blueprint.startswith("vehicle."):
            raise ValueError(f"{name}.ego_blueprint must start with vehicle.")
        fixed_delta = _bounded(
            raw["fixed_delta_seconds"],
            f"{name}.fixed_delta_seconds",
            0.005,
            0.1,
        )
        camera = CameraRecipe.from_mapping(_mapping(raw["camera"], f"{name}.camera"))
        ratio = camera.sensor_tick_seconds / fixed_delta
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"{name}.camera.sensor_tick_seconds must be an integer multiple "
                "of fixed_delta_seconds"
            )
        sensor_tick_multiple = round(ratio)
        capture = CaptureRecipe.from_mapping(_mapping(raw["capture"], f"{name}.capture"))
        if capture.warmup_ticks % sensor_tick_multiple:
            raise ValueError(
                f"{name}.capture.warmup_ticks must be a multiple of the "
                "camera sensor period in world ticks"
            )
        if capture.capture_every_ticks % sensor_tick_multiple:
            raise ValueError(
                f"{name}.capture.capture_every_ticks must be a multiple of the "
                "camera sensor period in world ticks"
            )
        props_raw = _sequence(raw["props"], f"{name}.props")
        return cls(
            recipe_id=_identifier(raw["recipe_id"], f"{name}.recipe_id"),
            map_name=map_name,
            route_region_id=route_region_id,
            ego_blueprint=ego_blueprint,
            ego_spawn_index=_non_negative_int(
                raw["ego_spawn_index"], f"{name}.ego_spawn_index", 1_000_000
            ),
            fixed_delta_seconds=fixed_delta,
            repetitions=_positive_int(raw["repetitions"], f"{name}.repetitions", 10_000),
            weather=WeatherRecipe.from_mapping(_mapping(raw["weather"], f"{name}.weather")),
            traffic=TrafficRecipe.from_mapping(_mapping(raw["traffic"], f"{name}.traffic")),
            camera=camera,
            capture=capture,
            props=tuple(
                PropRecipe.from_mapping(_mapping(item, f"{name}.props[{prop_index}]"), prop_index)
                for prop_index, item in enumerate(props_raw)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "map_name": self.map_name,
            "route_region_id": self.route_region_id,
            "ego_blueprint": self.ego_blueprint,
            "ego_spawn_index": self.ego_spawn_index,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "repetitions": self.repetitions,
            "weather": self.weather.as_dict(),
            "traffic": self.traffic.as_dict(),
            "camera": self.camera.as_dict(),
            "capture": self.capture.as_dict(),
            "props": [prop.as_dict() for prop in self.props],
        }


@dataclass(frozen=True)
class ScenarioSuite:
    suite_id: str
    schema_version: str
    carla_version: str
    master_seed: int
    traffic_manager_port: int
    recipes: tuple[ScenarioRecipe, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _validate_keys(
            raw,
            name="scenario suite",
            required={
                "suite_id",
                "schema_version",
                "carla_version",
                "master_seed",
                "traffic_manager_port",
                "recipes",
            },
        )
        schema_version = str(raw["schema_version"])
        if schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scenario schema {schema_version!r}; "
                f"expected {SCENARIO_SCHEMA_VERSION!r}"
            )
        master_seed = raw["master_seed"]
        if isinstance(master_seed, bool) or not isinstance(master_seed, int):
            raise TypeError("scenario suite master_seed must be an integer")
        if not 0 <= master_seed < 2**63:
            raise ValueError("scenario suite master_seed must be in [0, 2^63)")
        recipes_raw = _sequence(raw["recipes"], "scenario suite recipes")
        if not recipes_raw:
            raise ValueError("scenario suite must contain at least one recipe")
        recipes = tuple(
            ScenarioRecipe.from_mapping(_mapping(item, f"recipes[{index}]"), index)
            for index, item in enumerate(recipes_raw)
        )
        recipe_ids = [recipe.recipe_id for recipe in recipes]
        if len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("scenario recipe IDs must be unique")
        return cls(
            suite_id=_identifier(raw["suite_id"], "scenario suite suite_id"),
            schema_version=schema_version,
            carla_version=str(raw["carla_version"]),
            master_seed=master_seed,
            traffic_manager_port=_positive_int(
                raw["traffic_manager_port"],
                "scenario suite traffic_manager_port",
                65535,
            ),
            recipes=recipes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "schema_version": self.schema_version,
            "carla_version": self.carla_version,
            "master_seed": self.master_seed,
            "traffic_manager_port": self.traffic_manager_port,
            "recipes": [recipe.as_dict() for recipe in self.recipes],
        }


def load_scenario_suite(path: str | Path) -> ScenarioSuite:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return ScenarioSuite.from_mapping(_mapping(payload, "scenario suite"))


__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "CameraRecipe",
    "CaptureRecipe",
    "PropRecipe",
    "ScenarioRecipe",
    "ScenarioSuite",
    "TrafficRecipe",
    "TransformRecipe",
    "WeatherRecipe",
    "load_scenario_suite",
]
