from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from carla_vision.native.world_worker import WorldWorker


@dataclass
class FakeLocation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FakeRotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class FakeTransform:
    location: FakeLocation
    rotation: FakeRotation


class FakeCarla:
    Location = FakeLocation
    Rotation = FakeRotation
    Transform = FakeTransform


@dataclass
class Bounds:
    location: FakeLocation
    extent: FakeLocation


class Ego:
    def __init__(self, extent: tuple[float, float, float]) -> None:
        self.bounding_box = Bounds(
            location=FakeLocation(0.0, 0.0, extent[2]),
            extent=FakeLocation(*extent),
        )
        self._transform = FakeTransform(FakeLocation(), FakeRotation())

    def get_transform(self) -> FakeTransform:
        return self._transform


VEHICLE_CLASSES = {
    "compact": (1.85, 0.82, 0.72),
    "sedan": (2.45, 0.96, 0.80),
    "suv": (2.55, 1.05, 0.98),
    "truck": (4.60, 1.25, 1.75),
    "bus": (6.10, 1.30, 1.65),
}

EXTERIOR_PRESETS = {
    "orbit": (325.0, -10.0, 6.5),
    "front": (0.0, -8.0, 6.5),
    "rear": (180.0, -8.0, 6.5),
    "top": (0.0, -70.0, 8.0),
}


def worker() -> WorldWorker:
    value = object.__new__(WorldWorker)
    value._carla = FakeCarla()
    return value


def required_eye_distance(
    extent: tuple[float, float, float],
    *,
    yaw: float,
    pitch: float,
    width: int = 1280,
    height: int = 720,
    fov: float = 65.0,
) -> float:
    extent_x, extent_y, extent_z = extent
    aspect = width / height
    horizontal_half_fov = math.radians(fov) / 2.0
    vertical_half_fov = math.atan(math.tan(horizontal_half_fov) / aspect)
    azimuth = math.radians(yaw)
    pitch_radians = math.radians(pitch)
    half_width = abs(math.sin(azimuth)) * extent_x + abs(math.cos(azimuth)) * extent_y
    half_depth = abs(math.cos(azimuth)) * extent_x + abs(math.sin(azimuth)) * extent_y
    projected_height = (
        abs(math.sin(pitch_radians)) * half_depth
        + abs(math.cos(pitch_radians)) * extent_z
    )
    projected_depth = (
        abs(math.cos(pitch_radians)) * half_depth
        + abs(math.sin(pitch_radians)) * extent_z
    )
    return projected_depth + max(
        half_width / math.tan(horizontal_half_fov),
        projected_height / math.tan(vertical_half_fov),
    )


@pytest.mark.parametrize("vehicle_class,extent", VEHICLE_CLASSES.items())
@pytest.mark.parametrize("preset,view", EXTERIOR_PRESETS.items())
def test_exterior_presets_fit_representative_vehicle_bounds(
    vehicle_class: str,
    extent: tuple[float, float, float],
    preset: str,
    view: tuple[float, float, float],
) -> None:
    del vehicle_class
    yaw, pitch, distance = view
    ego = Ego(extent)
    transform = worker()._garage_camera_transform(
        ego,
        yaw=yaw,
        pitch=pitch,
        distance=distance,
        width=1280,
        height=720,
        fov=65.0,
        preset=preset,
    )
    target = ego.bounding_box.location
    actual_distance = math.dist(
        (transform.location.x, transform.location.y, transform.location.z),
        (target.x, target.y, target.z),
    )
    minimum_fit = required_eye_distance(extent, yaw=yaw, pitch=pitch)

    assert math.isfinite(actual_distance)
    assert actual_distance >= minimum_fit * 1.02
    assert transform.rotation.pitch == pytest.approx(pitch)
    expected_yaw = yaw + 180.0
    assert transform.rotation.yaw == pytest.approx(expected_yaw)


@pytest.mark.parametrize("vehicle_class,extent", VEHICLE_CLASSES.items())
def test_cockpit_preset_uses_a_bounded_vehicle_relative_eye_point(
    vehicle_class: str,
    extent: tuple[float, float, float],
) -> None:
    del vehicle_class
    ego = Ego(extent)
    transform = worker()._garage_camera_transform(
        ego,
        yaw=0.0,
        pitch=0.0,
        distance=4.0,
        width=1280,
        height=720,
        fov=65.0,
        preset="cockpit",
    )
    center = ego.bounding_box.location
    half = ego.bounding_box.extent

    assert center.x - half.x <= transform.location.x <= center.x + half.x
    assert center.y - half.y <= transform.location.y <= center.y + half.y
    assert center.z - half.z <= transform.location.z <= center.z + half.z
    assert transform.rotation == FakeRotation()
