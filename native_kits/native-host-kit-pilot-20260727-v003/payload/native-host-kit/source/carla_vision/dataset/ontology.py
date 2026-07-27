from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CarlaSemanticTag(IntEnum):
    """CARLA 0.9.16 ``CityObjectLabel`` values carried by semantic cameras."""

    NONE = 0
    ROADS = 1
    SIDEWALKS = 2
    BUILDINGS = 3
    WALLS = 4
    FENCES = 5
    POLES = 6
    TRAFFIC_LIGHT = 7
    TRAFFIC_SIGN = 8
    VEGETATION = 9
    TERRAIN = 10
    SKY = 11
    PEDESTRIAN = 12
    RIDER = 13
    CAR = 14
    TRUCK = 15
    BUS = 16
    TRAIN = 17
    MOTORCYCLE = 18
    BICYCLE = 19
    STATIC = 20
    DYNAMIC = 21
    OTHER = 22
    WATER = 23
    ROAD_LINES = 24
    GROUND = 25
    BRIDGE = 26
    RAIL_TRACK = 27
    GUARD_RAIL = 28


CARLA_SEMANTIC_TAGS: tuple[CarlaSemanticTag, ...] = tuple(CarlaSemanticTag)


@dataclass(frozen=True, slots=True)
class DetectorCategory:
    """Stable detector class independent of CARLA's sparse semantic tag ids."""

    id: int
    name: str
    semantic_tag: CarlaSemanticTag

    def as_dict(self) -> dict[str, int | str]:
        return {
            "id": self.id,
            "name": self.name,
            "semantic_tag": int(self.semantic_tag),
        }


# The order is a dataset contract. Append new categories rather than reordering
# existing ids, so labels remain comparable between runs.
DETECTOR_CATEGORIES: tuple[DetectorCategory, ...] = (
    DetectorCategory(0, "traffic_light", CarlaSemanticTag.TRAFFIC_LIGHT),
    DetectorCategory(1, "traffic_sign", CarlaSemanticTag.TRAFFIC_SIGN),
    DetectorCategory(2, "pedestrian", CarlaSemanticTag.PEDESTRIAN),
    DetectorCategory(3, "rider", CarlaSemanticTag.RIDER),
    DetectorCategory(4, "car", CarlaSemanticTag.CAR),
    DetectorCategory(5, "truck", CarlaSemanticTag.TRUCK),
    DetectorCategory(6, "bus", CarlaSemanticTag.BUS),
    DetectorCategory(7, "train", CarlaSemanticTag.TRAIN),
    DetectorCategory(8, "motorcycle", CarlaSemanticTag.MOTORCYCLE),
    DetectorCategory(9, "bicycle", CarlaSemanticTag.BICYCLE),
)

DETECTOR_CATEGORY_BY_TAG: dict[CarlaSemanticTag, DetectorCategory] = {
    category.semantic_tag: category for category in DETECTOR_CATEGORIES
}

THING_TAGS: tuple[CarlaSemanticTag, ...] = tuple(
    category.semantic_tag for category in DETECTOR_CATEGORIES
)


def detector_category_for_tag(
    semantic_tag: CarlaSemanticTag | int,
) -> DetectorCategory | None:
    """Return the stable detector category for an official thing tag."""

    try:
        normalized = CarlaSemanticTag(int(semantic_tag))
    except (TypeError, ValueError):
        return None
    return DETECTOR_CATEGORY_BY_TAG.get(normalized)
