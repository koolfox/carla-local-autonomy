"""CARLA 0.9.16 semantic contract shared by detector and voxel supervision.

Source: https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera
The old pre-0.9.14 table is intentionally NOT inferred from pixel values.
"""

CARLA_SEMANTIC_VERSION = "carla-0.9.16"
DYNAMIC_ROAD_USER_TAGS = (12, 13, 14, 15, 16, 17, 18, 19)
COMPACT_VOXEL_TAGS = {
    3: 0,  # building
    1: 1,
    24: 1,  # road and road marking
    2: 2,  # sidewalk
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,  # vehicles
    12: 4,
    13: 4,  # pedestrians and riders: documented compact human class
    9: 5,  # vegetation
}
