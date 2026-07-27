from __future__ import annotations

import numpy as np
import pytest

from carla_vision.voxel.contracts import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    CameraVoxelPrediction,
    VoxelGridSpec,
    validate_camera_voxel_prediction,
)
from carla_vision.voxel.geometry import (
    backproject_depth,
    decode_carla_depth_bgra,
    occupied_iou,
    raycast_voxel_grid,
)


def test_grid_shape_and_metric_indices() -> None:
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=4.0,
        y_min=-2.0,
        y_max=2.0,
        z_min=-1.0,
        z_max=1.0,
        resolution=0.5,
    )
    assert spec.shape == (4, 8, 8)
    indices, valid = spec.metric_to_indices(np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert valid.tolist() == [True, False]
    assert indices[0].tolist() == [2, 4, 2]


def test_decode_carla_depth_extremes() -> None:
    raw = bytes([0, 0, 0, 255, 255, 255, 255, 255])
    depth = decode_carla_depth_bgra(raw, width=2, height=1)
    assert depth[0, 0] == pytest.approx(0.0)
    assert depth[0, 1] == pytest.approx(1000.0)


def test_backproject_center_pixel_faces_forward() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 5.0
    points, rows, cols = backproject_depth(depth, fov_deg=90.0)
    assert rows.tolist() == [2]
    assert cols.tolist() == [2]
    assert points[0].tolist() == pytest.approx([5.0, 0.0, 0.0])


def test_raycast_marks_free_space_and_semantic_surface() -> None:
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=4.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=-1.0,
        z_max=1.0,
        resolution=0.5,
    )
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 2.5
    semantics = np.zeros((4, 4), dtype=np.uint8)
    semantics[2, 2] = 10
    occupancy, semantic_grid = raycast_voxel_grid(
        depth,
        semantics,
        spec=spec,
        fov_deg=90.0,
        pixel_stride=1,
        ray_step_m=0.5,
    )
    center_z = 2
    center_y = 2
    assert occupancy[center_z, center_y, 1] == FREE
    assert occupancy[center_z, center_y, 5] == OCCUPIED
    assert semantic_grid[center_z, center_y, 5] == 10
    assert np.any(occupancy == UNKNOWN)


def test_prediction_contract_and_iou() -> None:
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=2.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        resolution=1.0,
    )
    probability = np.zeros((2, *spec.shape), dtype=np.float32)
    probability[:, 0, 1, 1] = 0.9
    validated = validate_camera_voxel_prediction(
        CameraVoxelPrediction(probability, horizons_s=(0.0, 1.0)),
        spec,
    )
    target = np.zeros(spec.shape, dtype=np.int8)
    target[0, 1, 1] = OCCUPIED
    assert validated.occupancy_probability.shape == (2, *spec.shape)
    assert occupied_iou(validated.occupancy_probability[0], target) == pytest.approx(1.0)


def test_prediction_rejects_wrong_shape() -> None:
    spec = VoxelGridSpec()
    with pytest.raises(ValueError, match="shape"):
        validate_camera_voxel_prediction(
            CameraVoxelPrediction(np.zeros((1, 2, 3, 4), dtype=np.float32), (0.0,)),
            spec,
        )
