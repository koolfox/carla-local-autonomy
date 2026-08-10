from __future__ import annotations

import numpy as np
import pytest

from carla_vision.voxel.contracts import VoxelGridSpec
from carla_vision.voxel.flow_teacher import (
    build_dynamic_voxel_flow,
    decode_carla_optical_flow,
)


def test_decode_carla_optical_flow_round_trip() -> None:
    values = np.asarray([[[0.25, -0.5], [1.0, 0.0]]], dtype=np.float32)
    decoded = decode_carla_optical_flow(values.tobytes(), width=2, height=1)
    np.testing.assert_allclose(decoded, values)


def test_dynamic_depth_change_becomes_forward_voxel_velocity() -> None:
    depth_source = np.full((4, 4), 5.0, dtype=np.float32)
    depth_target = np.full((4, 4), 6.0, dtype=np.float32)
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[2, 2] = 10
    optical = np.zeros((4, 4, 2), dtype=np.float32)
    spec = VoxelGridSpec(
        x_min=0.0,
        x_max=10.0,
        y_min=-5.0,
        y_max=5.0,
        z_min=-5.0,
        z_max=5.0,
        resolution=1.0,
    )
    identity = np.eye(4, dtype=np.float64)
    flow, valid, stats = build_dynamic_voxel_flow(
        depth_source,
        depth_target,
        semantic,
        optical,
        source_to_world=identity,
        target_to_world=identity,
        spec=spec,
        fov_deg=90.0,
        dt_s=0.5,
        pixel_stride=1,
    )
    assert stats.dynamic_points == 1
    assert stats.valid_voxels == int(valid.sum()) == 1
    assert float(flow[0][valid][0]) == pytest.approx(2.0, abs=1e-5)
    assert float(flow[1][valid][0]) == pytest.approx(0.0, abs=1e-5)
    assert float(flow[2][valid][0]) == pytest.approx(0.0, abs=1e-5)


def test_invalid_flow_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected"):
        decode_carla_optical_flow(b"bad", width=2, height=2)
