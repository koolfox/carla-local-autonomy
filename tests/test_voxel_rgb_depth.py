from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from carla_vision.voxel.contracts import FREE, OCCUPIED, UNKNOWN, VoxelGridSpec
from carla_vision.voxel.rgb_depth import CameraMount, voxelize_predicted_depth


@pytest.fixture
def spec() -> VoxelGridSpec:
    return VoxelGridSpec(
        x_min=-4.0,
        x_max=8.0,
        y_min=-5.0,
        y_max=5.0,
        z_min=-5.0,
        z_max=5.0,
        resolution=0.25,
    )


def _rgb(height: int = 4, width: int = 4) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint32)
    return (values % 256).astype(np.uint8).reshape(height, width, 3)


def _at(occupancy: np.ndarray, spec: VoxelGridSpec, point: tuple[float, float, float]):
    indices, valid = spec.metric_to_indices(np.asarray([point], dtype=np.float32))
    assert valid[0], "the test's query point must be inside the voxel grid"
    return occupancy[tuple(indices[0])]


@pytest.mark.parametrize(
    ("row", "column", "expected"),
    [
        (2, 2, (2.0, 0.0, 0.0)),
        (2, 3, (2.0, 1.0, 0.0)),
        (1, 2, (2.0, 0.0, 1.0)),
        (3, 1, (2.0, -1.0, -1.0)),
    ],
)
def test_backprojection_uses_forward_right_up_axes(
    spec: VoxelGridSpec,
    row: int,
    column: int,
    expected: tuple[float, float, float],
) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[row, column] = 2.0
    rgb = _rgb()
    result = voxelize_predicted_depth(depth, rgb, spec=spec, fov=90.0, pixel_stride=1)

    assert result.spec == spec
    assert result.occupancy.shape == spec.shape
    assert result.occupancy.dtype == np.int8
    assert set(np.unique(result.occupancy)) <= {UNKNOWN, FREE, OCCUPIED}
    np.testing.assert_allclose(result.surface_points_xyz, [expected], atol=1e-6)
    np.testing.assert_array_equal(result.surface_rgb, [rgb[row, column]])
    indices, valid = spec.metric_to_indices(result.surface_points_xyz)
    assert valid.all()
    np.testing.assert_array_equal(result.surface_indices_zyx, indices)
    assert _at(result.occupancy, spec, expected) == OCCUPIED


def test_fractional_depth_is_preserved_in_surface_geometry(spec: VoxelGridSpec) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 2.75
    result = voxelize_predicted_depth(depth, _rgb(), spec=spec, fov=90.0, pixel_stride=1)

    assert result.surface_points_xyz.dtype == np.float32
    assert result.surface_rgb.dtype == np.uint8
    assert result.surface_indices_zyx.dtype == np.int32
    np.testing.assert_allclose(result.surface_points_xyz, [[2.75, 0.0, 0.0]])
    assert _at(result.occupancy, spec, (2.75, 0.0, 0.0)) == OCCUPIED
    assert _at(result.occupancy, spec, (2.0, 0.0, 0.0)) != OCCUPIED


def test_only_finite_positive_depth_within_limit_is_used(spec: VoxelGridSpec) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0] = [np.nan, np.inf, -np.inf, -1.0]
    depth[2, 2] = 0.75
    depth[2, 3] = 3.25
    rgb = _rgb()
    result = voxelize_predicted_depth(
        depth,
        rgb,
        spec=spec,
        fov=90.0,
        pixel_stride=1,
        max_depth_m=3.0,
    )

    np.testing.assert_allclose(result.surface_points_xyz, [[0.75, 0.0, 0.0]])
    np.testing.assert_array_equal(result.surface_rgb, [rgb[2, 2]])
    assert np.count_nonzero(result.occupancy == OCCUPIED) == 1
    assert _at(result.occupancy, spec, (3.25, 1.625, 0.0)) == UNKNOWN


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf, 81.0])
def test_no_valid_depth_leaves_every_voxel_unknown(spec: VoxelGridSpec, value: float) -> None:
    result = voxelize_predicted_depth(
        np.full((4, 4), value, dtype=np.float32),
        _rgb(),
        spec=spec,
        fov=90.0,
        pixel_stride=1,
    )

    assert np.all(result.occupancy == UNKNOWN)
    assert result.surface_points_xyz.shape == (0, 3)
    assert result.surface_rgb.shape == (0, 3)
    assert result.surface_indices_zyx.shape == (0, 3)


def test_unobserved_fov_and_space_behind_surface_stay_unknown(spec: VoxelGridSpec) -> None:
    result = voxelize_predicted_depth(
        np.full((8, 8), 4.0, dtype=np.float32),
        _rgb(8, 8),
        spec=spec,
        fov=60.0,
        pixel_stride=1,
    )

    assert np.any(result.occupancy == FREE)
    assert _at(result.occupancy, spec, (1.0, 4.0, 0.0)) == UNKNOWN
    assert _at(result.occupancy, spec, (-1.0, 0.0, 0.0)) == UNKNOWN
    assert _at(result.occupancy, spec, (6.0, 0.0, 0.0)) == UNKNOWN
    assert _at(result.occupancy, spec, (4.0, 0.0, 0.0)) == OCCUPIED


@pytest.mark.parametrize(("near_column", "far_column"), [(50, 51), (51, 50)])
def test_far_ray_never_erases_an_existing_near_surface(
    spec: VoxelGridSpec, near_column: int, far_column: int
) -> None:
    depth = np.zeros((4, 100), dtype=np.float32)
    depth[2, near_column] = 2.25
    depth[2, far_column] = 5.25
    result = voxelize_predicted_depth(
        depth, _rgb(4, 100), spec=spec, fov=90.0, pixel_stride=1
    )

    near_point = (2.25, (near_column - 50) * 2.25 / 50.0, 0.0)
    far_point = (5.25, (far_column - 50) * 5.25 / 50.0, 0.0)
    assert _at(result.occupancy, spec, near_point) == OCCUPIED
    assert _at(result.occupancy, spec, far_point) == OCCUPIED


def test_pixel_stride_samples_aligned_depth_and_color(spec: VoxelGridSpec) -> None:
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[4, 4] = 2.0
    depth[3, 3] = 3.0
    rgb = _rgb(8, 8)
    result = voxelize_predicted_depth(depth, rgb, spec=spec, fov=90.0, pixel_stride=2)

    np.testing.assert_allclose(result.surface_points_xyz, [[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(result.surface_rgb, [rgb[4, 4]])


def test_bounded_point_cloud_does_not_discard_occupied_grid_cells(spec: VoxelGridSpec) -> None:
    depth = np.full((16, 16), 3.0, dtype=np.float32)
    rgb = _rgb(16, 16)
    kwargs = dict(spec=spec, fov=90.0, pixel_stride=1, max_rays=23, max_surface_points=5)
    result = voxelize_predicted_depth(depth, rgb, **kwargs)
    repeated = voxelize_predicted_depth(depth, rgb, **kwargs)

    assert 0 < len(result.surface_points_xyz) <= 5
    assert len(result.surface_rgb) == len(result.surface_points_xyz)
    assert len(result.surface_indices_zyx) == len(result.surface_points_xyz)
    assert 5 < np.count_nonzero(result.occupancy == OCCUPIED) <= 23
    assert len(np.unique(result.surface_indices_zyx, axis=0)) == len(result.surface_points_xyz)
    indices = result.surface_indices_zyx
    assert np.all(result.occupancy[tuple(indices.T)] == OCCUPIED)
    np.testing.assert_array_equal(result.occupancy, repeated.occupancy)
    np.testing.assert_array_equal(result.surface_points_xyz, repeated.surface_points_xyz)
    np.testing.assert_array_equal(result.surface_rgb, repeated.surface_rgb)


def test_ray_budget_also_bounds_returned_surfaces(spec: VoxelGridSpec) -> None:
    result = voxelize_predicted_depth(
        np.full((16, 16), 3.0, dtype=np.float32),
        _rgb(16, 16),
        spec=spec,
        fov=90.0,
        pixel_stride=1,
        max_rays=7,
        max_surface_points=20,
    )

    assert 0 < len(result.surface_points_xyz) <= 7
    assert np.count_nonzero(result.occupancy == OCCUPIED) <= 7


@pytest.mark.parametrize(
    "name", ["occupancy", "surface_points_xyz", "surface_rgb", "surface_indices_zyx"]
)
def test_result_arrays_are_read_only(spec: VoxelGridSpec, name: str) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 2.75
    result = voxelize_predicted_depth(depth, _rgb(), spec=spec, fov=90.0, pixel_stride=1)
    array = getattr(result, name)

    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.flat[0] = 0


def test_result_owns_its_surface_data_and_preserves_source_metadata(spec: VoxelGridSpec) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 2.75
    rgb = _rgb()
    expected_color = rgb[2, 2].copy()
    result = voxelize_predicted_depth(
        depth,
        rgb,
        spec=spec,
        fov=90.0,
        pixel_stride=1,
        frame=42,
        timestamp=12.5,
        sequence=7,
    )
    depth[:] = 0
    rgb[:] = 0

    np.testing.assert_allclose(result.surface_points_xyz, [[2.75, 0.0, 0.0]])
    np.testing.assert_array_equal(result.surface_rgb, [expected_color])
    assert result.metadata["source_frame"] == 42
    assert result.metadata["source_timestamp"] == 12.5
    assert result.metadata["source_sequence"] == 7
    assert result.metadata["coordinate_frame"] == "camera"
    assert result.metadata["confidence_calibrated"] is False
    assert result.metadata["actuation_applied"] is False


def test_static_mount_translates_surfaces_and_ray_origin(spec: VoxelGridSpec) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2] = 2.75
    mount = CameraMount(translation_xyz=(1.5, 1.0, 1.0))
    result = voxelize_predicted_depth(
        depth,
        _rgb(),
        spec=spec,
        fov=90.0,
        pixel_stride=1,
        camera_mount=mount,
    )

    np.testing.assert_allclose(result.surface_points_xyz, [[4.25, 1.0, 1.0]])
    assert _at(result.occupancy, spec, (4.25, 1.0, 1.0)) == OCCUPIED
    assert _at(result.occupancy, spec, (2.5, 1.0, 1.0)) == FREE
    assert _at(result.occupancy, spec, (0.5, 0.0, 0.0)) == UNKNOWN
    assert _at(result.occupancy, spec, (2.75, 0.0, 0.0)) == UNKNOWN
    assert result.metadata["coordinate_frame"] == "static_mount"
    with pytest.raises(FrozenInstanceError):
        mount.translation_xyz = (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("rotation", "pixel", "expected"),
    [
        ((0.0, 0.0, 90.0), (2, 2), (0.0, 2.0, 0.0)),
        ((0.0, 90.0, 0.0), (2, 2), (0.0, 0.0, 2.0)),
        ((90.0, 0.0, 0.0), (2, 3), (2.0, 0.0, -1.0)),
    ],
)
def test_static_mount_rotation_uses_carla_axes(
    spec: VoxelGridSpec,
    rotation: tuple[float, float, float],
    pixel: tuple[int, int],
    expected: tuple[float, float, float],
) -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[pixel] = 2.0
    result = voxelize_predicted_depth(
        depth,
        _rgb(),
        spec=spec,
        fov=90.0,
        pixel_stride=1,
        camera_mount=CameraMount(rotation_rpy_degrees=rotation),
    )

    np.testing.assert_allclose(result.surface_points_xyz, [expected], atol=1e-6)
    indices = result.surface_indices_zyx
    assert np.all(result.occupancy[tuple(indices.T)] == OCCUPIED)


@pytest.mark.parametrize("fov", [0.0, 180.0, -10.0, np.nan, np.inf])
def test_invalid_fov_is_rejected(spec: VoxelGridSpec, fov: float) -> None:
    with pytest.raises(ValueError):
        voxelize_predicted_depth(np.ones((4, 4), np.float32), _rgb(), spec=spec, fov=fov)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pixel_stride": 0},
        {"pixel_stride": 1.5},
        {"pixel_stride": 257},
        {"max_rays": 0},
        {"max_rays": 16385},
        {"max_surface_points": 0},
        {"max_surface_points": 20001},
        {"max_depth_m": 0.0},
        {"max_depth_m": np.nan},
        {"max_depth_m": np.inf},
    ],
)
def test_invalid_processing_limits_are_rejected(spec: VoxelGridSpec, kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        voxelize_predicted_depth(
            np.ones((4, 4), np.float32), _rgb(), spec=spec, fov=90.0, **kwargs
        )


@pytest.mark.parametrize(
    ("depth", "rgb"),
    [
        (np.ones((4, 4, 1), np.float32), _rgb()),
        (np.ones((4, 4), np.float32), _rgb(3, 4)),
        (np.ones((4, 4), np.float32), np.ones((4, 4), np.uint8)),
        (np.ones((4, 4), np.float32), np.ones((4, 4, 4), np.uint8)),
        (np.ones((4, 4), np.float32), np.ones((4, 4, 3), np.float32)),
        (np.ones((4, 4), np.uint8), _rgb()),
    ],
)
def test_incompatible_images_are_rejected(
    spec: VoxelGridSpec, depth: np.ndarray, rgb: np.ndarray
) -> None:
    with pytest.raises((TypeError, ValueError)):
        voxelize_predicted_depth(depth, rgb, spec=spec, fov=90.0)


def test_grid_allocation_is_bounded_before_work() -> None:
    too_large = VoxelGridSpec(
        x_min=0, x_max=200, y_min=-100, y_max=100, z_min=-50, z_max=50, resolution=1
    )
    with pytest.raises(ValueError):
        voxelize_predicted_depth(
            np.ones((4, 4), np.float32), _rgb(), spec=too_large, fov=90.0
        )


def test_import_and_predictor_construction_need_no_model_or_carla_imports() -> None:
    repository = Path(__file__).resolve().parents[1]
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'carla'}:
        raise AssertionError('eager heavyweight import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from carla_vision.voxel.contracts import VoxelGridSpec
from carla_vision.voxel.rgb_depth import RgbDepthVoxelPredictor
predictor = RgbDepthVoxelPredictor(spec=VoxelGridSpec(), device='cpu', checkpoint=None)
assert predictor is not None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
