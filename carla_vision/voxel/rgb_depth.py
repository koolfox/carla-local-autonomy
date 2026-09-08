"""RGB-only predicted-depth voxel surfaces for a non-actuating live view.

This is monocular depth estimation followed by geometry, not a learned occupancy
forecast, ground-truth CARLA depth, calibrated free-space detector, or planner.
The optional model dependencies and weights are loaded only by ``load`` or the
first ``predict`` call; callers must run those operations off the Drive thread.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import FREE, OCCUPIED, UNKNOWN, VoxelGridSpec
from .geometry import backproject_depth, carla_rotation_matrix

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
# Resolved from the public Hugging Face model API on 2026-09-03. Both processor
# and safetensors weights use this immutable revision, never the moving main ref.
MODEL_REVISION = "fd2c22027eaf20374204f14099b8341e1925ad39"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}"
_MAX_GRID_CELLS = 2_000_000
_MAX_INPUT_PIXELS = 16_777_216
_MAX_RAY_STEPS = 1024
_RAY_CHUNK = 128
_LIMITS = (
    "Metric distance is a monocular model estimate, not calibrated CARLA ground truth.",
    "Free cells are sampled rays before predicted surfaces, not verified safe space.",
    "Behind surfaces, outside the camera view, and unsampled space remain unknown.",
    "No road, ground, lane, object-class, motion, or future-occupancy labels are inferred.",
    "This view has no vehicle-control or safety authority.",
)


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Explicit static camera-to-display mounting transform, never a live pose.

    Coordinates are x-forward, y-right, z-up. Angles are roll/pitch/yaw in
    degrees: positive yaw turns forward towards right; positive pitch raises
    forward towards up; positive roll turns right towards down.
    """

    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("translation_xyz", "rotation_rpy_degrees"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite numbers")
            if name == "translation_xyz" and max(map(abs, values)) > 1000.0:
                raise ValueError("static camera translation must be within 1000 metres")
            object.__setattr__(self, name, values)

    def rotation_matrix(self) -> np.ndarray:
        roll, pitch, yaw = self.rotation_rpy_degrees
        return carla_rotation_matrix(roll=roll, pitch=pitch, yaw=yaw).astype(np.float32)


@dataclass(frozen=True, slots=True)
class RgbDepthVoxelResult:
    spec: VoxelGridSpec
    occupancy: np.ndarray
    surface_points_xyz: np.ndarray
    surface_rgb: np.ndarray
    surface_indices_zyx: np.ndarray
    metadata: Mapping[str, Any]


def _bounded_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _validate_options(
    spec: VoxelGridSpec,
    pixel_stride: int,
    max_rays: int,
    max_surface_points: int,
    max_depth_m: float,
) -> None:
    if not isinstance(spec, VoxelGridSpec) or math.prod(spec.shape) > _MAX_GRID_CELLS:
        raise ValueError(f"voxel grid must contain at most {_MAX_GRID_CELLS} cells")
    _bounded_integer(pixel_stride, "pixel_stride", 256)
    _bounded_integer(max_rays, "max_rays", 16_384)
    _bounded_integer(max_surface_points, "max_surface_points", 20_000)
    if not math.isfinite(max_depth_m) or not 0.1 < max_depth_m <= 1000.0:
        raise ValueError("max_depth_m must be finite and in (0.1, 1000]")


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be a uint8 HxWx3 RGB array")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or height * width > _MAX_INPUT_PIXELS:
        raise ValueError(f"rgb must contain between 1 and {_MAX_INPUT_PIXELS} pixels")
    return image


def _readonly(array: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def voxelize_predicted_depth(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    *,
    spec: VoxelGridSpec,
    fov: float,
    pixel_stride: int = 8,
    max_rays: int = 4096,
    max_surface_points: int = 6000,
    max_depth_m: float = 80.0,
    camera_mount: CameraMount | None = None,
    frame: int | None = None,
    timestamp: float | None = None,
    sequence: int | None = None,
) -> RgbDepthVoxelResult:
    """Voxelize predicted axial depth in metres; invalid pixels remain unknown.

    RGB and depth must be aligned to the same exact source image. Surface points
    retain their floating-point metric positions, with one representative RGB
    sample per occupied voxel. Compact surface output may be capped; occupancy
    retains every sampled in-grid surface. Free rays are intentionally sparse.
    """

    _validate_options(spec, pixel_stride, max_rays, max_surface_points, max_depth_m)
    image = _validate_rgb(rgb)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("fov must be finite and in (0, 180) degrees")
    if camera_mount is not None and not isinstance(camera_mount, CameraMount):
        raise TypeError("camera_mount must be an explicit static CameraMount")
    for name, value in (("frame", frame), ("sequence", sequence)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"source {name} must be a non-negative integer")
    if timestamp is not None and not math.isfinite(timestamp):
        raise ValueError("source timestamp must be finite")
    if not np.issubdtype(np.asarray(depth_m).dtype, np.floating):
        raise TypeError("depth_m must contain floating-point metric depths, not an image mask")
    with np.errstate(over="ignore", invalid="ignore"):
        depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError("depth_m must be a 2D array aligned with rgb")

    height, width = depth.shape
    stride = max(pixel_stride, math.ceil(math.sqrt(height * width / max_rays)))
    while math.ceil(height / stride) * math.ceil(width / stride) > max_rays:
        stride += 1
    points, rows, cols = backproject_depth(
        depth,
        fov_deg=fov,
        pixel_stride=stride,
        min_depth_m=0.1,
        max_depth_m=max_depth_m,
    )
    origin = np.zeros(3, dtype=np.float32)
    if camera_mount is not None:
        origin = np.asarray(camera_mount.translation_xyz, dtype=np.float32)
        points = points @ camera_mount.rotation_matrix().T + origin
    occupancy = np.full(spec.shape, UNKNOWN, dtype=np.int8)
    indices, surface_valid = spec.metric_to_indices(points)
    surface_indices = indices[surface_valid]
    surface_points = points[surface_valid]
    surface_colors = image[rows[surface_valid], cols[surface_valid]]
    if surface_indices.size:
        linear = np.ravel_multi_index(surface_indices.T, spec.shape)
        _unique, representatives = np.unique(linear, return_index=True)
        surface_indices = surface_indices[representatives]
        surface_points = surface_points[representatives]
        surface_colors = surface_colors[representatives]
        occupancy[tuple(surface_indices.T)] = OCCUPIED
    surface_count = len(surface_points)

    # Install all surfaces first. A farther ray must never erase an occupied
    # cell, nor carve through that cell into space occluded by its near surface.
    ray_step = max(0.1, spec.resolution / 2.0)
    surface_margin = spec.resolution * math.sqrt(3.0)
    if len(points):
        vectors = points - origin
        lengths = np.linalg.norm(vectors, axis=1)
        extent = np.asarray(
            [
                max(abs(spec.x_min - origin[0]), abs(spec.x_max - origin[0])),
                max(abs(spec.y_min - origin[1]), abs(spec.y_max - origin[1])),
                max(abs(spec.z_min - origin[2]), abs(spec.z_max - origin[2])),
            ]
        )
        ray_limit = min(float(np.max(lengths)), float(np.linalg.norm(extent)))
        sample_count = min(_MAX_RAY_STEPS, max(0, math.ceil(ray_limit / ray_step)))
        distances = np.arange(1, sample_count + 1, dtype=np.float32) * ray_step
        for offset in range(0, len(points), _RAY_CHUNK):
            chunk = vectors[offset : offset + _RAY_CHUNK]
            chunk_lengths = lengths[offset : offset + _RAY_CHUNK]
            directions = chunk / chunk_lengths[:, None]
            samples = origin + directions[:, None, :] * distances[None, :, None]
            ray_indices, inside = spec.metric_to_indices(samples.reshape(-1, 3))
            shape = (len(chunk), len(distances))
            inside = inside.reshape(shape)
            before_surface = distances[None, :] < (chunk_lengths[:, None] - surface_margin)
            occupied = np.zeros(shape, dtype=bool)
            valid_indices = ray_indices[inside.reshape(-1)]
            occupied[inside] = occupancy[tuple(valid_indices.T)] == OCCUPIED
            occluded = np.maximum.accumulate(occupied, axis=1)
            free = inside & before_surface & ~occluded
            free_indices = ray_indices[free.reshape(-1)]
            occupancy[tuple(free_indices.T)] = FREE

    if surface_count > max_surface_points:
        selected = np.linspace(0, surface_count - 1, max_surface_points, dtype=np.int64)
        surface_indices = surface_indices[selected]
        surface_points = surface_points[selected]
        surface_colors = surface_colors[selected]
    metadata: dict[str, Any] = {
        "source_frame": frame,
        "source_timestamp": timestamp,
        "source_sequence": sequence,
        "source_shape_hw": (height, width),
        "horizontal_fov_degrees": float(fov),
        "coordinate_frame": "camera" if camera_mount is None else "static_mount",
        "axes": "x_forward_y_right_z_up",
        "camera_origin_xyz": tuple(float(value) for value in origin),
        "method": "predicted_metric_depth_surface_voxelization",
        "rgb_only": True,
        "depth_scale_approximate": True,
        "confidence_calibrated": False,
        "actuation_applied": False,
        "has_semantic_labels": False,
        "has_future_prediction": False,
        "grid": spec.as_dict(),
        "pixel_stride": stride,
        "valid_sampled_rays": len(points),
        "occupied_voxel_count": surface_count,
        "displayed_surface_count": len(surface_points),
        "ray_step_m": ray_step,
        "ray_step_limit": _MAX_RAY_STEPS,
        "max_depth_m": float(max_depth_m),
        "limits": _LIMITS,
    }
    return RgbDepthVoxelResult(
        spec=spec,
        occupancy=_readonly(occupancy, np.int8),
        surface_points_xyz=_readonly(surface_points, np.float32),
        surface_rgb=_readonly(surface_colors, np.uint8),
        surface_indices_zyx=_readonly(surface_indices, np.int32),
        metadata=MappingProxyType(metadata),
    )


class RgbDepthVoxelPredictor:
    """Lazy, safetensors-only Depth Anything V2 metric-outdoor inference.

    ``load()`` and ``predict()`` are blocking operations intended for an
    independent latest-frame worker, never the Drive/control thread. A local
    checkpoint must be a compatible Hugging Face directory; it is local-only.
    """

    def __init__(
        self,
        *,
        spec: VoxelGridSpec | None = None,
        device: str = "cpu",
        checkpoint: str | Path | None = None,
        workspace: str | Path | None = None,
        pixel_stride: int = 8,
        max_rays: int = 4096,
        max_surface_points: int = 6000,
        max_depth_m: float = 80.0,
        camera_mount: CameraMount | None = None,
    ) -> None:
        self.spec = VoxelGridSpec() if spec is None else spec
        _validate_options(self.spec, pixel_stride, max_rays, max_surface_points, max_depth_m)
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")
        if camera_mount is not None and not isinstance(camera_mount, CameraMount):
            raise TypeError("camera_mount must be an explicit static CameraMount")
        self.device = device
        self.checkpoint = None if checkpoint is None else Path(checkpoint).expanduser()
        self.workspace = workspace
        self.pixel_stride = pixel_stride
        self.max_rays = max_rays
        self.max_surface_points = max_surface_points
        self.max_depth_m = max_depth_m
        self.camera_mount = camera_mount
        self._lock = threading.RLock()
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._resolved_checkpoint: str | None = None
        self._weights_sha256: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
            except ImportError as error:
                raise RuntimeError("RGB voxel depth requires the optional voxel dependencies") from error
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for RGB voxel depth")
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available for RGB voxel depth")
            options: dict[str, Any] = {"trust_remote_code": False}
            from ..model_storage import model_directory
            cache_dir = str(model_directory(self.workspace) / "huggingface")
            if self.checkpoint is None:
                source = MODEL_ID
                options.update(revision=MODEL_REVISION, local_files_only=False, cache_dir=cache_dir)
            else:
                path = self.checkpoint.resolve(strict=True)
                if not path.is_dir():
                    raise ValueError("checkpoint must be a local Hugging Face model directory")
                source = str(path)
                options["local_files_only"] = True
            processor = transformers.AutoImageProcessor.from_pretrained(source, **options)
            model = transformers.AutoModelForDepthEstimation.from_pretrained(
                source, use_safetensors=True, **options
            )
            if (
                getattr(model.config, "model_type", None) != "depth_anything"
                or getattr(model.config, "depth_estimation_type", None) != "metric"
            ):
                raise ValueError("RGB voxel depth requires a metric Depth Anything model")
            model.to(self.device)
            model.eval()
            if self.checkpoint is None:
                hub = importlib.import_module("huggingface_hub")
                weights = hub.try_to_load_from_cache(
                    MODEL_ID, "model.safetensors", revision=MODEL_REVISION, cache_dir=cache_dir,
                )
            else:
                weights = str(path / "model.safetensors")
            if isinstance(weights, str) and Path(weights).is_file():
                with Path(weights).open("rb") as handle:
                    self._weights_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            self._torch = torch
            self._processor = processor
            self._model = model
            self._resolved_checkpoint = source

    def _predict_depth(self, rgb: np.ndarray) -> np.ndarray:
        with self._lock:
            self.load()
            torch = self._torch
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
                depth = outputs.predicted_depth
                if depth.ndim != 3 or depth.shape[0] != 1:
                    raise RuntimeError("metric depth model must return one BxHxW depth image")
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
            return depth.detach().cpu().numpy().astype(np.float32, copy=False)

    def predict(
        self,
        rgb: np.ndarray,
        fov: float,
        *,
        frame: int | None = None,
        timestamp: float | None = None,
        sequence: int | None = None,
    ) -> RgbDepthVoxelResult:
        image = _validate_rgb(rgb)
        if not math.isfinite(fov) or not 0.0 < fov < 180.0:
            raise ValueError("fov must be finite and in (0, 180) degrees")
        depth = self._predict_depth(image)
        result = voxelize_predicted_depth(
            depth,
            image,
            spec=self.spec,
            fov=fov,
            pixel_stride=self.pixel_stride,
            max_rays=self.max_rays,
            max_surface_points=self.max_surface_points,
            max_depth_m=self.max_depth_m,
            camera_mount=self.camera_mount,
            frame=frame,
            timestamp=timestamp,
            sequence=sequence,
        )
        return replace(
            result,
            metadata=MappingProxyType(
                {
                    **result.metadata,
                    "model_id": MODEL_ID if self.checkpoint is None else "local_metric_depth",
                    "model_revision": MODEL_REVISION if self.checkpoint is None else None,
                    "model_source": MODEL_URL if self.checkpoint is None else self._resolved_checkpoint,
                    "device": self.device,
                    "weights_sha256": self._weights_sha256,
                }
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._torch = None
