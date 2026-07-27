"""Frozen configuration for non-actuating live vision-shadow matrices."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from ..policy import VisionPolicyConfig

SHADOW_MATRIX_CONFIG_SCHEMA_VERSION = "1.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")


def _strict(raw: Mapping[str, Any], fields: set[str], name: str) -> None:
    actual = {str(key) for key in raw}
    missing = sorted(fields - actual)
    extra = sorted(actual - fields)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {', '.join(extra)}")


def _identifier(value: Any, name: str) -> str:
    result = str(value).strip()
    if not _ID_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must be lowercase and use letters, digits, '.' or '-'")
    return result


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _path(value: Any, *, base: Path, name: str) -> Path | None:
    if value is None:
        return None
    candidate = Path(_nonempty(value, name)).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


@dataclass(frozen=True)
class ShadowDetector:
    backend: str | None
    weights: Path | None
    factory: str | None
    model_package: Path | None
    device: str
    image_size: int
    confidence: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        _strict(
            raw,
            {
                "backend",
                "weights",
                "factory",
                "model_package",
                "device",
                "image_size",
                "confidence",
            },
            "shadow detector",
        )
        model_package = _path(
            raw["model_package"],
            base=base,
            name="detector.model_package",
        )
        backend_raw = raw["backend"]
        backend = (
            None if backend_raw is None else str(backend_raw).strip().lower().replace("_", "-")
        )
        aliases = {
            "rt-detr": "rtdetr",
            "ultralytics-rtdetr": "rtdetr",
            "ultralytics-yolo": "yolo",
        }
        backend = aliases.get(backend, backend)
        weights = _path(raw["weights"], base=base, name="detector.weights")
        factory_raw = raw["factory"]
        factory = None if factory_raw is None else _nonempty(factory_raw, "detector.factory")
        if model_package is not None:
            if any(value is not None for value in (backend, weights, factory)):
                raise ValueError("model_package is mutually exclusive with loose detector fields")
        else:
            if backend not in {"rtdetr", "yolo", "custom"}:
                raise ValueError("loose detector backend must be rtdetr, yolo, or custom")
            if backend in {"rtdetr", "yolo"} and weights is None:
                raise ValueError(f"{backend} shadow detector requires weights")
            if backend == "custom" and factory is None:
                raise ValueError("custom shadow detector requires factory")
            if backend != "custom" and factory is not None:
                raise ValueError("built-in shadow detector cannot declare factory")
        image_size = raw["image_size"]
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise TypeError("detector.image_size must be an integer")
        if not 1 <= image_size <= 16384:
            raise ValueError("detector.image_size must be in [1, 16384]")
        confidence = float(raw["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("detector.confidence must be in [0, 1]")
        return cls(
            backend=backend,
            weights=weights,
            factory=factory,
            model_package=model_package,
            device=_nonempty(raw["device"], "detector.device"),
            image_size=image_size,
            confidence=confidence,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "weights": str(self.weights) if self.weights is not None else None,
            "factory": self.factory,
            "model_package": (str(self.model_package) if self.model_package is not None else None),
            "device": self.device,
            "image_size": self.image_size,
            "confidence": self.confidence,
        }


def _policy(raw: Any, *, base: Path) -> VisionPolicyConfig:
    if not isinstance(raw, Mapping):
        raise TypeError("shadow policy must be an object")
    _strict(
        raw,
        {"backend", "factory", "checkpoint", "device", "options"},
        "shadow policy",
    )
    options = raw["options"]
    if not isinstance(options, Mapping):
        raise TypeError("shadow policy options must be an object")
    normalized_options = {str(key): value for key, value in options.items()}
    try:
        json.dumps(normalized_options, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("shadow policy options must be finite JSON") from error
    return VisionPolicyConfig(
        backend=str(raw["backend"]),
        factory=(None if raw["factory"] is None else _nonempty(raw["factory"], "policy.factory")),
        checkpoint=_path(
            raw["checkpoint"],
            base=base,
            name="policy.checkpoint",
        ),
        device=_nonempty(raw["device"], "policy.device"),
        options=normalized_options,
    )


@dataclass(frozen=True)
class ShadowMatrixCell:
    cell_id: str
    label: str
    detector: ShadowDetector
    policy: VisionPolicyConfig
    repetitions: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        _strict(
            raw,
            {"cell_id", "label", "detector", "policy", "repetitions"},
            "shadow matrix cell",
        )
        detector = raw["detector"]
        if not isinstance(detector, Mapping):
            raise TypeError("shadow cell detector must be an object")
        repetitions = raw["repetitions"]
        if isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise TypeError("shadow cell repetitions must be an integer")
        if not 1 <= repetitions <= 100:
            raise ValueError("shadow cell repetitions must be in [1, 100]")
        return cls(
            cell_id=_identifier(raw["cell_id"], "cell.cell_id"),
            label=_nonempty(raw["label"], "cell.label"),
            detector=ShadowDetector.from_mapping(detector, base=base),
            policy=_policy(raw["policy"], base=base),
            repetitions=repetitions,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "label": self.label,
            "detector": self.detector.as_dict(),
            "policy": self.policy.as_dict(),
            "repetitions": self.repetitions,
        }


@dataclass(frozen=True)
class ShadowMatrixConfig:
    schema_version: str
    matrix_id: str
    title: str
    purpose: str
    host: str
    port: int
    vehicle_id: int
    camera_id: int
    expected_map: str | None
    resolution: tuple[int, int]
    camera_fps: float
    camera_fov: float
    duration_seconds: float
    max_stale_seconds: float
    control: str
    view: str
    record_video: bool
    continue_on_failure: bool
    cells: tuple[ShadowMatrixCell, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Path) -> Self:
        required = {
            "schema_version",
            "matrix_id",
            "title",
            "purpose",
            "host",
            "port",
            "vehicle_id",
            "camera_id",
            "expected_map",
            "resolution",
            "camera_fps",
            "camera_fov",
            "duration_seconds",
            "max_stale_seconds",
            "control",
            "view",
            "record_video",
            "continue_on_failure",
            "cells",
        }
        _strict(raw, required, "shadow matrix config")
        schema = str(raw["schema_version"])
        if schema != SHADOW_MATRIX_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported shadow matrix schema {schema!r}; "
                f"expected {SHADOW_MATRIX_CONFIG_SCHEMA_VERSION!r}"
            )
        purpose = str(raw["purpose"]).strip().lower()
        if purpose not in {"development", "confirmatory"}:
            raise ValueError("shadow matrix purpose must be development or confirmatory")
        control = str(raw["control"]).strip().lower()
        if control not in {"none", "teacher"}:
            raise ValueError("shadow matrix control must be none or teacher")
        view = str(raw["view"]).strip().lower()
        if view not in {"none", "overlay", "split"}:
            raise ValueError("shadow matrix view must be none, overlay, or split")
        resolution_raw = raw["resolution"]
        if (
            isinstance(resolution_raw, (str, bytes))
            or not isinstance(resolution_raw, Sequence)
            or len(resolution_raw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in resolution_raw
            )
        ):
            raise TypeError("shadow matrix resolution must be [width, height]")
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        if not 320 <= resolution[0] <= 3840 or not 180 <= resolution[1] <= 2160:
            raise ValueError("shadow matrix resolution is outside runtime bounds")
        integers: dict[str, int] = {}
        for name, minimum, maximum in (
            ("port", 1, 65535),
            ("vehicle_id", 0, 2**31 - 1),
            ("camera_id", 0, 2**31 - 1),
        ):
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
            integers[name] = value
        camera_fps = float(raw["camera_fps"])
        camera_fov = float(raw["camera_fov"])
        duration = float(raw["duration_seconds"])
        stale = float(raw["max_stale_seconds"])
        if not 0.0 < camera_fps <= 60.0:
            raise ValueError("camera_fps must be in (0, 60]")
        if not 30.0 <= camera_fov <= 150.0:
            raise ValueError("camera_fov must be in [30, 150]")
        if not 0.0 < duration <= 3600.0:
            raise ValueError("duration_seconds must be in (0, 3600]")
        if stale <= 0.0:
            raise ValueError("max_stale_seconds must be positive")
        for name in ("record_video", "continue_on_failure"):
            if not isinstance(raw[name], bool):
                raise TypeError(f"{name} must be boolean")
        cells_raw = raw["cells"]
        if isinstance(cells_raw, (str, bytes)) or not isinstance(cells_raw, Sequence):
            raise TypeError("shadow matrix cells must be an array")
        cells = tuple(
            ShadowMatrixCell.from_mapping(item, base=base)
            if isinstance(item, Mapping)
            else (_ for _ in ()).throw(TypeError("shadow matrix cell must be an object"))
            for item in cells_raw
        )
        if not cells:
            raise ValueError("shadow matrix requires at least one cell")
        cell_ids = [cell.cell_id for cell in cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("shadow matrix cell IDs must be unique")
        if purpose == "confirmatory" and any(cell.detector.model_package is None for cell in cells):
            raise ValueError("confirmatory shadow matrix requires verified model packages")
        expected_map = raw["expected_map"]
        return cls(
            schema_version=schema,
            matrix_id=_identifier(raw["matrix_id"], "matrix_id"),
            title=_nonempty(raw["title"], "title"),
            purpose=purpose,
            host=_nonempty(raw["host"], "host"),
            port=integers["port"],
            vehicle_id=integers["vehicle_id"],
            camera_id=integers["camera_id"],
            expected_map=(
                None if expected_map is None else _nonempty(expected_map, "expected_map")
            ),
            resolution=resolution,
            camera_fps=camera_fps,
            camera_fov=camera_fov,
            duration_seconds=duration,
            max_stale_seconds=stale,
            control=control,
            view=view,
            record_video=raw["record_video"],
            continue_on_failure=raw["continue_on_failure"],
            cells=cells,
        )

    @property
    def planned_run_count(self) -> int:
        return sum(cell.repetitions for cell in self.cells)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "title": self.title,
            "purpose": self.purpose,
            "host": self.host,
            "port": self.port,
            "vehicle_id": self.vehicle_id,
            "camera_id": self.camera_id,
            "expected_map": self.expected_map,
            "resolution": list(self.resolution),
            "camera_fps": self.camera_fps,
            "camera_fov": self.camera_fov,
            "duration_seconds": self.duration_seconds,
            "max_stale_seconds": self.max_stale_seconds,
            "control": self.control,
            "view": self.view,
            "record_video": self.record_video,
            "continue_on_failure": self.continue_on_failure,
            "cells": [cell.as_dict() for cell in self.cells],
        }


def load_shadow_matrix_config(path: str | Path) -> ShadowMatrixConfig:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("shadow matrix config must contain an object")
    return ShadowMatrixConfig.from_mapping(payload, base=resolved.parent)


__all__ = [
    "SHADOW_MATRIX_CONFIG_SCHEMA_VERSION",
    "ShadowDetector",
    "ShadowMatrixCell",
    "ShadowMatrixConfig",
    "load_shadow_matrix_config",
]
