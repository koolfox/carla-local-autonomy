from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..contracts import Detection, DetectorConfig, DetectorMetadata


class UltralyticsDetector:
    """Adapter that normalizes Ultralytics YOLO and RT-DETR results."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        architecture: Literal["yolo", "rtdetr"],
    ) -> None:
        if config.weights is None:
            raise ValueError("Ultralytics detector requires a weights path or model name")

        # Lazy import keeps contract tests independent from heavyweight runtimes.
        from ultralytics import RTDETR, YOLO

        self._architecture = architecture
        self._config = config
        loader = YOLO if architecture == "yolo" else RTDETR
        self._model = loader(str(config.weights))
        model_stem = Path(str(config.weights)).stem
        self._name = f"ultralytics-{architecture}:{model_stem}"
        self._metadata = DetectorMetadata(
            name=self._name,
            backend=f"ultralytics-{architecture}",
            weights=str(config.weights),
            device=config.device,
            image_size=config.image_size,
            confidence=config.confidence,
            extra=dict(config.options),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> DetectorMetadata:
        return self._metadata

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names[class_id])
        return str(names[class_id])

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be a uint8 HxWx3 array")

        predict_options = {
            "imgsz": self._config.image_size,
            "conf": self._config.confidence,
            "device": self._config.device,
            "verbose": False,
            **dict(self._config.options),
        }
        results = self._model(image_bgr, **predict_options)
        if not results:
            return ()
        result = results[0]
        if result.boxes is None:
            return ()

        detections: list[Detection] = []
        for box in result.boxes:
            source_class_id = int(box.cls.item())
            label = self._class_name(result.names, source_class_id)
            confidence = float(box.conf.item())
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=source_class_id,
                    source_class_id=source_class_id,
                    label=label,
                    confidence=confidence,
                    xyxy=xyxy,
                )
            )
        return tuple(detections)

    def close(self) -> None:
        self._model = None
