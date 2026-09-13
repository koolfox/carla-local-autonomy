"""Notebook-compatible M9 -> RGB sign crops -> DeiT-64, without actuation.

M9 boxes and both independent head scores are preserved. Sign confidence is a
separate, uncalibrated softmax score, not a replacement detector confidence.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Detection, Detector, DetectorMetadata
from ..pytorch_loading import resolve_torch_device
from .sign_config import SignClassifierConfig, read_sign_ontology


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sign_crop_box(
    box: tuple[float, float, float, float], width: int, height: int, scale: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    # A box outside the source cannot become a valid sign by adding context.
    if min(x2, width) - max(x1, 0) < 2 or min(y2, height) - max(y1, 0) < 2:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return (max(0, round(cx - half_w)), max(0, round(cy - half_h)),
            min(width, round(cx + half_w)), min(height, round(cy + half_h)))


class DeiT64Classifier:
    """Fixed architecture; tensor-only load; no hub downloads or executable factory."""

    def __init__(self, config: SignClassifierConfig, *, device: str) -> None:
        self.config = config
        self.labels = read_sign_ontology(config.ontology)
        try:
            import timm
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "DeiT-64 needs the local deit64 extra: pip install -e '.[m9,deit64]'"
            ) from error
        self._torch = torch
        self.device = resolve_torch_device(device)
        self.identity = {
            "name": "DeiT-64", "architecture": "deit_small_patch16_224",
            "checkpoint_sha256": _sha256(config.checkpoint),
            "ontology_sha256": _sha256(config.ontology),
            **config.as_dict(), "device": str(self.device),
            "torch_version": torch.__version__, "timm_version": timm.__version__,
            "preprocessing": "source RGB crop; PIL bilinear 224x224; ImageNet mean/std",
            "score": "softmax; independent of M9 confidence; not calibrated",
        }
        checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("DeiT-64 checkpoint must contain model_state_dict tensors")
        state = checkpoint["model_state_dict"]
        if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("DeiT-64 state_dict must contain only tensors")
        model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=64)
        model.head = nn.Sequential(nn.Linear(384, 512), nn.SiLU(), nn.Dropout(0.3), nn.Linear(512, 64))
        model.load_state_dict(state, strict=True)
        self._model = model.eval().to(self.device)

    def predict(self, crops_rgb: list[np.ndarray]) -> list[dict[str, Any]]:
        if not crops_rgb:
            return []
        from PIL import Image

        torch = self._torch
        results: list[dict[str, Any]] = []
        # Bound classifier memory independently of the number of M9 detections.
        for offset in range(0, len(crops_rgb), 16):
            tensors = []
            for crop in crops_rgb[offset:offset + 16]:
                resized = Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR)
                values = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
                values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array(
                    [0.229, 0.224, 0.225], dtype=np.float32,
                )[:, None, None]
                tensors.append(torch.from_numpy(values))
            with torch.inference_mode():
                logits = self._model(torch.stack(tensors).to(self.device))
                if logits.shape != (len(tensors), 64) or not torch.isfinite(logits).all():
                    raise ValueError("DeiT-64 returned invalid logits")
                scores, ids = logits.softmax(dim=1).max(dim=1)
            for score, index in zip(scores.cpu().tolist(), ids.cpu().tolist(), strict=True):
                results.append({"class_id": index, "label": self.labels[index],
                                "confidence": score, "accepted": score >= self.config.confidence})
        return results

    def close(self) -> None:
        self._model = None


class SignRecognitionDetector:
    """Composable Detector contract. One exact source frame, with serialized M9 hooks."""

    def __init__(self, detector: Detector, classifier: DeiT64Classifier) -> None:
        self._detector = detector
        self._classifier = classifier
        self._lock = threading.Lock()
        self._closed = False
        self.name = f"{detector.name} + DeiT-64"
        self.metadata: DetectorMetadata = replace(
            detector.metadata, name=self.name,
            extra={**detector.metadata.extra, "sign_classifier": classifier.identity},
        )

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sign recognition detector is closed")
            detections = list(self._detector.infer(image_bgr))
            crops, indices, boxes = [], [], []
            height, width = image_bgr.shape[:2]
            for index, detection in enumerate(detections):
                if detection.attributes.get("fine_label") != "traffic_signs":
                    continue
                box = sign_crop_box(detection.xyxy, width, height, self._classifier.config.crop_scale)
                if box is None:
                    detections[index] = replace(detection, attributes={**detection.attributes,
                        "sign_classification": {"accepted": False, "reason": "invalid_or_tiny_crop"}})
                    continue
                x1, y1, x2, y2 = box
                crops.append(image_bgr[y1:y2, x1:x2, ::-1].copy())
                indices.append(index)
                boxes.append(box)
            predictions = self._classifier.predict(crops) if crops else []
            for index, box, prediction in zip(indices, boxes, predictions, strict=True):
                detection = detections[index]
                detections[index] = replace(detection, attributes={**detection.attributes,
                    "sign_classification": {**prediction, "crop_xyxy": list(box)}})
            return tuple(detections)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._detector.close()
                finally:
                    self._classifier.close()
