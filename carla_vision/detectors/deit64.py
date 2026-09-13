"""Notebook-compatible M9 -> RGB sign crops -> DeiT-64/68, without actuation.

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
    """Stage B (64) or Stage C (68); retained class name for existing callers.

    Both notebooks use the same fixed backbone and MLP head, changing only the
    final class count. Never expand or randomly initialize a head at inference.
    """

    def __init__(self, config: SignClassifierConfig, *, device: str) -> None:
        self.config = config
        self.labels = read_sign_ontology(config.ontology)
        try:
            import timm
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "DeiT needs the local deit64 extra: pip install -e '.[m9,deit64]'"
            ) from error
        self._torch = torch
        self.device = resolve_torch_device(device)
        # Stage C stores training metrics/history alongside tensors. Depending
        # on sklearn/NumPy versions these contain NumPy floating scalars. Allow
        # only those data types (including the NumPy 1.x pickle path), never
        # arbitrary checkpoint globals or a weights_only=False fallback.
        try:
            from numpy._core.multiarray import scalar
        except ImportError:  # NumPy 1.x
            from numpy.core.multiarray import scalar
        with torch.serialization.safe_globals([
            (scalar, "numpy.core.multiarray.scalar"),
            (scalar, "numpy._core.multiarray.scalar"),
            np.dtype, type(np.dtype(np.float32)), type(np.dtype(np.float64)),
        ]):
            checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("DeiT checkpoint must contain model_state_dict tensors")
        state = checkpoint["model_state_dict"]
        if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("DeiT state_dict must contain only tensors")
        output_weight, output_bias = state.get("head.3.weight"), state.get("head.3.bias")
        if (output_weight is None or output_weight.ndim != 2
                or output_weight.shape[0] not in {64, 68} or output_weight.shape[1] != 512
                or output_bias is None or output_bias.shape != (output_weight.shape[0],)):
            raise ValueError("DeiT state_dict requires a trained 512→64 or 512→68 output head (head.3)")
        self.num_classes = int(output_weight.shape[0])
        if len(self.labels) != self.num_classes:
            raise ValueError(
                f"DeiT checkpoint has {self.num_classes} classes but ontology has {len(self.labels)}; "
                "select the matching training ontology CSV, not the other stage's labels"
            )
        self.identity = {
            "name": f"DeiT-{self.num_classes}", "architecture": "deit_small_patch16_224",
            "num_classes": self.num_classes,
            "checkpoint_sha256": _sha256(config.checkpoint),
            "ontology_sha256": _sha256(config.ontology),
            **config.as_dict(), "device": str(self.device),
            "torch_version": torch.__version__, "timm_version": timm.__version__,
            "preprocessing": "source RGB crop; PIL bilinear 224x224; ImageNet mean/std",
            "score": "softmax; independent of M9 confidence; not calibrated",
        }
        model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=self.num_classes)
        model.head = nn.Sequential(nn.Linear(384, 512), nn.SiLU(), nn.Dropout(0.3), nn.Linear(512, self.num_classes))
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
                if logits.shape != (len(tensors), self.num_classes) or not torch.isfinite(logits).all():
                    raise ValueError(f"DeiT-{self.num_classes} returned invalid logits")
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
        self.name = f"{detector.name} + {classifier.identity['name']}"
        self.detection_only_name = detector.name
        self.metadata: DetectorMetadata = replace(
            detector.metadata, name=self.name,
            extra={**detector.metadata.extra, "sign_classifier": classifier.identity},
        )

    def infer_without_signs(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        return self._infer(image_bgr, read_signs=False)

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        return self._infer(image_bgr, read_signs=True)

    def _infer(self, image_bgr: np.ndarray, *, read_signs: bool) -> tuple[Detection, ...]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sign recognition detector is closed")
            detections = list(self._detector.infer(image_bgr))
            if not read_signs:
                return tuple(detections)
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
