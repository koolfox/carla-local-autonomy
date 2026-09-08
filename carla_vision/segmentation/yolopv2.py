"""Official CAIC-AD YOLOPv2 TorchScript road/lane inference.

Only the pinned release is executable. RGB /255 (no ImageNet normalization),
stride-32 letterboxing and separate two-class road / one-channel lane heads
follow the upstream demo. Padding is removed dynamically for any source aspect.
"""
from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..model_storage import model_directory
from .contracts import SegmentationConfig, SegmentationMetadata, SegmentationResult
from .yolop import _digest

URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
SHA256 = "f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c"


def default_checkpoint(workspace=None) -> Path:
    directory = model_directory(workspace) / "yolopv2"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "yolopv2.pt"
    if target.is_file() and _digest(target) == SHA256:
        return target
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(URL, timeout=30) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 170 * 1024 * 1024:
                        raise ValueError("YOLOPv2 download exceeds expected size")
                    stream.write(chunk)
            stream.flush()
            if _digest(temporary) != SHA256:
                raise ValueError("YOLOPv2 checksum mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(target)
    return target


def preprocess(image: np.ndarray):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("YOLOPv2 input must be nonempty uint8 BGR")
    ratio = 640 / max(image.shape[:2])
    w, h = max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))
    dw, dh = (640 - w) % 32, (640 - h) % 32
    left, top = dw // 2, dh // 2
    canvas = np.full((h + dh, w + dw, 3), 114, np.uint8)
    canvas[top:top+h, left:left+w] = cv2.resize(image[:, :, ::-1], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255, (left, top, w, h)


class YoloPv2Segmenter:
    def __init__(self, config: SegmentationConfig):
        import torch

        if config.device not in {"cpu", "cuda"}:
            raise ValueError("YOLOPv2 supports cpu or cuda")
        path = Path(config.checkpoint) if config.checkpoint else default_checkpoint(config.options.get("workspace"))
        if not path.is_file() or _digest(path) != SHA256:
            raise ValueError("YOLOPv2 requires the checksum-verified official V0.0.1 checkpoint")
        self._device = config.device
        self._model = torch.jit.load(str(path), map_location=config.device).float().eval()
        self.name = "yolopv2:V0.0.1"
        self.metadata = SegmentationMetadata(
            name=self.name, backend="yolopv2-torchscript", checkpoint=str(path),
            device=config.device, resolved_revision=SHA256,
            source_labels={0: "other", 1: "road", 2: "road_line"},
            source_to_canonical={0: "other", 1: "road", 2: "road_line"},
        )

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        import torch

        if self._model is None:
            raise RuntimeError("YOLOPv2 is closed")
        tensor, (left, top, w, h) = preprocess(image_bgr)
        with torch.inference_mode():
            _, road, lane = self._model(torch.from_numpy(tensor).to(self._device))
        size = (image_bgr.shape[1], image_bgr.shape[0])
        restored = []
        for head, channels in ((road, 2), (lane, 1)):
            values = head.float().cpu().numpy()
            if values.shape != (1, channels, *tensor.shape[2:]) or not np.isfinite(values).all():
                raise ValueError("Invalid YOLOPv2 output shape/values")
            if values.min() < 0 or values.max() > 1:
                raise ValueError("YOLOPv2 output must be probabilities")
            restored.append(np.stack([cv2.resize(channel[top:top+h, left:left+w], size, interpolation=cv2.INTER_LINEAR) for channel in values[0]]))
        road, lane = restored
        classes = road.argmax(0).astype(np.uint8)
        lane_mask = lane[0] > .5
        classes[lane_mask] = 2
        confidence = np.where(lane_mask, lane[0], road.max(0)).astype(np.float32)
        return SegmentationResult(classes, confidence, self.name)

    def close(self):
        self._model = None
