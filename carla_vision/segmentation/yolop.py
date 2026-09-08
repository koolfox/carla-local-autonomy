"""Official YOLOP ONNX road/lane heads, independent of object detection.

Preprocessing follows hustvl/YOLOP test_onnx.py: RGB, centered 114 padding,
INTER_AREA resize and ImageNet normalization. No upstream Python is executed.
"""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .contracts import SegmentationConfig, SegmentationMetadata, SegmentationResult

YOLOP_REVISION = "8d8f68df318c71f01d6f813c024df646c7d1978f"
YOLOP_SHA256 = "cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696"
YOLOP_URL = f"https://raw.githubusercontent.com/hustvl/YOLOP/{YOLOP_REVISION}/weights/yolop-640-640.onnx"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def default_checkpoint() -> Path:
    directory = Path.home() / ".cache" / "carla-vision" / "yolop"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "yolop-640-640.onnx"
    if target.is_file() and _digest(target) == YOLOP_SHA256:
        return target
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(YOLOP_URL, timeout=30) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 40 * 1024 * 1024:
                        raise ValueError("YOLOP download exceeds expected size")
                    output.write(chunk)
            output.flush()
            if _digest(temporary) != YOLOP_SHA256:
                raise ValueError("YOLOP download checksum mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(target)
    return target


def preprocess(image: np.ndarray, height: int, width: int):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("YOLOP input must be nonempty uint8 BGR")
    ratio = min(height / image.shape[0], width / image.shape[1])
    w, h = round(image.shape[1] * ratio), round(image.shape[0] * ratio)
    left, top = (width - w) // 2, (height - h) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top+h, left:left+w] = cv2.resize(image[:, :, ::-1], (w, h), interpolation=cv2.INTER_AREA)
    tensor = (canvas.astype(np.float32) / 255 - np.array([.485, .456, .406], np.float32))
    tensor /= np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), (left, top, w, h)


class YoloPSegmenter:
    def __init__(self, config: SegmentationConfig):
        import onnxruntime as ort

        if config.device not in {"cpu", "cuda"}:
            raise ValueError("YOLOP supports cpu or cuda; MPS is not an ONNX provider")
        provider = "CUDAExecutionProvider" if config.device == "cuda" else "CPUExecutionProvider"
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"YOLOP requires {provider}; install the matching ONNX Runtime")
        path = Path(config.checkpoint) if config.checkpoint else default_checkpoint()
        if not path.is_file() or path.suffix != ".onnx":
            raise ValueError("YOLOP requires an ONNX checkpoint")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("YOLOP requires one float32 input")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(x, int) and 0 < x <= 2048 for x in shape[2:]):
            raise ValueError("YOLOP requires fixed 1x3xHxW input, at most 2048 pixels")
        if not {"drive_area_seg", "lane_line_seg"} <= {x.name for x in self._session.get_outputs()}:
            raise ValueError("YOLOP checkpoint lacks named road/lane outputs")
        self._input = inputs[0].name
        self._height, self._width = shape[2:]
        self.name = f"yolop:{path.stem}"
        self.metadata = SegmentationMetadata(
            name=self.name, backend="yolop-onnx", checkpoint=str(path), device=config.device,
            resolved_revision=_digest(path), source_labels={0: "other", 1: "road", 2: "road_line"},
            source_to_canonical={0: "other", 1: "road", 2: "road_line"},
        )

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        if self._session is None:
            raise RuntimeError("YOLOP is closed")
        tensor, (left, top, w, h) = preprocess(image_bgr, self._height, self._width)
        outputs = self._session.run(["drive_area_seg", "lane_line_seg"], {self._input: tensor})
        masks, scores = [], []
        size = (image_bgr.shape[1], image_bgr.shape[0])
        for output in outputs:
            if output.shape != (1, 2, self._height, self._width) or not np.isfinite(output).all():
                raise ValueError("YOLOP segmentation output has invalid shape or values")
            if output.min() < 0 or output.max() > 1:
                raise ValueError("YOLOP expects sigmoid probabilities from the official export")
            cropped = output[0, :, top:top+h, left:left+w]
            restored = np.stack([cv2.resize(channel, size, interpolation=cv2.INTER_LINEAR) for channel in cropped])
            masks.append(restored.argmax(0).astype(np.uint8))
            scores.append(restored.max(0))
        classes = masks[0].copy()
        classes[masks[1] == 1] = 2  # Lane marking takes priority over road.
        confidence = np.where(masks[1] == 1, scores[1], scores[0]).astype(np.float32)
        return SegmentationResult(classes, confidence, self.name)

    def close(self):
        self._session = None
