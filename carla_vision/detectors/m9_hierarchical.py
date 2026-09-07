from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contracts import Detection, DetectorConfig, DetectorMetadata
from ..pytorch_loading import resolve_torch_device

M9_IMAGE_SIZE = 800
M9_FINE_THRESHOLD = 0.10
M9_MAX_PREDICTIONS = 100
M9_ALPHA = 1.20
M9_BETA = 0.50
M9_GAMMA = 1.10
M9_STAGE = "m9_precal_m6_query_film_img800"
M9_CERTIFIED_SHA256 = "8cd8db62a9a77257294a3e145839d94438e516d6225e03e20fdb7599f7df5d09"
M9_TRAINING_ULTRALYTICS = "8.4.137"
M9_COARSE_NAMES = (
    "vehicles",
    "road_users",
    "two_wheelers",
    "traffic_controls",
)


class QueryFiLMAdapter(nn.Module):
    """Exact Query-FiLM module used by the notebook-produced M9 checkpoint."""

    def __init__(self, dim: int = 256, hidden_dim: int = 512, scale: float = 0.10) -> None:
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.context_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, dec_features: torch.Tensor) -> torch.Tensor:
        context = self.context_norm(dec_features.mean(dim=1))
        gamma_raw, beta_raw = self.mlp(context).chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma_raw)
        beta = self.scale * torch.tanh(beta_raw)
        return dec_features * gamma[:, None, :] + beta[:, None, :]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_supported_ultralytics() -> str:
    version = importlib.metadata.version("ultralytics")
    if tuple(version.split(".")[:2]) != ("8", "4"):
        raise RuntimeError(
            "M9 hierarchical detector was trained with Ultralytics 8.4.137 and requires "
            f"an 8.4.x runtime; installed version is {version!r}"
        )
    return version


def _load_certified_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    """Load only the exact certified legacy notebook checkpoint.

    The notebook saved a complete ``RTDETRDetectionModel`` object and therefore requires
    pickle-capable deserialization. This exception is intentionally quarantined here and
    guarded by an exact SHA-256 before ``weights_only=False`` is ever reached. Generic model
    loaders in the repository must not copy this behavior.
    """

    if path.is_symlink():
        raise ValueError("M9 checkpoint must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("M9 checkpoint must be a regular file")

    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != M9_CERTIFIED_SHA256:
        raise ValueError(
            "selected checkpoint is not the certified M9 artifact; "
            f"expected sha256={M9_CERTIFIED_SHA256}, got {actual_sha256}"
        )

    _require_supported_ultralytics()

    main_module = sys.modules.get("__main__")
    sentinel = object()
    previous = sentinel
    if main_module is not None:
        previous = getattr(main_module, "QueryFiLMAdapter", sentinel)
        setattr(main_module, "QueryFiLMAdapter", QueryFiLMAdapter)

    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    finally:
        if main_module is not None:
            if previous is sentinel:
                delattr(main_module, "QueryFiLMAdapter")
            else:
                setattr(main_module, "QueryFiLMAdapter", previous)

    if not isinstance(payload, dict):
        raise RuntimeError("certified M9 checkpoint payload must be a dictionary")
    if payload.get("stage") != M9_STAGE:
        raise RuntimeError(
            f"unexpected M9 checkpoint stage {payload.get('stage')!r}; expected {M9_STAGE!r}"
        )
    if payload.get("IMG_SIZE") != M9_IMAGE_SIZE:
        raise RuntimeError(
            f"unexpected M9 image size {payload.get('IMG_SIZE')!r}; expected {M9_IMAGE_SIZE}"
        )
    if "model" not in payload:
        raise RuntimeError("certified M9 checkpoint has no 'model' entry")
    return payload, actual_sha256


def _final_decoder_outputs(output: Any, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError("unexpected M9 RT-DETR output envelope")
    raw = output[1]
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        raise RuntimeError("unexpected M9 raw RT-DETR output")

    boxes = raw[0]
    fine_logits = raw[1]
    if not isinstance(boxes, torch.Tensor) or not isinstance(fine_logits, torch.Tensor):
        raise RuntimeError("M9 raw RT-DETR boxes/scores must be tensors")
    if boxes.ndim == 4:
        boxes = boxes[-1]
    if fine_logits.ndim == 4:
        fine_logits = fine_logits[-1]
    if boxes.ndim != 3 or fine_logits.ndim != 3:
        raise RuntimeError(
            f"unexpected M9 decoder tensor shapes boxes={tuple(boxes.shape)} "
            f"scores={tuple(fine_logits.shape)}"
        )
    if boxes.shape[0] != batch_size or fine_logits.shape[0] != batch_size:
        raise RuntimeError("M9 decoder batch dimension does not match input batch")
    return boxes, fine_logits


def _boxes_to_original_xyxy(
    raw_boxes: torch.Tensor,
    *,
    source_width: int,
    source_height: int,
) -> torch.Tensor:
    boxes = raw_boxes.clone()
    if boxes.numel() and float(boxes.detach().max().item()) <= 2.0:
        boxes = boxes * float(M9_IMAGE_SIZE)

    cx, cy, width, height = boxes.unbind(-1)
    boxes = torch.stack(
        (
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ),
        dim=-1,
    )
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, M9_IMAGE_SIZE)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, M9_IMAGE_SIZE)

    boxes[..., [0, 2]] *= float(source_width) / float(M9_IMAGE_SIZE)
    boxes[..., [1, 3]] *= float(source_height) / float(M9_IMAGE_SIZE)
    return boxes


def _image_tensor(image_bgr: np.ndarray, *, device: torch.device) -> torch.Tensor:
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (M9_IMAGE_SIZE, M9_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    contiguous = np.ascontiguousarray(resized)
    return (
        torch.from_numpy(contiguous)
        .to(dtype=torch.float32)
        .permute(2, 0, 1)
        .div_(255.0)
        .unsqueeze(0)
        .to(device)
    )


class M9HierarchicalDetector:
    """Runtime adapter for the certified hierarchical RT-DETR M9 notebook model."""

    def __init__(self, config: DetectorConfig) -> None:
        if config.weights is None:
            raise ValueError("M9 hierarchical detector requires the certified checkpoint")
        if config.image_size != M9_IMAGE_SIZE:
            raise ValueError(
                f"M9 hierarchical detector requires image_size={M9_IMAGE_SIZE}; "
                f"received {config.image_size}"
            )

        payload, checkpoint_sha256 = _load_certified_checkpoint(config.weights)
        model = payload["model"]
        if model.__class__.__module__ != "ultralytics.nn.tasks" or model.__class__.__name__ != (
            "RTDETRDetectionModel"
        ):
            raise RuntimeError(
                "certified M9 checkpoint did not contain an Ultralytics RTDETRDetectionModel"
            )

        head = model.model[-1]
        required = (
            "dec_score_head",
            "coarse_dec_score_head",
            "coarse_quality_head",
            "query_film_adapter",
        )
        missing = [name for name in required if not hasattr(head, name)]
        if missing:
            raise RuntimeError(f"M9 checkpoint is missing required heads: {', '.join(missing)}")

        self._config = config
        self._device = resolve_torch_device(config.device)
        self._model = model.to(self._device)
        self._model.eval()
        self._head = head
        self._capture: dict[str, torch.Tensor] = {}
        self._hook = self._head.dec_score_head[-1].register_forward_hook(self._capture_features)
        self._name = f"m9-hierarchical-rtdetr:{Path(config.weights).stem}"
        self._metadata = DetectorMetadata(
            name=self._name,
            backend="m9-hierarchical-rtdetr",
            weights=str(config.weights),
            device=str(self._device),
            image_size=M9_IMAGE_SIZE,
            confidence=config.confidence,
            extra={
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_stage": M9_STAGE,
                "training_ultralytics": M9_TRAINING_ULTRALYTICS,
                "runtime_ultralytics": importlib.metadata.version("ultralytics"),
                "fine_candidate_threshold": M9_FINE_THRESHOLD,
                "max_predictions": M9_MAX_PREDICTIONS,
                "score_formula": {
                    "alpha": M9_ALPHA,
                    "beta": M9_BETA,
                    "gamma": M9_GAMMA,
                },
                "classes": list(M9_COARSE_NAMES),
                "legacy_pickle": True,
            },
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> DetectorMetadata:
        return self._metadata

    def _capture_features(self, _module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("M9 fine-head hook received no decoder feature tensor")
        self._capture["dec_features"] = inputs[0]

    def infer(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        source_height, source_width = image_bgr.shape[:2]
        image = _image_tensor(image_bgr, device=self._device)
        self._capture.clear()

        with torch.inference_mode():
            output = self._model(image)
            raw_boxes, raw_fine_logits = _final_decoder_outputs(output, batch_size=1)
            dec_features = self._capture.get("dec_features")
            if dec_features is None:
                raise RuntimeError("M9 decoder feature hook did not run")

            query_count = min(
                raw_boxes.shape[1],
                raw_fine_logits.shape[1],
                dec_features.shape[1],
            )
            raw_boxes = raw_boxes[:, :query_count]
            raw_fine_logits = raw_fine_logits[:, :query_count]
            dec_features = dec_features[:, :query_count]

            boxes = _boxes_to_original_xyxy(
                raw_boxes,
                source_width=source_width,
                source_height=source_height,
            )
            fine_probs = raw_fine_logits.sigmoid()
            fine_conf, fine_class = fine_probs.max(dim=-1)

            mod_features = self._head.query_film_adapter(dec_features)
            coarse_logits = self._head.coarse_dec_score_head[-1](mod_features)
            coarse_probs = F.softmax(coarse_logits, dim=-1)
            coarse_conf, coarse_class = coarse_probs.max(dim=-1)
            quality = self._head.coarse_quality_head(mod_features).squeeze(-1).sigmoid().clamp(0, 1)

            candidate = fine_conf[0] > M9_FINE_THRESHOLD
            candidate_indices = candidate.nonzero(as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                return ()

            fused = (
                fine_conf[0, candidate_indices].clamp_min(1e-8).pow(M9_ALPHA)
                * coarse_conf[0, candidate_indices].clamp_min(1e-8).pow(M9_BETA)
                * quality[0, candidate_indices].clamp_min(1e-8).pow(M9_GAMMA)
            )
            keep = fused >= float(self._config.confidence)
            candidate_indices = candidate_indices[keep]
            fused = fused[keep]
            if candidate_indices.numel() == 0:
                return ()

            if fused.numel() > M9_MAX_PREDICTIONS:
                ranking = torch.topk(fused, M9_MAX_PREDICTIONS).indices
            else:
                ranking = torch.argsort(fused, descending=True)
            candidate_indices = candidate_indices[ranking]
            fused = fused[ranking]

            detections: list[Detection] = []
            for index_tensor, score_tensor in zip(candidate_indices, fused, strict=True):
                index = int(index_tensor.item())
                coarse_id = int(coarse_class[0, index].item())
                fine_id = int(fine_class[0, index].item())
                xyxy = tuple(float(value) for value in boxes[0, index].tolist())
                detections.append(
                    Detection(
                        class_id=coarse_id,
                        source_class_id=fine_id,
                        label=M9_COARSE_NAMES[coarse_id],
                        confidence=float(score_tensor.item()),
                        xyxy=xyxy,
                        attributes={
                            "fine_class_id": fine_id,
                            "fine_confidence": float(fine_conf[0, index].item()),
                            "coarse_confidence": float(coarse_conf[0, index].item()),
                            "quality": float(quality[0, index].item()),
                        },
                    )
                )
            return tuple(detections)

    def close(self) -> None:
        hook = getattr(self, "_hook", None)
        if hook is not None:
            hook.remove()
            self._hook = None
        self._capture.clear()
        self._head = None
        self._model = None


__all__ = [
    "M9_CERTIFIED_SHA256",
    "M9_COARSE_NAMES",
    "M9HierarchicalDetector",
    "QueryFiLMAdapter",
]
