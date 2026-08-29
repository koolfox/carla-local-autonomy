"""Hugging Face SegFormer adapter with canonical road-class normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import (
    CANONICAL_CLASS_NAMES,
    RoadClass,
    SegmentationConfig,
    SegmentationMetadata,
    SegmentationResult,
)

_LABEL_ALIASES: dict[RoadClass, frozenset[str]] = {
    RoadClass.OTHER: frozenset({"other", "background", "unlabeled", "unlabelled", "void"}),
    RoadClass.ROAD: frozenset(
        {
            "road",
            "street",
            "road_surface",
            "drivable",
            "drivable_area",
            "asphalt",
            "construction_flat_road",
        }
    ),
    RoadClass.ROAD_LINE: frozenset(
        {
            "road_line",
            "road_lines",
            "road_marking",
            "road_markings",
            "lane_line",
            "lane_lines",
            "lane_marking",
            "lane_markings",
            "lane_divider",
            "marking_general",
            "marking_crosswalk_zebra",
            "lane_marking_general",
            "lane_marking_crosswalk",
            "lane_marking_general_crosswalk",
            "solid_line",
            "dashed_line",
            "double_solid_line",
            "zebra",
        }
    ),
    RoadClass.SIDEWALK: frozenset(
        {"sidewalk", "pavement", "footpath", "walkway", "construction_flat_sidewalk"}
    ),
    RoadClass.NON_DRIVABLE_GROUND: frozenset(
        {
            "non_drivable_ground",
            "terrain",
            "ground",
            "earth",
            "grass",
            "dirt",
            "sand",
            "nature_terrain",
        }
    ),
}


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def canonical_class_for_label(label: str) -> RoadClass:
    """Map a checkpoint label name to the stable project ontology."""

    normalized = _normalize_label(label)
    for road_class, aliases in _LABEL_ALIASES.items():
        if normalized in aliases:
            return road_class
    return RoadClass.OTHER


def _checkpoint_labels(model_config: Any) -> dict[int, str]:
    raw_id2label = getattr(model_config, "id2label", None)
    labels: dict[int, str] = {}
    if isinstance(raw_id2label, Mapping):
        for raw_id, raw_label in raw_id2label.items():
            try:
                source_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid checkpoint label id {raw_id!r}") from exc
            labels[source_id] = str(raw_label)
    if not labels:
        raw_label2id = getattr(model_config, "label2id", None)
        if isinstance(raw_label2id, Mapping):
            for raw_label, raw_id in raw_label2id.items():
                try:
                    source_id = int(raw_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid checkpoint label id {raw_id!r}") from exc
                labels[source_id] = str(raw_label)
    if not labels:
        raise ValueError(
            "SegFormer checkpoint must declare id2label or label2id for name-based mapping"
        )

    num_labels_raw = getattr(model_config, "num_labels", None)
    num_labels = int(num_labels_raw) if num_labels_raw is not None else max(labels) + 1
    if num_labels <= 0 or set(labels) != set(range(num_labels)):
        raise ValueError("SegFormer checkpoint labels must use contiguous ids from zero")

    normalized = {_normalize_label(label) for label in labels.values()}
    generic = {f"label_{index}" for index in range(num_labels)}
    if normalized == generic:
        raise ValueError(
            "checkpoint uses generic LABEL_n names; save canonical id2label metadata "
            "before inference"
        )
    if not any(canonical_class_for_label(label) is RoadClass.ROAD for label in labels.values()):
        raise ValueError("checkpoint label metadata does not contain a recognized road class")
    return dict(sorted(labels.items()))


def _load_huggingface_runtime(
    config: SegmentationConfig,
) -> tuple[Any, Any, Any]:
    """Load optional heavyweight dependencies only when this adapter is created."""

    try:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
        raise RuntimeError(
            "SegFormer inference requires PyTorch and transformers; install them in the "
            "vision environment"
        ) from exc

    checkpoint = str(config.checkpoint)
    load_options = dict(config.options)
    processor = AutoImageProcessor.from_pretrained(checkpoint, **load_options)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, **load_options)
    model.to(config.device)
    model.eval()
    return torch, processor, model


def _to_numpy(value: Any) -> np.ndarray:
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def _move_inputs(inputs: Any, device: str) -> Mapping[str, Any]:
    move_batch = getattr(inputs, "to", None)
    if callable(move_batch):
        moved = move_batch(device)
        return moved if isinstance(moved, Mapping) else inputs
    if not isinstance(inputs, Mapping):
        raise TypeError("SegFormer processor must return a mapping of model inputs")
    return {
        key: value.to(device) if callable(getattr(value, "to", None)) else value
        for key, value in inputs.items()
    }


class SegFormerSegmenter:
    """Normalize any label-aware Hugging Face SegFormer checkpoint to five classes."""

    def __init__(self, config: SegmentationConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("SegFormer segmenter requires a checkpoint")
        self._config = config
        self._torch, self._processor, self._model = _load_huggingface_runtime(config)
        self._source_labels = _checkpoint_labels(self._model.config)
        self._source_to_canonical = tuple(
            canonical_class_for_label(self._source_labels[index])
            for index in range(len(self._source_labels))
        )
        checkpoint_name = Path(str(config.checkpoint).rstrip("/")).name
        self._name = f"hf-segformer:{checkpoint_name}"
        requested_revision = config.options.get("revision")
        resolved_revision = getattr(self._model.config, "_commit_hash", None)
        self._metadata = SegmentationMetadata(
            name=self._name,
            backend="huggingface-segformer",
            checkpoint=str(config.checkpoint),
            device=config.device,
            revision=(None if requested_revision is None else str(requested_revision)),
            resolved_revision=(
                None if resolved_revision is None else str(resolved_revision)
            ),
            source_labels=self._source_labels,
            source_to_canonical={
                source_id: CANONICAL_CLASS_NAMES[int(road_class)]
                for source_id, road_class in enumerate(self._source_to_canonical)
            },
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> SegmentationMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("SegFormer segmenter is closed")
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("image_bgr must be a uint8 HxWx3 array")

        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        model_inputs = _move_inputs(inputs, self._config.device)
        with self._torch.inference_mode():
            outputs = self._model(**model_inputs)
        logits = _to_numpy(outputs.logits)
        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError("SegFormer logits must have shape 1xCxHxW")
        if logits.shape[1] != len(self._source_to_canonical):
            raise ValueError("SegFormer logits do not match checkpoint label metadata")
        if not np.all(np.isfinite(logits)):
            raise ValueError("SegFormer logits must be finite")

        source_logits = logits[0].astype(np.float32, copy=False)
        height, width = image_bgr.shape[:2]
        if source_logits.shape[1:] != (height, width):
            # Follow the SegFormer post-processing contract: interpolate logits
            # to the requested image size before softmax and class collapsing.
            source_logits = np.stack(
                [
                    cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                    for channel in source_logits
                ],
                axis=0,
            )
        source_logits -= source_logits.max(axis=0, keepdims=True)
        source_probabilities = np.exp(source_logits)
        source_probabilities /= source_probabilities.sum(axis=0, keepdims=True)

        canonical_probabilities = np.zeros(
            (len(CANONICAL_CLASS_NAMES), *source_probabilities.shape[1:]),
            dtype=np.float32,
        )
        for source_id, road_class in enumerate(self._source_to_canonical):
            canonical_probabilities[int(road_class)] += source_probabilities[source_id]

        class_ids = canonical_probabilities.argmax(axis=0).astype(np.uint8)
        # Several source classes intentionally collapse into ``other``.  Their
        # float32 sum can exceed 1.0 by a few ULPs even though the source
        # softmax is valid (observed on the real MPS Cityscapes runtime).
        # Clamp only the exported confidence; class selection remains based on
        # the unmodified canonical probabilities.
        confidence = np.clip(
            canonical_probabilities.max(axis=0),
            0.0,
            1.0,
        ).astype(np.float32)
        return SegmentationResult(
            class_ids=class_ids,
            confidence=confidence,
            model_name=self.name,
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None


__all__ = [
    "SegFormerSegmenter",
    "canonical_class_for_label",
]
