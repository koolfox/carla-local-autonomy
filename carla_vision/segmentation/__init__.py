"""RGB road/lane semantic-segmentation adapters and canonical contracts."""

from .contracts import (
    CANONICAL_CLASS_NAMES,
    DEFAULT_SEGFORMER_B0_CHECKPOINT,
    RoadClass,
    RoadSegmenter,
    SegmentationConfig,
    SegmentationMetadata,
    SegmentationResult,
)
from .factory import create_segmenter
from .segformer import SegFormerSegmenter, canonical_class_for_label

__all__ = [
    "CANONICAL_CLASS_NAMES",
    "DEFAULT_SEGFORMER_B0_CHECKPOINT",
    "RoadClass",
    "RoadSegmenter",
    "SegFormerSegmenter",
    "SegmentationConfig",
    "SegmentationMetadata",
    "SegmentationResult",
    "canonical_class_for_label",
    "create_segmenter",
]
