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
from .overlay import ROAD_CLASS_COLORS_BGR, render_segmentation_overlay
from .segformer import SegFormerSegmenter, canonical_class_for_label
from .worker import (
    AsyncSegmentationRuntime,
    SegmentationFrameInput,
    SegmentationFrameResult,
    SegmentationWorker,
    SegmentationWorkerStats,
)

__all__ = [
    "CANONICAL_CLASS_NAMES",
    "AsyncSegmentationRuntime",
    "DEFAULT_SEGFORMER_B0_CHECKPOINT",
    "RoadClass",
    "RoadSegmenter",
    "ROAD_CLASS_COLORS_BGR",
    "SegFormerSegmenter",
    "SegmentationFrameInput",
    "SegmentationFrameResult",
    "SegmentationConfig",
    "SegmentationMetadata",
    "SegmentationResult",
    "SegmentationWorker",
    "SegmentationWorkerStats",
    "canonical_class_for_label",
    "create_segmenter",
    "render_segmentation_overlay",
]
