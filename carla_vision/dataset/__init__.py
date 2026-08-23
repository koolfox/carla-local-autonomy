"""Offline dataset contracts with plotting-heavy QA loaded on demand."""

from typing import Any

from .instance_labels import InstanceLabel, decode_instance_bgra, extract_instance_labels
from .ontology import (
    CARLA_SEMANTIC_TAGS,
    DETECTOR_CATEGORIES,
    DETECTOR_CATEGORY_BY_TAG,
    THING_TAGS,
    CarlaSemanticTag,
    DetectorCategory,
    detector_category_for_tag,
)
from .sync import ExactFramePairer, SynchronizedFramePair
from .verified import DatasetIntegrityError, VerifiedDataset, load_verified_dataset
from .writer import DATASET_PARTITIONS, AuxiliaryArtifact, DatasetSample, DatasetWriter

__all__ = [
    "CARLA_SEMANTIC_TAGS",
    "DETECTOR_CATEGORIES",
    "DETECTOR_CATEGORY_BY_TAG",
    "THING_TAGS",
    "CarlaSemanticTag",
    "DetectorCategory",
    "DATASET_PARTITIONS",
    "AuxiliaryArtifact",
    "DatasetSample",
    "DatasetAuditError",
    "DatasetIntegrityError",
    "DatasetWriter",
    "ExactFramePairer",
    "InstanceLabel",
    "SynchronizedFramePair",
    "VerifiedDataset",
    "decode_instance_bgra",
    "detector_category_for_tag",
    "extract_instance_labels",
    "load_verified_dataset",
    "audit_dataset",
]


def __getattr__(name: str) -> Any:
    if name in {"DatasetAuditError", "audit_dataset"}:
        from . import qa

        return getattr(qa, name)
    raise AttributeError(name)
