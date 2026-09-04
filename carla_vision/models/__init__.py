"""Model execution adapters for research policies and detectors.

This package intentionally keeps model loading separate from CARLA ownership.
A model receives explicit observations and returns explicit outputs.
"""

from .adapter import ModelAdapter, ModelMetadata, TorchModelAdapter

__all__ = ["ModelAdapter", "ModelMetadata", "TorchModelAdapter"]
