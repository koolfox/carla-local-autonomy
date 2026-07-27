"""Model-neutral CARLA vision research framework."""

from .artifacts import RunArtifactTracker, fingerprint_file
from .bridge import CarlaCameraStream, CarlaRpc, spawn_front_camera
from .contracts import Detection, Detector, DetectorConfig, PerceptionResult
from .controller import PurePursuitController
from .detectors import create_detector
from .display import DisplayMode, LiveViewer, OverlayRenderer
from .perception import PerceptionWorker
from .recording import AsyncVideoRecorder, RecordingStats
from .reproducibility import canonical_experiment_id, configuration_identity
from .risk import HazardPolicy, RiskAssessment

__all__ = [
    "AsyncVideoRecorder",
    "CarlaCameraStream",
    "CarlaRpc",
    "Detection",
    "Detector",
    "DetectorConfig",
    "DisplayMode",
    "HazardPolicy",
    "LiveViewer",
    "OverlayRenderer",
    "PerceptionResult",
    "PerceptionWorker",
    "PurePursuitController",
    "RecordingStats",
    "RiskAssessment",
    "RunArtifactTracker",
    "canonical_experiment_id",
    "configuration_identity",
    "create_detector",
    "fingerprint_file",
    "spawn_front_camera",
]
