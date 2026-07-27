"""Post-hoc analysis for immutable CARLA perception runs.

The analyzer consumes one completed runtime run, validates its manifest and
``logs/detections.jsonl`` artifact, and writes a separate derived run through
``RunArtifactTracker``.  The source run is opened read-only and is fingerprinted
again before the analysis run is finalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from .artifacts import DEFAULT_PACKAGE_NAMES, SCHEMA_VERSION, RunArtifactTracker

ANALYSIS_SCHEMA_VERSION = "1.0"
DETECTIONS_RELATIVE_PATH = Path("logs/detections.jsonl")
_SHA256_HEX_LENGTH = 64


class AnalysisInputError(ValueError):
    """Raised when a source run is absent, inconsistent, or malformed."""


@dataclass(frozen=True)
class DetectionRecord:
    class_id: int
    label: str
    confidence: float
    hazard: bool


@dataclass(frozen=True)
class FrameRecord:
    line_number: int
    sequence: int
    carla_frame: int
    source_timestamp: float
    source_received_monotonic: float
    completed_monotonic: float
    pipeline_latency_seconds: float
    model_inference_seconds: float | None
    detector: str
    detections: tuple[DetectionRecord, ...]
    hazard: bool
    mode: str
    simulator_speed_mps: float
    route_progress: str

    @property
    def inference_seconds(self) -> float:
        """Backward-compatible name for end-to-end pipeline latency."""

        return self.pipeline_latency_seconds


@dataclass(frozen=True)
class FileFingerprint:
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    manifest: Mapping[str, Any]
    frames: tuple[FrameRecord, ...]
    manifest_fingerprint: FileFingerprint
    detections_fingerprint: FileFingerprint

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def reference(self) -> dict[str, Any]:
        return {
            "type": "source_run",
            "run_id": self.run_id,
            "path": self.run_dir.as_posix(),
            "status": self.manifest["status"],
            "schema_version": self.manifest["schema_version"],
            "manifest": {
                "path": "manifest.json",
                **self.manifest_fingerprint.as_dict(),
            },
            "detections": {
                "path": DETECTIONS_RELATIVE_PATH.as_posix(),
                **self.detections_fingerprint.as_dict(),
            },
        }


@dataclass(frozen=True)
class AnalysisResult:
    run_id: str
    run_dir: Path
    manifest_path: Path
    summary_path: Path
    summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir.as_posix(),
            "manifest": self.manifest_path.as_posix(),
            "summary": self.summary_path.as_posix(),
        }


def _fingerprint(path: Path) -> FileFingerprint:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise AnalysisInputError(f"could not read source artifact {path}: {error}") from error

    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AnalysisInputError(f"source artifact changed while being read: {path}")
    return FileFingerprint(sha256=digest.hexdigest(), size_bytes=after.st_size)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as error:
        raise AnalysisInputError(f"{description} is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise AnalysisInputError(
            f"{description} is not valid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise AnalysisInputError(f"could not read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{description} must contain a JSON object")
    return payload


def _require_mapping(
    value: Any,
    *,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisInputError(f"{location} must be a JSON object")
    return value


def _require_list(value: Any, *, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnalysisInputError(f"{location} must be a JSON array")
    return value


def _require_string(
    value: Any,
    *,
    location: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise AnalysisInputError(f"{location} must be {qualifier}")
    return value


def _require_boolean(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisInputError(f"{location} must be a boolean")
    return value


def _require_integer(
    value: Any,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisInputError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise AnalysisInputError(f"{location} must be at least {minimum}")
    return value


def _require_number(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisInputError(f"{location} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise AnalysisInputError(f"{location} must be a finite number")
    if minimum is not None and converted < minimum:
        raise AnalysisInputError(f"{location} must be at least {minimum}")
    if maximum is not None and converted > maximum:
        raise AnalysisInputError(f"{location} must be at most {maximum}")
    return converted


def _required(payload: Mapping[str, Any], key: str, *, location: str) -> Any:
    if key not in payload:
        raise AnalysisInputError(f"{location} is missing required field {key!r}")
    return payload[key]


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    detections_fingerprint: FileFingerprint,
) -> None:
    schema_version = _require_string(
        _required(manifest, "schema_version", location="manifest.json"),
        location="manifest.json field 'schema_version'",
    )
    if schema_version != SCHEMA_VERSION:
        raise AnalysisInputError(
            "manifest.json field 'schema_version' is unsupported: "
            f"expected {SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    _require_string(
        _required(manifest, "run_id", location="manifest.json"),
        location="manifest.json field 'run_id'",
    )
    status = _require_string(
        _required(manifest, "status", location="manifest.json"),
        location="manifest.json field 'status'",
    )
    if status not in {"success", "failed"}:
        raise AnalysisInputError(
            f"source run must be terminal before analysis; manifest.json status is {status!r}"
        )

    artifacts = _require_list(
        _required(manifest, "artifacts", location="manifest.json"),
        location="manifest.json field 'artifacts'",
    )
    matching_entries: list[Mapping[str, Any]] = []
    for index, raw_entry in enumerate(artifacts):
        entry = _require_mapping(
            raw_entry,
            location=f"manifest.json artifacts[{index}]",
        )
        path = _require_string(
            _required(entry, "path", location=f"manifest.json artifacts[{index}]"),
            location=f"manifest.json artifacts[{index}].path",
        )
        if path == DETECTIONS_RELATIVE_PATH.as_posix():
            matching_entries.append(entry)

    if not matching_entries:
        raise AnalysisInputError(
            "manifest.json does not register logs/detections.jsonl as an artifact"
        )
    if len(matching_entries) != 1:
        raise AnalysisInputError("manifest.json registers logs/detections.jsonl more than once")

    entry = matching_entries[0]
    expected_digest = _require_string(
        _required(entry, "sha256", location="detections artifact entry"),
        location="detections artifact sha256",
    ).lower()
    if len(expected_digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise AnalysisInputError(
            "detections artifact sha256 must contain 64 hexadecimal characters"
        )
    expected_size = _require_integer(
        _required(entry, "size_bytes", location="detections artifact entry"),
        location="detections artifact size_bytes",
        minimum=0,
    )
    if expected_digest != detections_fingerprint.sha256:
        raise AnalysisInputError(
            "logs/detections.jsonl checksum does not match its source manifest"
        )
    if expected_size != detections_fingerprint.size_bytes:
        raise AnalysisInputError("logs/detections.jsonl size does not match its source manifest")


def _validate_detection(value: Any, *, location: str) -> DetectionRecord:
    payload = _require_mapping(value, location=location)
    class_id = _require_integer(
        _required(payload, "class_id", location=location),
        location=f"{location} field 'class_id'",
        minimum=0,
    )
    source_class_id = _required(payload, "source_class_id", location=location)
    if source_class_id is not None:
        _require_integer(
            source_class_id,
            location=f"{location} field 'source_class_id'",
            minimum=0,
        )
    label = _require_string(
        _required(payload, "label", location=location),
        location=f"{location} field 'label'",
    )
    confidence = _require_number(
        _required(payload, "confidence", location=location),
        location=f"{location} field 'confidence'",
        minimum=0.0,
        maximum=1.0,
    )
    box = _require_list(
        _required(payload, "xyxy", location=location),
        location=f"{location} field 'xyxy'",
    )
    if len(box) != 4:
        raise AnalysisInputError(f"{location} field 'xyxy' must contain four values")
    x1, y1, x2, y2 = (
        _require_number(item, location=f"{location} field 'xyxy'[{index}]")
        for index, item in enumerate(box)
    )
    if x2 < x1 or y2 < y1:
        raise AnalysisInputError(f"{location} field 'xyxy' must have non-negative width and height")
    _require_mapping(
        _required(payload, "attributes", location=location),
        location=f"{location} field 'attributes'",
    )
    risk = _require_mapping(
        _required(payload, "risk", location=location),
        location=f"{location} field 'risk'",
    )
    _require_boolean(
        _required(risk, "in_driving_corridor", location=f"{location} risk"),
        location=f"{location} risk.in_driving_corridor",
    )
    _require_boolean(
        _required(risk, "visually_close", location=f"{location} risk"),
        location=f"{location} risk.visually_close",
    )
    hazard = _require_boolean(
        _required(risk, "hazard", location=f"{location} risk"),
        location=f"{location} risk.hazard",
    )
    _require_number(
        _required(risk, "confidence_threshold", location=f"{location} risk"),
        location=f"{location} risk.confidence_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    return DetectionRecord(
        class_id=class_id,
        label=label,
        confidence=confidence,
        hazard=hazard,
    )


def _validate_frame(value: Any, *, line_number: int) -> FrameRecord:
    location = f"logs/detections.jsonl line {line_number}"
    payload = _require_mapping(value, location=location)
    sequence = _require_integer(
        _required(payload, "sequence", location=location),
        location=f"{location} field 'sequence'",
        minimum=0,
    )
    carla_frame = _require_integer(
        _required(payload, "carla_frame", location=location),
        location=f"{location} field 'carla_frame'",
        minimum=0,
    )
    source_timestamp = _require_number(
        _required(payload, "source_timestamp", location=location),
        location=f"{location} field 'source_timestamp'",
        minimum=0.0,
    )
    received = _require_number(
        _required(payload, "source_received_monotonic", location=location),
        location=f"{location} field 'source_received_monotonic'",
        minimum=0.0,
    )
    completed = _require_number(
        _required(payload, "completed_monotonic", location=location),
        location=f"{location} field 'completed_monotonic'",
        minimum=0.0,
    )
    if completed < received:
        raise AnalysisInputError(
            f"{location}: completed_monotonic cannot precede source_received_monotonic"
        )
    pipeline_field = (
        "pipeline_latency_seconds" if "pipeline_latency_seconds" in payload else "inference_seconds"
    )
    pipeline_latency_seconds = _require_number(
        _required(payload, pipeline_field, location=location),
        location=f"{location} field {pipeline_field!r}",
        minimum=0.0,
    )
    measured_inference = completed - received
    if not math.isclose(
        pipeline_latency_seconds,
        measured_inference,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise AnalysisInputError(
            f"{location}: {pipeline_field} is inconsistent with its monotonic timestamps"
        )
    raw_model_inference = payload.get("model_inference_seconds")
    model_inference_seconds = (
        None
        if raw_model_inference is None
        else _require_number(
            raw_model_inference,
            location=f"{location} field 'model_inference_seconds'",
            minimum=0.0,
        )
    )
    if (
        model_inference_seconds is not None
        and model_inference_seconds > pipeline_latency_seconds + 1e-6
    ):
        raise AnalysisInputError(
            f"{location}: model_inference_seconds cannot exceed pipeline latency"
        )
    detector = _require_string(
        _required(payload, "detector", location=location),
        location=f"{location} field 'detector'",
    )
    raw_detections = _require_list(
        _required(payload, "detections", location=location),
        location=f"{location} field 'detections'",
    )
    detections = tuple(
        _validate_detection(
            raw_detection,
            location=f"{location} detections[{index}]",
        )
        for index, raw_detection in enumerate(raw_detections)
    )
    hazard = _require_boolean(
        _required(payload, "hazard", location=location),
        location=f"{location} field 'hazard'",
    )
    if hazard != any(detection.hazard for detection in detections):
        raise AnalysisInputError(
            f"{location}: frame hazard disagrees with per-detection risk flags"
        )
    mode = _require_string(
        _required(payload, "mode", location=location),
        location=f"{location} field 'mode'",
    )
    speed = _require_number(
        _required(payload, "simulator_speed_mps", location=location),
        location=f"{location} field 'simulator_speed_mps'",
    )
    route_progress = _require_string(
        _required(payload, "route_progress", location=location),
        location=f"{location} field 'route_progress'",
        allow_empty=True,
    )
    return FrameRecord(
        line_number=line_number,
        sequence=sequence,
        carla_frame=carla_frame,
        source_timestamp=source_timestamp,
        source_received_monotonic=received,
        completed_monotonic=completed,
        pipeline_latency_seconds=pipeline_latency_seconds,
        model_inference_seconds=model_inference_seconds,
        detector=detector,
        detections=detections,
        hazard=hazard,
        mode=mode,
        simulator_speed_mps=speed,
        route_progress=route_progress,
    )


def _load_frames(path: Path) -> tuple[FrameRecord, ...]:
    frames: list[FrameRecord] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise AnalysisInputError(f"logs/detections.jsonl line {line_number} is blank")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AnalysisInputError(
                        f"logs/detections.jsonl line {line_number} is invalid JSON "
                        f"at column {error.colno}: {error.msg}"
                    ) from error
                frame = _validate_frame(payload, line_number=line_number)
                if frames:
                    previous = frames[-1]
                    if frame.sequence <= previous.sequence:
                        raise AnalysisInputError(
                            f"logs/detections.jsonl line {line_number}: sequence must "
                            "increase strictly"
                        )
                    if frame.carla_frame <= previous.carla_frame:
                        raise AnalysisInputError(
                            f"logs/detections.jsonl line {line_number}: carla_frame "
                            "must increase strictly"
                        )
                    if frame.source_timestamp <= previous.source_timestamp:
                        raise AnalysisInputError(
                            f"logs/detections.jsonl line {line_number}: source_timestamp "
                            "must increase strictly"
                        )
                    if frame.completed_monotonic <= previous.completed_monotonic:
                        raise AnalysisInputError(
                            f"logs/detections.jsonl line {line_number}: "
                            "completed_monotonic must increase strictly"
                        )
                frames.append(frame)
    except AnalysisInputError:
        raise
    except FileNotFoundError as error:
        raise AnalysisInputError(f"detections log is missing: {path}") from error
    except (OSError, UnicodeError) as error:
        raise AnalysisInputError(f"could not read detections log {path}: {error}") from error

    if not frames:
        raise AnalysisInputError("logs/detections.jsonl contains no frame records")
    return tuple(frames)


def load_source_run(source_run: str | Path) -> SourceRun:
    """Load and fully validate one immutable runtime run."""

    raw_path = Path(source_run).expanduser()
    try:
        run_dir = raw_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AnalysisInputError(f"source run does not exist: {raw_path}") from error
    if not run_dir.is_dir():
        raise AnalysisInputError(f"source run is not a directory: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    detections_path = run_dir / DETECTIONS_RELATIVE_PATH
    manifest_before = _fingerprint(manifest_path)
    detections_before = _fingerprint(detections_path)
    manifest = _load_json_object(manifest_path, description="source manifest")
    frames = _load_frames(detections_path)
    manifest_after = _fingerprint(manifest_path)
    detections_after = _fingerprint(detections_path)
    if manifest_before != manifest_after or detections_before != detections_after:
        raise AnalysisInputError("source run changed while it was being validated")
    _validate_manifest(manifest, detections_fingerprint=detections_after)
    return SourceRun(
        run_dir=run_dir,
        manifest=manifest,
        frames=frames,
        manifest_fingerprint=manifest_after,
        detections_fingerprint=detections_after,
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _descriptive_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
            "stddev": None,
        }
    converted = [float(value) for value in values]
    return {
        "count": len(converted),
        "min": min(converted),
        "max": max(converted),
        "mean": statistics.fmean(converted),
        "median": statistics.median(converted),
        "p95": _percentile(converted, 0.95),
        "stddev": statistics.pstdev(converted),
    }


def _successive_differences(values: Sequence[float]) -> list[float]:
    return [
        float(current) - float(previous)
        for previous, current in zip(values, values[1:], strict=False)
    ]


def _estimate_distance(frames: Sequence[FrameRecord]) -> float:
    distance = 0.0
    for previous, current in zip(frames, frames[1:], strict=False):
        delta = current.source_timestamp - previous.source_timestamp
        distance += (
            (abs(previous.simulator_speed_mps) + abs(current.simulator_speed_mps)) * 0.5 * delta
        )
    return distance


def compute_summary(
    frames: Sequence[FrameRecord],
    *,
    analysis_run_id: str,
    source_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute JSON-safe aggregate metrics from validated frame records."""

    pipeline_latency_ms = [frame.pipeline_latency_seconds * 1000.0 for frame in frames]
    model_inference_ms = [
        frame.model_inference_seconds * 1000.0
        for frame in frames
        if frame.model_inference_seconds is not None
    ]
    source_timestamps = [frame.source_timestamp for frame in frames]
    completion_timestamps = [frame.completed_monotonic for frame in frames]
    source_cadence_ms = [
        difference * 1000.0 for difference in _successive_differences(source_timestamps)
    ]
    completion_cadence_ms = [
        difference * 1000.0 for difference in _successive_differences(completion_timestamps)
    ]
    speeds = [frame.simulator_speed_mps for frame in frames]
    absolute_speeds = [abs(speed) for speed in speeds]
    detection_counts = [len(frame.detections) for frame in frames]
    all_detections = [detection for frame in frames for detection in frame.detections]
    hazard_frame_count = sum(frame.hazard for frame in frames)
    hazard_detection_count = sum(detection.hazard for detection in all_detections)
    class_frequency = Counter(detection.label for detection in all_detections)
    class_id_frequency = Counter(
        (detection.class_id, detection.label) for detection in all_detections
    )
    detector_frequency = Counter(frame.detector for frame in frames)
    mode_frequency = Counter(frame.mode for frame in frames)
    simulator_duration = source_timestamps[-1] - source_timestamps[0]

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "post_hoc_detection_run",
        "analysis_run_id": analysis_run_id,
        "source_run": dict(source_reference),
        "metrics": {
            "frames": {
                "count": len(frames),
                "first_sequence": frames[0].sequence,
                "last_sequence": frames[-1].sequence,
                "first_carla_frame": frames[0].carla_frame,
                "last_carla_frame": frames[-1].carla_frame,
                "simulator_duration_seconds": simulator_duration,
                "effective_source_fps": (
                    (len(frames) - 1) / simulator_duration
                    if len(frames) > 1 and simulator_duration > 0.0
                    else None
                ),
            },
            "timing": {
                "pipeline_latency_ms": _descriptive_statistics(pipeline_latency_ms),
                "model_inference_ms": _descriptive_statistics(model_inference_ms),
                # Retained so analyses of pre-v0.2 runs remain machine-readable.
                "inference_ms": _descriptive_statistics(pipeline_latency_ms),
                "source_cadence_ms": _descriptive_statistics(source_cadence_ms),
                "completion_cadence_ms": _descriptive_statistics(completion_cadence_ms),
            },
            "simulator_speed_mps": {
                "signed": _descriptive_statistics(speeds),
                "absolute": _descriptive_statistics(absolute_speeds),
                "estimated_distance_metres": _estimate_distance(frames),
            },
            "detections": {
                "total": len(all_detections),
                "frames_with_detections": sum(count > 0 for count in detection_counts),
                "per_frame": _descriptive_statistics(detection_counts),
                "hazard_frame_count": hazard_frame_count,
                "hazard_frame_rate": hazard_frame_count / len(frames),
                "hazard_detection_count": hazard_detection_count,
                "confidence": _descriptive_statistics(
                    [detection.confidence for detection in all_detections]
                ),
                "class_frequency": dict(sorted(class_frequency.items())),
                "class_id_frequency": [
                    {
                        "class_id": class_id,
                        "label": label,
                        "count": count,
                    }
                    for (class_id, label), count in sorted(
                        class_id_frequency.items(),
                        key=lambda item: (item[0][0], item[0][1]),
                    )
                ],
            },
            "detector_frequency": dict(sorted(detector_frequency.items())),
            "mode_frequency": dict(sorted(mode_frequency.items())),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _write_frames_csv(path: Path, frames: Sequence[FrameRecord]) -> None:
    fieldnames = (
        "line_number",
        "sequence",
        "carla_frame",
        "source_timestamp",
        "elapsed_source_seconds",
        "source_received_monotonic",
        "completed_monotonic",
        "inference_ms",
        "pipeline_latency_ms",
        "model_inference_ms",
        "source_cadence_ms",
        "completion_cadence_ms",
        "simulator_speed_mps",
        "detection_count",
        "hazard_detection_count",
        "hazard",
        "detector",
        "mode",
        "route_progress",
        "class_counts_json",
    )
    first_timestamp = frames[0].source_timestamp
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        previous: FrameRecord | None = None
        for frame in frames:
            class_counts = Counter(detection.label for detection in frame.detections)
            writer.writerow(
                {
                    "line_number": frame.line_number,
                    "sequence": frame.sequence,
                    "carla_frame": frame.carla_frame,
                    "source_timestamp": frame.source_timestamp,
                    "elapsed_source_seconds": frame.source_timestamp - first_timestamp,
                    "source_received_monotonic": frame.source_received_monotonic,
                    "completed_monotonic": frame.completed_monotonic,
                    "inference_ms": frame.pipeline_latency_seconds * 1000.0,
                    "pipeline_latency_ms": (frame.pipeline_latency_seconds * 1000.0),
                    "model_inference_ms": (
                        ""
                        if frame.model_inference_seconds is None
                        else frame.model_inference_seconds * 1000.0
                    ),
                    "source_cadence_ms": (
                        ""
                        if previous is None
                        else (frame.source_timestamp - previous.source_timestamp) * 1000.0
                    ),
                    "completion_cadence_ms": (
                        ""
                        if previous is None
                        else (frame.completed_monotonic - previous.completed_monotonic) * 1000.0
                    ),
                    "simulator_speed_mps": frame.simulator_speed_mps,
                    "detection_count": len(frame.detections),
                    "hazard_detection_count": sum(
                        detection.hazard for detection in frame.detections
                    ),
                    "hazard": str(frame.hazard).lower(),
                    "detector": frame.detector,
                    "mode": frame.mode,
                    "route_progress": frame.route_progress,
                    "class_counts_json": json.dumps(
                        dict(sorted(class_counts.items())),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            previous = frame


def _elapsed_times(frames: Sequence[FrameRecord]) -> list[float]:
    origin = frames[0].source_timestamp
    return [frame.source_timestamp - origin for frame in frames]


def _save_figure(figure: Any, path: Path) -> None:
    try:
        figure.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
            metadata={"Software": "carla_vision.analysis"},
        )
    finally:
        plt.close(figure)


def _plot_latency_cadence(path: Path, frames: Sequence[FrameRecord]) -> None:
    elapsed = _elapsed_times(frames)
    inference = [frame.inference_seconds * 1000.0 for frame in frames]
    source_cadence = [
        (current.source_timestamp - previous.source_timestamp) * 1000.0
        for previous, current in zip(frames, frames[1:], strict=False)
    ]
    completion_cadence = [
        (current.completed_monotonic - previous.completed_monotonic) * 1000.0
        for previous, current in zip(frames, frames[1:], strict=False)
    ]

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(elapsed, inference, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Inference latency (ms)")
    axes[0].set_title("Detector latency and frame cadence")
    axes[0].grid(alpha=0.25)
    if source_cadence:
        cadence_x = elapsed[1:]
        axes[1].plot(
            cadence_x,
            source_cadence,
            label="CARLA source cadence",
            color="#2ca02c",
            linewidth=1.4,
        )
        axes[1].plot(
            cadence_x,
            completion_cadence,
            label="Completed-result cadence",
            color="#ff7f0e",
            linewidth=1.4,
        )
        axes[1].legend(loc="best")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Cadence requires at least two frames",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("CARLA elapsed time (s)")
    axes[1].set_ylabel("Cadence (ms)")
    axes[1].grid(alpha=0.25)
    _save_figure(figure, path)


def _plot_simulator_speed(path: Path, frames: Sequence[FrameRecord]) -> None:
    elapsed = _elapsed_times(frames)
    speeds = [frame.simulator_speed_mps for frame in frames]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(elapsed, speeds, color="#9467bd", linewidth=1.6)
    axis.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    axis.set(
        title="Simulator speed",
        xlabel="CARLA elapsed time (s)",
        ylabel="Speed (m/s)",
    )
    axis.grid(alpha=0.25)
    _save_figure(figure, path)


def _plot_detections_hazards(path: Path, frames: Sequence[FrameRecord]) -> None:
    elapsed = _elapsed_times(frames)
    counts = [len(frame.detections) for frame in frames]
    hazard_x = [
        time_value for time_value, frame in zip(elapsed, frames, strict=True) if frame.hazard
    ]
    hazard_y = [count for count, frame in zip(counts, frames, strict=True) if frame.hazard]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.step(
        elapsed,
        counts,
        where="mid",
        color="#1f77b4",
        linewidth=1.5,
        label="Detection count",
    )
    if hazard_x:
        axis.scatter(
            hazard_x,
            hazard_y,
            color="#d62728",
            marker="x",
            s=45,
            linewidths=1.8,
            label="Hazard frame",
            zorder=3,
        )
    axis.set(
        title="Per-frame detections and visual hazards",
        xlabel="CARLA elapsed time (s)",
        ylabel="Detections",
    )
    axis.set_ylim(bottom=-0.1)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    _save_figure(figure, path)


def _plot_class_frequency(path: Path, frames: Sequence[FrameRecord]) -> None:
    counts = Counter(detection.label for frame in frames for detection in frame.detections)
    figure, axis = plt.subplots(figsize=(10, max(4.5, min(12.0, 0.42 * max(1, len(counts)) + 2.0))))
    if counts:
        ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
        labels = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        axis.barh(labels, values, color="#17becf")
        axis.set_xlabel("Detection instances")
    else:
        axis.text(
            0.5,
            0.5,
            "No detections in this run",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title("Detected class frequency")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(figure, path)


def _verify_source_unchanged(source: SourceRun) -> None:
    current_manifest = _fingerprint(source.run_dir / "manifest.json")
    current_detections = _fingerprint(source.run_dir / DETECTIONS_RELATIVE_PATH)
    if (
        current_manifest != source.manifest_fingerprint
        or current_detections != source.detections_fingerprint
    ):
        raise AnalysisInputError("source run changed during post-hoc analysis")


def analyze_run(
    source_run: str | Path,
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> AnalysisResult:
    """Analyze a completed source run and return its derived run identity."""

    source = load_source_run(source_run)
    output_root = Path(runs_root).expanduser().resolve(strict=False)
    try:
        output_root.relative_to(source.run_dir)
    except ValueError:
        pass
    else:
        raise AnalysisInputError("runs_root must not be the source run or a directory inside it")

    source_carla = _require_mapping(
        _required(source.manifest, "carla", location="manifest.json"),
        location="manifest.json field 'carla'",
    )
    project_root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    tracker = RunArtifactTracker(
        output_root,
        run_id=run_id,
        cli_args=cli_args,
        config={
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_type": "post_hoc_detection_run",
            "source_run": source.reference,
        },
        repository_root=project_root,
        carla_endpoint=source_carla.get("endpoint"),
        carla_version=source_carla.get("version"),
        carla_map=source_carla.get("map"),
        dataset_refs=(source.reference,),
        package_names=tuple(sorted(set(DEFAULT_PACKAGE_NAMES).union({"matplotlib"}))),
    )

    summary: dict[str, Any]
    summary_path: Path
    with tracker:
        summary_path = tracker.artifact_path("summary_metrics.json")
        frames_path = tracker.artifact_path("frames.csv")
        latency_path = tracker.artifact_path("plots/latency_cadence.png")
        speed_path = tracker.artifact_path("plots/simulator_speed.png")
        detections_path = tracker.artifact_path("plots/detections_hazards.png")
        classes_path = tracker.artifact_path("plots/class_frequency.png")

        summary = compute_summary(
            source.frames,
            analysis_run_id=tracker.run_id,
            source_reference=source.reference,
        )
        _write_json(summary_path, summary)
        _write_frames_csv(frames_path, source.frames)
        _plot_latency_cadence(latency_path, source.frames)
        _plot_simulator_speed(speed_path, source.frames)
        _plot_detections_hazards(detections_path, source.frames)
        _plot_class_frequency(classes_path, source.frames)

        common_metadata = {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "source_run_id": source.run_id,
            "source_detections_sha256": source.detections_fingerprint.sha256,
        }
        tracker.register_artifact(
            summary_path,
            role="analysis_summary_metrics",
            metadata=common_metadata,
        )
        tracker.register_artifact(
            frames_path,
            role="analysis_frame_table",
            metadata={
                **common_metadata,
                "rows": len(source.frames),
            },
        )
        tracker.register_artifact(
            latency_path,
            role="analysis_plot_latency_cadence",
            metadata=common_metadata,
        )
        tracker.register_artifact(
            speed_path,
            role="analysis_plot_simulator_speed",
            metadata=common_metadata,
        )
        tracker.register_artifact(
            detections_path,
            role="analysis_plot_detections_hazards",
            metadata=common_metadata,
        )
        tracker.register_artifact(
            classes_path,
            role="analysis_plot_class_frequency",
            metadata=common_metadata,
        )
        _verify_source_unchanged(source)

    return AnalysisResult(
        run_id=tracker.run_id,
        run_dir=tracker.run_dir,
        manifest_path=tracker.manifest_path,
        summary_path=summary_path,
        summary=summary,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze a CARLA perception run into a separate "
            "checksum-tracked derived run"
        )
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="completed source run containing manifest.json and logs/detections.jsonl",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="root directory in which the derived analysis run is created",
    )
    parser.add_argument(
        "--run-id",
        help="optional explicit derived run id",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    invocation = tuple(sys.argv[1:] if argv is None else argv)
    args = parse_args(invocation)
    try:
        result = analyze_run(
            args.source_run,
            runs_root=args.runs_root,
            run_id=args.run_id,
            cli_args=invocation,
        )
    except (AnalysisInputError, FileExistsError, OSError, ValueError) as error:
        print(f"analysis error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AnalysisInputError",
    "AnalysisResult",
    "FrameRecord",
    "analyze_run",
    "compute_summary",
    "load_source_run",
    "main",
    "parse_args",
]
