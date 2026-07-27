"""Built-in non-driving RGB hazard-stop baseline for shadow validation."""

from __future__ import annotations

from typing import Any

from .contracts import (
    POLICY_CONTRACT_VERSION,
    PolicyMetadata,
    VisionControlProposal,
    VisionObservation,
    VisionPolicyConfig,
)

_DEFAULT_HAZARD_LABELS = frozenset(
    {
        "bicycle",
        "bus",
        "car",
        "motorcycle",
        "person",
        "stop sign",
        "traffic light",
        "truck",
    }
)


class HazardStopShadowPolicy:
    """Propose a stop for close corridor detections; never actuate a vehicle."""

    def __init__(self, config: VisionPolicyConfig) -> None:
        options: dict[str, Any] = dict(config.options)
        self._confidence = float(options.pop("confidence", 0.35))
        self._corridor_left = float(options.pop("corridor_left", 0.30))
        self._corridor_right = float(options.pop("corridor_right", 0.70))
        self._close_bottom = float(options.pop("close_bottom", 0.72))
        self._cruise_throttle = float(options.pop("cruise_throttle", 0.15))
        labels = options.pop("hazard_labels", sorted(_DEFAULT_HAZARD_LABELS))
        if options:
            raise ValueError("unknown hazard-stop policy options: " + ", ".join(sorted(options)))
        if not 0.0 <= self._confidence <= 1.0:
            raise ValueError("hazard-stop confidence must be in [0, 1]")
        if not 0.0 <= self._corridor_left < self._corridor_right <= 1.0:
            raise ValueError("hazard-stop corridor bounds are invalid")
        if not 0.0 <= self._close_bottom <= 1.0:
            raise ValueError("hazard-stop close_bottom must be in [0, 1]")
        if not 0.0 <= self._cruise_throttle <= 1.0:
            raise ValueError("hazard-stop cruise_throttle must be in [0, 1]")
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError("hazard-stop hazard_labels must be a non-empty array")
        self._hazard_labels = frozenset(str(label).strip().casefold() for label in labels)
        if "" in self._hazard_labels:
            raise ValueError("hazard-stop labels must not be empty")
        self._metadata = PolicyMetadata(
            name="hazard-stop-shadow-v1",
            backend="builtin-hazard-stop",
            contract_version=POLICY_CONTRACT_VERSION,
            input_fields=(
                "sequence",
                "rgb_bgr",
                "detections",
            ),
            temporal=False,
            checkpoint=None,
            device=config.device,
            extra={
                "confidence": self._confidence,
                "corridor": [self._corridor_left, self._corridor_right],
                "close_bottom": self._close_bottom,
                "cruise_throttle": self._cruise_throttle,
                "hazard_labels": sorted(self._hazard_labels),
                "actuation_authorized": False,
            },
        )

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    def reset(self) -> None:
        return None

    def propose(self, observation: VisionObservation) -> VisionControlProposal:
        height, width = observation.rgb_bgr.shape[:2]
        hazards: list[dict[str, Any]] = []
        for detection in observation.detections:
            if (
                detection.label.casefold() not in self._hazard_labels
                or detection.confidence < self._confidence
            ):
                continue
            x1, _y1, x2, y2 = detection.xyxy
            center_x = ((x1 + x2) / 2.0) / width
            bottom = y2 / height
            if (
                self._corridor_left <= center_x <= self._corridor_right
                and bottom >= self._close_bottom
            ):
                hazards.append(
                    {
                        "class_id": detection.class_id,
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "center_x": center_x,
                        "bottom": bottom,
                    }
                )
        if hazards:
            confidence = max(float(item["confidence"]) for item in hazards)
            return VisionControlProposal(
                sequence=observation.sequence,
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                confidence=confidence,
                reason="close visual hazard",
                state={"hazards": hazards},
            )
        return VisionControlProposal(
            sequence=observation.sequence,
            throttle=self._cruise_throttle,
            steer=0.0,
            brake=0.0,
            confidence=0.25,
            reason="no close corridor hazard",
            state={"hazards": []},
        )

    def close(self) -> None:
        return None


__all__ = ["HazardStopShadowPolicy"]
