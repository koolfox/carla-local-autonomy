"""Structural audit of the policy input lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from .contracts import POLICY_CONTRACT_VERSION, VisionObservation, VisionPolicy

ALLOWED_POLICY_INPUT_FIELDS = (
    "sequence",
    "source_timestamp",
    "rgb_bgr",
    "detections",
    "camera_fov_degrees",
)
PROHIBITED_PRIVILEGED_FIELDS = (
    "actor_id",
    "carla_frame",
    "depth",
    "ego_speed",
    "gnss",
    "ground_truth",
    "imu",
    "instance_segmentation",
    "lane_invasion",
    "lidar",
    "map",
    "pose",
    "route",
    "semantic_segmentation",
    "telemetry",
    "transform",
    "velocity",
    "waypoint",
)


@dataclass(frozen=True)
class PolicyInputAudit:
    schema_version: str
    status: str
    policy_name: str
    policy_backend: str
    observation_type: str
    observation_fields: tuple[str, ...]
    declared_input_fields: tuple[str, ...]
    allowed_input_fields: tuple[str, ...]
    prohibited_privileged_fields: tuple[str, ...]
    direct_carla_objects_exposed: bool
    privileged_fields_exposed: tuple[str, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


def audit_policy(policy: VisionPolicy) -> PolicyInputAudit:
    metadata = policy.metadata
    observation_fields = tuple(field.name for field in fields(VisionObservation))
    declared = tuple(metadata.input_fields)
    allowed = set(ALLOWED_POLICY_INPUT_FIELDS)
    exposed = tuple(sorted(set(declared) - allowed))
    prohibited = tuple(
        sorted(
            {
                value
                for value in (*observation_fields, *declared)
                if value.casefold() in PROHIBITED_PRIVILEGED_FIELDS
            }
        )
    )
    if metadata.contract_version != POLICY_CONTRACT_VERSION:
        raise ValueError("vision policy declares an unsupported contract version")
    if observation_fields != ALLOWED_POLICY_INPUT_FIELDS:
        raise RuntimeError(
            "VisionObservation fields changed without updating the audited input contract"
        )
    if exposed or prohibited:
        details = ", ".join((*exposed, *prohibited))
        raise ValueError(f"vision policy requests prohibited or unknown inputs: {details}")
    return PolicyInputAudit(
        schema_version=POLICY_CONTRACT_VERSION,
        status="pass",
        policy_name=metadata.name,
        policy_backend=metadata.backend,
        observation_type="VisionObservation",
        observation_fields=observation_fields,
        declared_input_fields=declared,
        allowed_input_fields=ALLOWED_POLICY_INPUT_FIELDS,
        prohibited_privileged_fields=PROHIBITED_PRIVILEGED_FIELDS,
        direct_carla_objects_exposed=False,
        privileged_fields_exposed=(),
        limitations=(
            "Structural isolation prevents accidental privileged inputs but cannot sandbox "
            "malicious custom policy code.",
            "RGB-derived detections are allowed; their errors and latency propagate to policy.",
            "A passing input audit does not authorize vehicle actuation.",
        ),
    )


__all__ = [
    "ALLOWED_POLICY_INPUT_FIELDS",
    "PROHIBITED_PRIVILEGED_FIELDS",
    "PolicyInputAudit",
    "audit_policy",
]
