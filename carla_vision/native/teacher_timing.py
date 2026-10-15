"""Decision-target timing, deliberately distinct from transition-associated controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationActionTransition:
    observation_frame: int
    decision_frame: int
    action_applies_before_frame: int
    teacher_id: str
    resulting_frame: int | None = None
    schema_version: str = "1.0"
    target_semantics: str = "decision_from_observed_state"

    def __post_init__(self) -> None:
        for name in ("observation_frame", "decision_frame", "action_applies_before_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.schema_version != "1.0" or self.target_semantics != "decision_from_observed_state":
            raise ValueError("unsupported decision-target timing contract")
        if self.decision_frame != self.observation_frame:
            raise ValueError("decision must use the same state frame as the student observation")
        if self.action_applies_before_frame != self.decision_frame + 1:
            raise ValueError("action must be issued before the next world tick")
        if self.resulting_frame is not None and (
            isinstance(self.resulting_frame, bool)
            or not isinstance(self.resulting_frame, int)
            or self.resulting_frame != self.action_applies_before_frame
        ):
            raise ValueError("resulting frame is inconsistent with the action interval")
        if not isinstance(self.teacher_id, str) or not self.teacher_id.strip():
            raise ValueError("teacher identity is required")

    def as_dict(self) -> dict[str, Any]:
        # None explicitly means the future state has not been observed at sample-write time.
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ObservationActionTransition:
        return cls(**dict(raw))


def validate_decision_target(context: Mapping[str, Any], frame: int) -> None:
    raw = context.get("teacher_transition")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "teacher decision timing is unknown; legacy targets require explicit development opt-in"
        )
    transition = ObservationActionTransition.from_mapping(raw)
    if transition.observation_frame != frame:
        raise ValueError("teacher observation frame does not match the RGB sample")
    control = context.get("privileged_teacher_control")
    if not isinstance(control, Mapping) or control.get("carla_frame") != frame:
        raise ValueError("teacher decision control frame does not match its observation")
