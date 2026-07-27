"""Non-actuating execution wrapper for audited vision policies."""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

from ..contracts import PerceptionResult
from .audit import PolicyInputAudit, audit_policy
from .contracts import VisionControlProposal, VisionObservation, VisionPolicy


@dataclass(frozen=True)
class ShadowPolicyStats:
    proposal_count: int
    braking_proposal_count: int
    throttle_proposal_count: int
    latency_mean_ms: float | None
    latency_median_ms: float | None
    latency_p95_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_count": self.proposal_count,
            "braking_proposal_count": self.braking_proposal_count,
            "throttle_proposal_count": self.throttle_proposal_count,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class ShadowPolicyRunner:
    """Run a policy on an isolated RGB observation and retain proposal evidence."""

    def __init__(self, policy: VisionPolicy) -> None:
        self.policy = policy
        self.audit: PolicyInputAudit = audit_policy(policy)
        self._closed = False
        self._latencies_ms: list[float] = []
        self._braking = 0
        self._throttle = 0
        self.policy.reset()

    def propose(
        self,
        result: PerceptionResult,
    ) -> tuple[VisionControlProposal, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("shadow policy runner is closed")
        observation = VisionObservation.from_perception(result)
        started = time.monotonic()
        proposal = self.policy.propose(observation)
        completed = time.monotonic()
        if not isinstance(proposal, VisionControlProposal):
            raise TypeError("vision policy must return VisionControlProposal")
        if proposal.sequence != observation.sequence:
            raise RuntimeError("vision policy proposal sequence differs from observation")
        latency_ms = (completed - started) * 1000.0
        self._latencies_ms.append(latency_ms)
        if proposal.brake > 0.05:
            self._braking += 1
        if proposal.throttle > 0.05:
            self._throttle += 1
        record = {
            "schema_version": "1.0",
            "sequence": observation.sequence,
            "source_timestamp": observation.source_timestamp,
            "rgb_sha256": hashlib.sha256(observation.rgb_bgr.tobytes()).hexdigest(),
            "image_width": int(observation.rgb_bgr.shape[1]),
            "image_height": int(observation.rgb_bgr.shape[0]),
            "detection_count": len(observation.detections),
            "camera_fov_degrees": observation.camera_fov_degrees,
            "policy": self.policy.metadata.as_dict(),
            "input_contract": list(self.audit.declared_input_fields),
            "proposal": proposal.as_dict(),
            "policy_latency_ms": latency_ms,
            "actuation_applied": False,
        }
        return proposal, record

    def stats(self) -> ShadowPolicyStats:
        latencies = self._latencies_ms
        return ShadowPolicyStats(
            proposal_count=len(latencies),
            braking_proposal_count=self._braking,
            throttle_proposal_count=self._throttle,
            latency_mean_ms=statistics.fmean(latencies) if latencies else None,
            latency_median_ms=statistics.median(latencies) if latencies else None,
            latency_p95_ms=_percentile(latencies, 0.95) if latencies else None,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.policy.close()

    def __enter__(self) -> "ShadowPolicyRunner":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["ShadowPolicyRunner", "ShadowPolicyStats"]
