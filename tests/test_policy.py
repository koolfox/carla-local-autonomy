from __future__ import annotations

import hashlib
import unittest

import numpy as np

from carla_vision.contracts import Detection, PerceptionResult
from carla_vision.policy import (
    POLICY_CONTRACT_VERSION,
    PolicyMetadata,
    ShadowPolicyRunner,
    VisionControlProposal,
    VisionObservation,
    VisionPolicyConfig,
    audit_policy,
    create_vision_policy,
)


def _result(
    *,
    detections: tuple[Detection, ...] = (),
    sequence: int = 4,
) -> PerceptionResult:
    image = np.full((100, 200, 3), 17, dtype=np.uint8)
    return PerceptionResult(
        sequence=sequence,
        carla_frame=1234,
        source_timestamp=5.5,
        source_received_monotonic=10.0,
        inference_started_monotonic=10.1,
        completed_monotonic=10.2,
        detections=detections,
        source_bgr=image,
        detector_name="fake",
        source_transform=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        source_fov=90.0,
    )


class PolicyContractsTests(unittest.TestCase):
    def test_observation_copies_only_approved_rgb_lane_and_is_read_only(self) -> None:
        result = _result()
        observation = VisionObservation.from_perception(result)

        self.assertEqual(
            tuple(observation.__dataclass_fields__),
            (
                "sequence",
                "source_timestamp",
                "rgb_bgr",
                "detections",
                "camera_fov_degrees",
            ),
        )
        self.assertFalse(hasattr(observation, "source_transform"))
        self.assertFalse(hasattr(observation, "carla_frame"))
        self.assertFalse(np.shares_memory(observation.rgb_bgr, result.source_bgr))
        self.assertFalse(observation.rgb_bgr.flags.writeable)
        result.source_bgr[:] = 99
        self.assertTrue(np.all(observation.rgb_bgr == 17))
        with self.assertRaises(ValueError):
            observation.rgb_bgr[0, 0, 0] = 0

    def test_proposal_rejects_conflicting_throttle_and_brake(self) -> None:
        with self.assertRaisesRegex(ValueError, "throttle and brake"):
            VisionControlProposal(
                sequence=1,
                throttle=0.5,
                steer=0.0,
                brake=0.5,
                confidence=1.0,
                reason="invalid",
            )


class BuiltinShadowPolicyTests(unittest.TestCase):
    def test_hazard_stop_policy_brakes_only_for_close_corridor_objects(self) -> None:
        policy = create_vision_policy(VisionPolicyConfig(backend="hazard-stop"))
        audit = audit_policy(policy)
        self.assertEqual(audit.status, "pass")
        self.assertFalse(audit.direct_carla_objects_exposed)

        close_person = Detection(
            class_id=0,
            label="person",
            confidence=0.9,
            xyxy=(80.0, 20.0, 120.0, 90.0),
        )
        proposal = policy.propose(
            VisionObservation.from_perception(_result(detections=(close_person,)))
        )
        self.assertEqual(proposal.brake, 1.0)
        self.assertEqual(proposal.throttle, 0.0)
        self.assertEqual(proposal.reason, "close visual hazard")

        side_person = Detection(
            class_id=0,
            label="person",
            confidence=0.9,
            xyxy=(0.0, 20.0, 20.0, 90.0),
        )
        proposal = policy.propose(
            VisionObservation.from_perception(_result(detections=(side_person,)))
        )
        self.assertEqual(proposal.brake, 0.0)
        self.assertGreater(proposal.throttle, 0.0)

    def test_shadow_runner_records_exact_rgb_and_never_applies_actuation(self) -> None:
        policy = create_vision_policy(VisionPolicyConfig(backend="hazard-stop"))
        result = _result(sequence=9)
        with ShadowPolicyRunner(policy) as runner:
            proposal, record = runner.propose(result)
            stats = runner.stats()

        self.assertEqual(proposal.sequence, 9)
        self.assertEqual(
            record["rgb_sha256"],
            hashlib.sha256(result.source_bgr.tobytes()).hexdigest(),
        )
        self.assertFalse(record["actuation_applied"])
        self.assertNotIn("source_transform", record)
        self.assertNotIn("simulator_speed", record)
        self.assertEqual(stats.proposal_count, 1)
        self.assertEqual(stats.throttle_proposal_count, 1)

    def test_audit_rejects_policy_declaring_privileged_input(self) -> None:
        class InvalidPolicy:
            metadata = PolicyMetadata(
                name="invalid",
                backend="test",
                contract_version=POLICY_CONTRACT_VERSION,
                input_fields=("rgb_bgr", "ego_speed"),
                temporal=False,
            )

            def reset(self) -> None:
                pass

            def propose(self, observation: VisionObservation) -> VisionControlProposal:
                return VisionControlProposal(
                    sequence=observation.sequence,
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    confidence=1.0,
                    reason="test",
                )

            def close(self) -> None:
                pass

        with self.assertRaisesRegex(ValueError, "prohibited"):
            audit_policy(InvalidPolicy())


if __name__ == "__main__":
    unittest.main()
