from __future__ import annotations

from carla_vision.voxel.actuation_gate import (
    VoxelActuationReadinessPolicy,
    evaluate_voxel_actuation_readiness,
)


def _evidence() -> dict:
    return {
        "issue_7_acceptance_complete": True,
        "operator_review_complete": True,
        "model": {
            "rgb_only_runtime_contract": True,
            "checkpoint_sha256": "a" * 64,
        },
        "training": {
            "dataset_verification_status": "passed",
            "route_group_leakage": False,
            "dataset_count": 4,
            "route_group_count": 30,
        },
        "validation": {
            "current_occupied_iou": 0.50,
            "future_occupied_iou_1s": 0.30,
            "brier_score": 0.12,
            "p95_latency_ms": 60.0,
        },
        "shadow": {
            "run_count": 4,
            "record_count": 500,
            "error_rate": 0.002,
            "p95_latency_ms": 70.0,
            "maximum_uncertain_voxel_fraction": 0.30,
            "actuation_enabled": False,
            "control_calls": 0,
        },
    }


def test_readiness_passes_complete_evidence() -> None:
    report = evaluate_voxel_actuation_readiness(_evidence())
    assert report["status"] == "passed"
    assert report["actuation_readiness"] is True
    assert report["actuation_enabled_by_this_report"] is False
    assert report["failed_checks"] == []


def test_readiness_blocks_until_issue_7_is_complete() -> None:
    evidence = _evidence()
    evidence["issue_7_acceptance_complete"] = False
    report = evaluate_voxel_actuation_readiness(evidence)
    assert report["status"] == "blocked"
    assert "issue_7_acceptance_complete" in report["failed_checks"]


def test_readiness_blocks_low_future_accuracy_and_shadow_errors() -> None:
    evidence = _evidence()
    evidence["validation"]["future_occupied_iou_1s"] = 0.05
    evidence["shadow"]["error_rate"] = 0.2
    report = evaluate_voxel_actuation_readiness(evidence)
    assert report["status"] == "blocked"
    assert "future_occupied_iou_1s" in report["failed_checks"]
    assert "shadow_error_rate" in report["failed_checks"]


def test_custom_policy_can_be_stricter() -> None:
    policy = VoxelActuationReadinessPolicy(
        minimum_current_occupied_iou=0.7,
        minimum_future_occupied_iou_1s=0.5,
    )
    report = evaluate_voxel_actuation_readiness(_evidence(), policy)
    assert report["status"] == "blocked"
    assert "current_occupied_iou" in report["failed_checks"]
