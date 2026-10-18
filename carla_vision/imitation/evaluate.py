"""Explicit locked-test evaluation of a frozen imitation checkpoint (no CARLA)."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..reproduction.consumption import ConsumptionRecorder
from .dataset import BehaviorImitationDataset
from .model import ImitationControlNet, ImitationModelConfig
from .runner import _loader, _resolve_device, _run_epoch, _write_json


def validate_external_protocol(path, *, checkpoint, payload, data):
    """Bind a separately stored test release without admitting it to training.

    This verifies frozen identities, not when a human first inspected the test.
    Group identity quality remains the responsibility of the study split design.
    """
    protocol_ref = fingerprint_file(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {
        "schema_version",
        "checkpoint",
        "training_manifest_sha256",
        "test_manifests",
        "group_policy",
    }:
        raise ValueError("external evaluation protocol fields invalid")
    if (
        raw["schema_version"] != "1.0"
        or raw["group_policy"] != "disjoint_scene_roots_and_exact_rgb"
    ):
        raise ValueError("external evaluation group policy invalid")
    if raw["checkpoint"] != fingerprint_file(checkpoint):
        raise ValueError("external evaluation checkpoint identity mismatch")
    if sorted(raw["training_manifest_sha256"]) != sorted(
        r["sha256"] for r in payload["source_references"]
    ):
        raise ValueError("external evaluation training source mismatch")
    if raw["test_manifests"] != data.source_references:
        raise ValueError("external evaluation test source mismatch")
    split = payload["split_summary"]
    if "group_ids_by_partition" not in split or "rgb_hashes_by_partition" not in split:
        raise ValueError("checkpoint lacks the group/RGB inventory needed for external evaluation")
    groups = {g for partition in split["group_ids_by_partition"].values() for g in partition}
    hashes = {h for partition in split["rgb_hashes_by_partition"].values() for h in partition}
    if groups & {s.route_group_key for s in data.samples} or hashes & {
        s.rgb_sha256 for s in data.samples
    }:
        raise ValueError("external test overlaps model-development groups or exact RGB")
    return protocol_ref


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-inventory", type=Path)
    parser.add_argument(
        "--training-run", type=Path, help="Defaults to checkpoint parent in attested mode"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        help="Required to bind a separately stored external test release",
    )
    parser.add_argument("--acknowledge-locked-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.acknowledge_locked_test:
        parser.error("locked-test evaluation requires --acknowledge-locked-test")
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    checkpoint_ref = fingerprint_file(args.checkpoint)
    training_origin = None
    if args.sample_inventory is not None:
        from ..reproduction.consumption_verify import training_origin_reference

        training_origin = training_origin_reference(
            args.training_run or args.checkpoint.parent,
            checkpoint=checkpoint_ref,
            inventory_path=args.sample_inventory,
        )
    elif args.training_run is not None:
        raise ValueError("training origin requires --sample-inventory")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if fingerprint_file(args.checkpoint) != checkpoint_ref:
        raise ValueError("checkpoint changed while loading model")
    if payload.get("task") != "behavior_agent_control_imitation":
        raise ValueError("checkpoint task mismatch")
    if payload.get("runtime_input_contract") != "front_rgb_bgr_uint8_plus_speed_mps":
        raise ValueError("checkpoint input contract mismatch")
    config = ImitationModelConfig.from_dict(payload["model_config"])
    data = BehaviorImitationDataset(
        args.dataset,
        partition="test",
        attest_consumption=args.sample_inventory is not None,
        image_size=config.image_size_hw,
        speed_scale_mps=config.speed_scale_mps,
    )
    if not len(data):
        raise ValueError("no locked-test samples in the supplied releases")
    sources = payload.get("source_references")
    protocol_ref = None
    if not sources:
        raise ValueError("checkpoint source references missing")
    if args.evaluation_protocol is not None:
        protocol_ref = validate_external_protocol(
            args.evaluation_protocol, checkpoint=args.checkpoint, payload=payload, data=data
        )
    elif sorted(item["sha256"] for item in sources) != sorted(
        item["sha256"] for item in data.source_references
    ):
        raise ValueError("separate test releases require an explicit --evaluation-protocol")
    device = _resolve_device(args.device)
    model = ImitationControlNet(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    output = args.output.expanduser().resolve()
    with (
        RunArtifactTracker(
            output.parent,
            run_id=output.name,
            cli_args=vars(args),
            config={
                "object_type": "imitation_locked_evaluation",
                "input_contract": payload["runtime_input_contract"],
                "locked_test_acknowledged": True,
                "sample_consumption_required": args.sample_inventory is not None,
                "training_origin": training_origin,
            },
            dataset_refs=data.source_references,
            model_refs=[fingerprint_file(args.checkpoint)],
            input_refs=[r for r in (protocol_ref, training_origin) if r is not None],
        ) as tracker,
        ExitStack() as consumption_stack,
    ):
        consumption = None
        if args.sample_inventory is not None:
            consumption = consumption_stack.enter_context(
                ConsumptionRecorder(
                    tracker,
                    args.sample_inventory,
                    args.dataset,
                    phases={"test": sorted({s.source_partition for s in data.samples})},
                    task="imitation_evaluation",
                    model=fingerprint_file(args.checkpoint),
                )
            )
            tracker.add_input_reference(
                {"kind": "frozen_sample_inventory", **consumption.inventory_ref}
            )
        metrics = _run_epoch(
            model,
            _loader(data, batch_size=args.batch_size, num_workers=0, shuffle=False, seed=0),
            device=device,
            optimizer=None,
            **payload["objective_config"],
            max_batches=None,
            consumption=consumption,
            phase="test",
            epoch=0,
        )
        path = tracker.artifact_path("summary.json")
        _write_json(
            path,
            {
                "schema_version": "1.0",
                "metrics": metrics,
                "closed_loop_evaluation_completed": False,
                "metric_scope": "offline_action_prediction_not_driving_safety",
            },
        )
        tracker.register_artifact(path, role="imitation_test_summary")
        if consumption is not None:
            consumption.finish({"evaluated": fingerprint_file(args.checkpoint)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
