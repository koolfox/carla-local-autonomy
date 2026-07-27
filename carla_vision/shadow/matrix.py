"""Plan or execute a safe, non-actuating live vision-shadow matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..model_release.verified import load_verified_model
from ..policy.verified import load_verified_vision_shadow_run
from ..reproducibility import canonical_json_bytes
from .contracts import (
    ShadowMatrixCell,
    ShadowMatrixConfig,
    load_shadow_matrix_config,
)

SHADOW_MATRIX_RUN_SCHEMA_VERSION = "1.0"


def _atomic_text(path: Path, payload: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fields),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _cell_command(
    config: ShadowMatrixConfig,
    cell: ShadowMatrixCell,
    *,
    repetition: int,
    runs_root: Path,
    executable: str,
) -> tuple[str, list[str]]:
    run_id = f"{config.matrix_id}--{cell.cell_id}-r{repetition:02d}"
    command = [
        executable,
        "-m",
        "carla_vision.runtime",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--vehicle-id",
        str(config.vehicle_id),
        "--camera-id",
        str(config.camera_id),
        "--resolution",
        f"{config.resolution[0]}x{config.resolution[1]}",
        "--camera-fps",
        str(config.camera_fps),
        "--camera-fov",
        str(config.camera_fov),
        "--duration",
        str(config.duration_seconds),
        "--max-stale-seconds",
        str(config.max_stale_seconds),
        "--control",
        config.control,
        "--view",
        config.view,
        "--runs-root",
        str(runs_root),
        "--run-id",
        run_id,
        "--device",
        cell.detector.device,
        "--confidence",
        str(cell.detector.confidence),
        "--shadow-policy",
        cell.policy.backend,
        "--policy-device",
        cell.policy.device,
        "--policy-options",
        json.dumps(
            dict(cell.policy.options),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    ]
    if config.expected_map is not None:
        command.extend(("--expected-map", config.expected_map))
    if not config.record_video:
        command.append("--no-video")
    if cell.detector.model_package is not None:
        command.extend(("--model-package", str(cell.detector.model_package)))
    else:
        command.extend(
            (
                "--detector",
                str(cell.detector.backend),
                "--image-size",
                str(cell.detector.image_size),
            )
        )
        if cell.detector.weights is not None:
            command.extend(("--weights", str(cell.detector.weights)))
        if cell.detector.factory is not None:
            command.extend(("--detector-factory", cell.detector.factory))
    if cell.policy.factory is not None:
        command.extend(("--policy-factory", cell.policy.factory))
    if cell.policy.checkpoint is not None:
        command.extend(("--policy-checkpoint", str(cell.policy.checkpoint)))
    return run_id, command


def build_planned_rows(
    config: ShadowMatrixConfig,
    *,
    runs_root: str | Path,
    executable: str,
) -> list[dict[str, Any]]:
    root = Path(runs_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for cell in config.cells:
        for repetition in range(cell.repetitions):
            run_id, command = _cell_command(
                config,
                cell,
                repetition=repetition,
                runs_root=root,
                executable=executable,
            )
            try:
                control_index = command.index("--control")
                generated_control = command[control_index + 1]
            except (ValueError, IndexError) as error:
                raise RuntimeError(
                    "shadow matrix generated a command without a control value"
                ) from error
            if generated_control not in {"none", "teacher"}:
                raise RuntimeError("shadow matrix generated an unsafe control command")
            if "--shadow-policy" not in command:
                raise RuntimeError("shadow matrix generated a command without a policy")
            rows.append(
                {
                    "ordinal": ordinal,
                    "cell_id": cell.cell_id,
                    "label": cell.label,
                    "repetition": repetition,
                    "run_id": run_id,
                    "control": config.control,
                    "actuation_source": (
                        "privileged_teacher" if config.control == "teacher" else "none"
                    ),
                    "vision_policy_actuation_authorized": False,
                    "detector": cell.detector.as_dict(),
                    "policy": cell.policy.as_dict(),
                    "command": command,
                    "command_sha256": hashlib.sha256(canonical_json_bytes(command)).hexdigest(),
                }
            )
            ordinal += 1
    return rows


def _input_references(
    config: ShadowMatrixConfig,
    config_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = [
        {
            "kind": "shadow_matrix_preregistration",
            **fingerprint_file(config_path),
        }
    ]
    models: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cell in config.cells:
        detector = cell.detector
        if detector.model_package is not None:
            model = load_verified_model(detector.model_package)
            key = ("model_package", model.model_id)
            reference = {
                "kind": "vision_shadow_model_package",
                "cell_id": cell.cell_id,
                **dict(model.reference),
            }
        elif detector.weights is not None:
            key = ("weights", str(detector.weights))
            reference = {
                "kind": "vision_shadow_development_weights",
                "cell_id": cell.cell_id,
                "backend": detector.backend,
                **fingerprint_file(detector.weights),
            }
        else:
            key = ("factory", str(detector.factory))
            reference = {
                "kind": "vision_shadow_custom_detector",
                "cell_id": cell.cell_id,
                "factory": detector.factory,
            }
        if key not in seen:
            models.append(reference)
            seen.add(key)
        if cell.policy.checkpoint is not None:
            policy_key = ("policy_checkpoint", str(cell.policy.checkpoint))
            if policy_key not in seen:
                models.append(
                    {
                        "kind": "vision_shadow_policy_checkpoint",
                        "cell_id": cell.cell_id,
                        **fingerprint_file(cell.policy.checkpoint),
                    }
                )
                seen.add(policy_key)
    return inputs, models


def _checksum_index(root: Path, paths: Sequence[Path], output: Path) -> None:
    lines = []
    for path in sorted(
        paths,
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        reference = fingerprint_file(path)
        lines.append(f"{reference['sha256']}  {path.relative_to(root).as_posix()}")
    _atomic_text(output, "\n".join(lines) + "\n")


def _report(
    config: ShadowMatrixConfig,
    *,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> str:
    successes = sum(result.get("status") == "success" for result in results)
    failures = sum(result.get("status") == "failed" for result in results)
    command_lines = "\n".join(
        f"- `{row['run_id']}`: `{' '.join(shlex.quote(token) for token in row['command'])}`"
        for row in rows
    )
    return f"""# {config.title}

- Matrix ID: `{config.matrix_id}`
- Purpose: `{config.purpose}`
- Mode: `{mode}`
- Planned child runs: {len(rows)}
- Successful child runs: {successes}
- Failed child runs: {failures}
- Vehicle control source: `{config.control}`
- Vision-policy actuation authorized: **no**
- Expected map: `{config.expected_map or "not gated"}`

Every child command contains a vision-only shadow policy. `--control vision`
is prohibited. In teacher mode, only the existing privileged teacher may move
the simulator vehicle; policy proposals are logged and displayed but never sent
to CARLA.

## Planned commands

{command_lines}
"""


def plan_or_run_shadow_matrix(
    *,
    config_path: str | Path,
    runs_root: str | Path = "runs",
    execute: bool = False,
    acknowledge_teacher_motion: bool = False,
    output_id: str | None = None,
    executable: str = sys.executable,
    cli_args: Sequence[str] | Mapping[str, Any] = (),
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve(strict=True)
    config = load_shadow_matrix_config(resolved_config_path)
    if execute and config.control == "teacher" and not acknowledge_teacher_motion:
        raise PermissionError(
            "executing a teacher-driven shadow matrix requires --acknowledge-teacher-motion"
        )
    root = Path(runs_root).expanduser().resolve()
    parent_id = output_id or (config.matrix_id if execute else f"{config.matrix_id}-plan")
    rows = build_planned_rows(
        config,
        runs_root=root,
        executable=executable,
    )
    inputs, models = _input_references(config, resolved_config_path)
    tracker = RunArtifactTracker(
        root,
        run_id=parent_id,
        cli_args=cli_args,
        config={
            "schema_version": SHADOW_MATRIX_RUN_SCHEMA_VERSION,
            "object_type": "vision_shadow_matrix",
            "mode": "execute" if execute else "plan",
            "matrix": config.as_dict(),
            "runtime_executable": executable,
            "runs_root": str(root),
            "vision_policy_actuation_authorized": False,
        },
        repository_root=repository_root,
        carla_endpoint={"host": config.host, "port": config.port},
        carla_map=config.expected_map,
        model_refs=models,
        input_refs=inputs,
    )
    results: list[dict[str, Any]] = []
    execution_files: list[Path] = []
    with tracker:
        resolved_path = tracker.artifact_path("resolved_shadow_matrix.json")
        plan_jsonl_path = tracker.artifact_path("planned_runs.jsonl")
        plan_csv_path = tracker.artifact_path("planned_runs.csv")
        results_path = tracker.artifact_path("execution_results.jsonl")
        descriptor_path = tracker.artifact_path("shadow_matrix.json")
        report_path = tracker.artifact_path("shadow_matrix.md")
        checksum_path = tracker.artifact_path("checksums.sha256")
        _write_json(
            resolved_path,
            {
                "schema_version": SHADOW_MATRIX_RUN_SCHEMA_VERSION,
                "object_type": "vision_shadow_matrix",
                "mode": "execute" if execute else "plan",
                "matrix": config.as_dict(),
                "source_config": fingerprint_file(resolved_config_path),
                "runtime_executable": executable,
                "runs_root": str(root),
                "vision_policy_actuation_authorized": False,
            },
        )
        _write_jsonl(plan_jsonl_path, rows)
        _write_csv(
            plan_csv_path,
            [
                {
                    "ordinal": row["ordinal"],
                    "cell_id": row["cell_id"],
                    "label": row["label"],
                    "repetition": row["repetition"],
                    "run_id": row["run_id"],
                    "control": row["control"],
                    "actuation_source": row["actuation_source"],
                    "vision_policy_actuation_authorized": row["vision_policy_actuation_authorized"],
                    "command_sha256": row["command_sha256"],
                    "command": json.dumps(
                        row["command"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
                for row in rows
            ],
            (
                "ordinal",
                "cell_id",
                "label",
                "repetition",
                "run_id",
                "control",
                "actuation_source",
                "vision_policy_actuation_authorized",
                "command_sha256",
                "command",
            ),
        )
        if execute:
            for row in rows:
                completed: subprocess.CompletedProcess[str] | None = None
                stdout_path = tracker.artifact_path(f"children/{row['run_id']}/stdout.log")
                stderr_path = tracker.artifact_path(f"children/{row['run_id']}/stderr.log")
                try:
                    completed = subprocess.run(
                        row["command"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=config.duration_seconds + 180.0,
                    )
                    _atomic_text(stdout_path, completed.stdout)
                    _atomic_text(stderr_path, completed.stderr)
                    execution_files.extend((stdout_path, stderr_path))
                    if completed.returncode != 0:
                        raise RuntimeError(f"runtime exited with code {completed.returncode}")
                    child = load_verified_vision_shadow_run(root / str(row["run_id"]))
                    child_manifest = fingerprint_file(child.root / "manifest.json")
                    tracker.add_input_reference(
                        {
                            "kind": "vision_shadow_matrix_child",
                            "cell_id": row["cell_id"],
                            "run_id": child.run_id,
                            **child_manifest,
                        }
                    )
                    result = {
                        "ordinal": row["ordinal"],
                        "cell_id": row["cell_id"],
                        "run_id": row["run_id"],
                        "status": "success",
                        "returncode": completed.returncode,
                        "child_manifest": child_manifest,
                        "proposal_count": len(child.proposal_records),
                        "error": None,
                    }
                except BaseException as error:
                    result = {
                        "ordinal": row["ordinal"],
                        "cell_id": row["cell_id"],
                        "run_id": row["run_id"],
                        "status": "failed",
                        "returncode": (
                            completed.returncode
                            if isinstance(completed, subprocess.CompletedProcess)
                            else None
                        ),
                        "child_manifest": None,
                        "proposal_count": 0,
                        "error": {
                            "type": type(error).__qualname__,
                            "message": str(error),
                        },
                    }
                    if not config.continue_on_failure:
                        results.append(result)
                        break
                results.append(result)
        _write_jsonl(
            results_path,
            results
            or [
                {
                    "status": "not_executed",
                    "planned_run_count": len(rows),
                    "vision_policy_actuation_authorized": False,
                }
            ],
        )
        report = _report(
            config,
            mode="execute" if execute else "plan",
            rows=rows,
            results=results,
        )
        _atomic_text(report_path, report)
        successful = sum(result.get("status") == "success" for result in results)
        failed = sum(result.get("status") == "failed" for result in results)
        descriptor = {
            "schema_version": SHADOW_MATRIX_RUN_SCHEMA_VERSION,
            "object_type": "vision_shadow_matrix_release",
            "status": (
                "planned"
                if not execute
                else ("complete" if failed == 0 and successful == len(rows) else "partial")
            ),
            "matrix_id": config.matrix_id,
            "run_id": parent_id,
            "title": config.title,
            "purpose": config.purpose,
            "mode": "execute" if execute else "plan",
            "planned_run_count": len(rows),
            "successful_run_count": successful,
            "failed_run_count": failed,
            "control": config.control,
            "vision_policy_actuation_authorized": False,
            "expected_map": config.expected_map,
            "runtime_sensor_contract": "front_monocular_rgb_only",
            "limitations": [
                "Shadow proposals are not vehicle commands and do not establish driving safety.",
                "Teacher motion is privileged and evaluated separately from policy inputs.",
                "A planned matrix contains no empirical runtime evidence.",
                "Live endpoint state and traffic can change between sequential child runs.",
            ],
        }
        _write_json(descriptor_path, descriptor)
        payloads = [
            resolved_path,
            plan_jsonl_path,
            plan_csv_path,
            results_path,
            descriptor_path,
            report_path,
            *execution_files,
        ]
        _checksum_index(tracker.run_dir, payloads, checksum_path)
        role_by_path = {
            resolved_path: "resolved_shadow_matrix_configuration",
            plan_jsonl_path: "shadow_matrix_plan_jsonl",
            plan_csv_path: "shadow_matrix_plan_csv",
            results_path: "shadow_matrix_execution_results",
            descriptor_path: "shadow_matrix_release_manifest",
            report_path: "shadow_matrix_report",
            checksum_path: "shadow_matrix_checksum_index",
            **{path: "shadow_matrix_child_log" for path in execution_files},
        }
        for path, role in role_by_path.items():
            tracker.register_artifact(
                path,
                role=role,
                metadata={
                    "matrix_id": config.matrix_id,
                    "mode": "execute" if execute else "plan",
                },
            )
    return {
        "matrix_id": config.matrix_id,
        "run_id": parent_id,
        "run_dir": str(tracker.run_dir),
        "mode": "execute" if execute else "plan",
        "descriptor": descriptor,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or sequentially execute a live vision-only shadow matrix; "
            "policy proposals are never applied to the vehicle"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-teacher-motion", action="store_true")
    parser.add_argument("--output-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = plan_or_run_shadow_matrix(
        config_path=args.config,
        runs_root=args.runs_root,
        execute=args.execute,
        acknowledge_teacher_motion=args.acknowledge_teacher_motion,
        output_id=args.output_id,
        cli_args=vars(args),
        repository_root=Path.cwd(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "SHADOW_MATRIX_RUN_SCHEMA_VERSION",
    "build_planned_rows",
    "main",
    "parse_args",
    "plan_or_run_shadow_matrix",
]
