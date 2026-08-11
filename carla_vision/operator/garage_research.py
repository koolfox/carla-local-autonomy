"""Allow-listed research launcher used by the additive Garage UI.

No request can provide a shell command or Python module.  Each supported action
maps to one fixed project module and a small validated parameter contract.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .commands import CommandPlan

GARAGE_RESEARCH_SCHEMA_VERSION = "1.0"
GARAGE_RESEARCH_KINDS = frozenset(
    {
        "teacher_capture",
        "imitation_train",
        "voxel_capture",
        "voxel_flow_capture",
        "voxel_train",
        "voxel_flow_train",
        "voxel_shadow",
        "voxel_benchmark",
        "closed_loop_evaluate",
    }
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class GarageResearchRequest:
    schema_version: str
    kind: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GarageResearchRequest":
        if set(raw) != {"schema_version", "kind", "parameters"}:
            raise ValueError("Garage research request requires schema_version, kind, and parameters")
        schema = str(raw["schema_version"])
        if schema != GARAGE_RESEARCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported Garage research schema {schema!r}")
        kind = str(raw["kind"]).strip().lower()
        if kind not in GARAGE_RESEARCH_KINDS:
            raise ValueError(f"unsupported Garage research kind {kind!r}")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("Garage research parameters must be an object")
        return cls(schema_version=schema, kind=kind, parameters=dict(parameters))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


def _keys(
    raw: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    keys = {str(key) for key in raw}
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{name} is missing parameters: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown parameters: {', '.join(unknown)}")


def _identifier(value: Any, name: str) -> str:
    result = str(value).strip()
    if not _ID.fullmatch(result):
        raise ValueError(f"{name} must use 1-128 letters, digits, periods, underscores, or hyphens")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = str(value).strip().lower()
    if result not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


def _workspace_path(
    workspace: Path,
    value: Any,
    name: str,
    *,
    kind: str,
    suffixes: tuple[str, ...] = (),
) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{name} is required")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(workspace)
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{name} must be a file")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"{name} has an unsupported file type")
    return resolved


def _unused_directory(workspace: Path, root: str, run_id: str) -> Path:
    output = (workspace / root / _identifier(run_id, "run_id")).resolve()
    output.relative_to(workspace)
    if output.exists():
        raise FileExistsError(f"output already exists: {output.relative_to(workspace)}")
    return output


def _module_command(module: str, *tokens: str) -> tuple[str, ...]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        raise ValueError("internal Garage module name is invalid")
    invocation = f"from {module} import main; raise SystemExit(main())"
    return (sys.executable, "-c", invocation, *tokens)


def _teacher_plan(request: GarageResearchRequest, workspace: Path, host: str, port: int) -> CommandPlan:
    raw = request.parameters
    _keys(
        raw,
        required={"scenario_plan", "dataset_id", "behavior", "max_episodes", "acknowledge"},
        name="teacher capture",
    )
    scenario = _workspace_path(workspace, raw["scenario_plan"], "scenario_plan", kind="file", suffixes=(".json",))
    dataset_id = _identifier(raw["dataset_id"], "dataset_id")
    output = workspace / "datasets" / dataset_id
    if output.exists():
        raise FileExistsError(f"dataset already exists: datasets/{dataset_id}")
    behavior = _choice(raw["behavior"], "behavior", {"cautious", "normal", "aggressive"})
    max_episodes = _integer(raw["max_episodes"], "max_episodes", 1, 1000)
    acknowledged = _boolean(raw["acknowledge"], "acknowledge")
    if not acknowledged:
        raise ValueError("teacher capture requires exclusive-tick/map-reload acknowledgement")
    command = _module_command(
        "carla_vision.native.behavior_teacher",
        "--scenario-plan", str(scenario),
        "--dataset-id", dataset_id,
        "--datasets-root", str(workspace / "datasets"),
        "--host", host,
        "--port", str(port),
        "--behavior", behavior,
        "--max-episodes", str(max_episodes),
        "--acknowledge-exclusive-tick-owner",
    )
    return CommandPlan(
        kind=request.kind,
        title=f"Behavior teacher · {dataset_id}",
        command=command,
        expected_output=output,
        motion_authorized=True,
        destructive=True,
        note="Loads scenario maps, owns world ticks, records BehaviorAgent controls, then verifies the dataset.",
    )


def _training_plan(
    request: GarageResearchRequest,
    workspace: Path,
    *,
    module: str,
    output_root: str,
    title: str,
) -> CommandPlan:
    raw = request.parameters
    _keys(
        raw,
        required={"dataset", "run_id", "device", "epochs", "dry_run"},
        name=title,
    )
    dataset = _workspace_path(workspace, raw["dataset"], "dataset", kind="directory")
    run_id = _identifier(raw["run_id"], "run_id")
    output = _unused_directory(workspace, output_root, run_id)
    device = _choice(raw["device"], "device", {"auto", "cpu", "cuda", "mps"})
    epochs = _integer(raw["epochs"], "epochs", 1, 1000)
    dry_run = _boolean(raw["dry_run"], "dry_run")
    tokens = [
        "--dataset", str(dataset),
        "--output", str(output),
        "--device", device,
        "--epochs", str(epochs),
    ]
    if dry_run:
        tokens.append("--dry-run")
    return CommandPlan(
        kind=request.kind,
        title=f"{title} · {run_id}",
        command=_module_command(module, *tokens),
        expected_output=None if dry_run else output,
        motion_authorized=False,
        destructive=False,
        note="Uses the existing project trainer; no CARLA vehicle control is issued.",
    )


def _capture_plan(
    request: GarageResearchRequest,
    workspace: Path,
    host: str,
    port: int,
    *,
    flow: bool,
) -> CommandPlan:
    raw = request.parameters
    required = {"run_id", "frames", "dry_run"}
    optional = {"mode", "checkpoint", "device"}
    _keys(raw, required=required, optional=optional, name="voxel capture")
    run_id = _identifier(raw["run_id"], "run_id")
    output = _unused_directory(workspace, "runs", run_id)
    frames = _integer(raw["frames"], "frames", 1, 100000)
    dry_run = _boolean(raw["dry_run"], "dry_run")
    tokens = [
        "--host", host,
        "--port", str(port),
        "--role-name", "research_drive_ego",
        "--frames", str(frames),
        "--output", str(output),
    ]
    if flow:
        module = "carla_vision.voxel.flow_capture"
        title = "Voxel flow capture"
    else:
        module = "carla_vision.voxel.capture"
        title = "Voxel capture"
        mode = _choice(raw.get("mode", "teacher"), "mode", {"teacher", "rgb-only"})
        tokens.extend(("--mode", mode))
        if mode == "rgb-only":
            checkpoint = _workspace_path(
                workspace,
                raw.get("checkpoint", ""),
                "checkpoint",
                kind="file",
                suffixes=(".pt", ".pth", ".ckpt"),
            )
            device = _choice(raw.get("device", "cpu"), "device", {"cpu", "cuda", "mps"})
            tokens.extend(
                (
                    "--predictor-factory",
                    "carla_vision.voxel.model_examples.temporal_flow:create_predictor",
                    "--predictor-checkpoint",
                    str(checkpoint),
                    "--predictor-device",
                    device,
                )
            )
    if dry_run:
        tokens.append("--dry-run")
    return CommandPlan(
        kind=request.kind,
        title=f"{title} · {run_id}",
        command=_module_command(module, *tokens),
        expected_output=None if dry_run else output,
        motion_authorized=False,
        destructive=False,
        note="Attaches temporary camera sensors to the current Garage ego; it never owns vehicle control.",
    )


def _shadow_plan(request: GarageResearchRequest, workspace: Path, host: str, port: int) -> CommandPlan:
    raw = request.parameters
    _keys(
        raw,
        required={"run_id", "checkpoint", "device", "frames", "dry_run"},
        name="voxel shadow",
    )
    run_id = _identifier(raw["run_id"], "run_id")
    output = _unused_directory(workspace, "runs", run_id)
    checkpoint = _workspace_path(
        workspace,
        raw["checkpoint"],
        "checkpoint",
        kind="file",
        suffixes=(".pt", ".pth", ".ckpt"),
    )
    device = _choice(raw["device"], "device", {"cpu", "cuda", "mps"})
    frames = _integer(raw["frames"], "frames", 1, 100000)
    dry_run = _boolean(raw["dry_run"], "dry_run")
    tokens = [
        "--host", host,
        "--port", str(port),
        "--role-name", "research_drive_ego",
        "--frames", str(frames),
        "--output", str(output),
        "--predictor-factory", "carla_vision.voxel.model_examples.temporal_flow:create_predictor",
        "--predictor-checkpoint", str(checkpoint),
        "--predictor-device", device,
    ]
    if dry_run:
        tokens.append("--dry-run")
    return CommandPlan(
        kind=request.kind,
        title=f"Voxel shadow · {run_id}",
        command=_module_command("carla_vision.voxel.shadow", *tokens),
        expected_output=None if dry_run else output,
        motion_authorized=False,
        destructive=False,
        note="Observes and scores trajectories beside the current ego; shadow code has zero actuation calls.",
    )


def _benchmark_plan(request: GarageResearchRequest, workspace: Path) -> CommandPlan:
    raw = request.parameters
    _keys(raw, required={"run", "run_id"}, name="voxel benchmark")
    source = _workspace_path(workspace, raw["run"], "run", kind="directory")
    run_id = _identifier(raw["run_id"], "run_id")
    output_dir = _unused_directory(workspace, "runs", run_id)
    output_file = output_dir / "voxel-benchmark.json"
    # The benchmark CLI creates the parent directory itself.
    command = _module_command(
        "carla_vision.voxel.benchmark",
        "--run", str(source),
        "--output", str(output_file),
    )
    return CommandPlan(
        kind=request.kind,
        title=f"Voxel benchmark · {run_id}",
        command=command,
        expected_output=None,
        motion_authorized=False,
        destructive=False,
        note="Offline benchmark over recorded teacher/prediction artifacts; no CARLA connection or control.",
    )


def _closed_loop_plan(
    request: GarageResearchRequest,
    workspace: Path,
    host: str,
    port: int,
) -> CommandPlan:
    raw = request.parameters
    _keys(
        raw,
        required={"run_id", "driver_label", "duration", "dry_run"},
        name="closed-loop evaluation",
    )
    run_id = _identifier(raw["run_id"], "run_id")
    driver_label = _identifier(raw["driver_label"], "driver_label")
    duration = _number(raw["duration"], "duration", 1.0, 3600.0)
    dry_run = _boolean(raw["dry_run"], "dry_run")
    output = _unused_directory(workspace, "runs", run_id)
    tokens = [
        "--host", host,
        "--port", str(port),
        "--role-name", "research_drive_ego",
        "--driver-label", driver_label,
        "--run-label", run_id,
        "--duration", str(duration),
        "--output", str(output),
    ]
    if dry_run:
        tokens.append("--dry-run")
    return CommandPlan(
        kind=request.kind,
        title=f"Closed-loop observer · {run_id}",
        command=_module_command("carla_vision.closed_loop_cli", *tokens),
        expected_output=None if dry_run else output,
        motion_authorized=False,
        destructive=False,
        note="Read-only observer: collision/lane/route/brake metrics; it never calls vehicle control.",
    )


def build_garage_research_plan(
    request: GarageResearchRequest,
    *,
    workspace: str | Path,
    carla_host: str,
    carla_port: int,
) -> CommandPlan:
    root = Path(workspace).expanduser().resolve(strict=True)
    kind = request.kind
    if kind == "teacher_capture":
        return _teacher_plan(request, root, carla_host, carla_port)
    if kind == "imitation_train":
        return _training_plan(
            request,
            root,
            module="carla_vision.imitation.runner",
            output_root="models/imitation",
            title="Imitation training",
        )
    if kind == "voxel_train":
        return _training_plan(
            request,
            root,
            module="carla_vision.voxel.training.runner",
            output_root="models/voxel",
            title="Voxel training",
        )
    if kind == "voxel_flow_train":
        return _training_plan(
            request,
            root,
            module="carla_vision.voxel.training.flow_runner",
            output_root="models/voxel-flow",
            title="Voxel flow training",
        )
    if kind == "voxel_capture":
        return _capture_plan(request, root, carla_host, carla_port, flow=False)
    if kind == "voxel_flow_capture":
        return _capture_plan(request, root, carla_host, carla_port, flow=True)
    if kind == "voxel_shadow":
        return _shadow_plan(request, root, carla_host, carla_port)
    if kind == "voxel_benchmark":
        return _benchmark_plan(request, root)
    if kind == "closed_loop_evaluate":
        return _closed_loop_plan(request, root, carla_host, carla_port)
    raise ValueError(f"unsupported Garage research kind {kind!r}")


__all__ = [
    "GARAGE_RESEARCH_KINDS",
    "GARAGE_RESEARCH_SCHEMA_VERSION",
    "GarageResearchRequest",
    "build_garage_research_plan",
]
