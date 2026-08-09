"""Safe tokenized command plans for the local operator UI.

The UI never accepts a shell command. Every workflow has an explicit parameter
allow-list and produces a ``sys.executable -m ...`` token array.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..replay.contracts import ReplayConfig
from ..shadow.contracts import ShadowMatrixConfig
from .contracts import OperatorJobRequest

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class CommandPlan:
    kind: str
    title: str
    command: tuple[str, ...]
    expected_output: Path | None
    motion_authorized: bool
    destructive: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "command": list(self.command),
            "expected_output": (
                str(self.expected_output) if self.expected_output is not None else None
            ),
            "motion_authorized": self.motion_authorized,
            "destructive": self.destructive,
            "note": self.note,
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


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in result or "\n" in result or "\r" in result:
        raise ValueError(f"{name} must be a single-line string")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name)
    if not _ID_PATTERN.fullmatch(result):
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
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _text(value, name)
    if result not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


def _workspace_path(
    workspace: Path,
    value: Any,
    name: str,
    *,
    kind: str,
) -> Path:
    raw = _text(value, name)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {raw}") from error
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{name} must stay inside the project workspace") from error
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{name} must be a file")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _optional_workspace_path(
    workspace: Path,
    value: Any,
    name: str,
    *,
    kind: str,
) -> Path | None:
    if value is None or value == "":
        return None
    return _workspace_path(workspace, value, name, kind=kind)


def _unused_output(path: Path, name: str) -> Path:
    if path.exists():
        raise FileExistsError(f"{name} already exists and will not be overwritten: {path}")
    return path


def _module_command(executable: str, module: str, *tokens: str) -> tuple[str, ...]:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", module):
        raise ValueError("internal module name is invalid")
    invocation = f"from {module} import main; raise SystemExit(main())"
    return (executable, "-c", invocation, *tokens)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _derived_replay_config(
    workspace: Path,
    source: Path,
    replay_id: str,
) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("replay configuration must contain an object")
    resolved = ReplayConfig.from_mapping(payload, base=source.parent).as_dict()
    resolved["replay_id"] = replay_id
    resolved["title"] = f"{resolved['title']} · operator {replay_id}"
    ReplayConfig.from_mapping(resolved, base=workspace)
    output = _unused_output(
        workspace / "operator_configs" / "replay" / f"{replay_id}.json",
        "derived replay configuration",
    )
    _atomic_json(output, resolved)
    return output


def _derived_shadow_config(
    workspace: Path,
    source: Path,
    matrix_id: str,
) -> tuple[Path, ShadowMatrixConfig]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("shadow matrix configuration must contain an object")
    resolved = ShadowMatrixConfig.from_mapping(payload, base=source.parent).as_dict()
    resolved["matrix_id"] = matrix_id
    resolved["title"] = f"{resolved['title']} · operator {matrix_id}"
    validated = ShadowMatrixConfig.from_mapping(resolved, base=workspace)
    output = _unused_output(
        workspace / "operator_configs" / "shadow" / f"{matrix_id}.json",
        "derived shadow configuration",
    )
    _atomic_json(output, validated.as_dict())
    return output, validated


def _build_live(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    required = {
        "run_id",
        "host",
        "port",
        "vehicle_id",
        "camera_id",
        "resolution",
        "camera_fps",
        "camera_fov",
        "expected_map",
        "detector",
        "weights",
        "model_package",
        "device",
        "image_size",
        "confidence",
        "control",
        "cruise_speed",
        "duration",
        "max_stale_seconds",
        "view",
        "record_video",
        "shadow_policy",
        "policy_options",
        "acknowledge_teacher_motion",
    }
    _keys(parameters, required=required, optional={"spectator_follow"}, name="live job")
    run_id = _identifier(parameters["run_id"], "run_id")
    host = _text(parameters["host"], "host")
    port = _integer(parameters["port"], "port", 1, 65535)
    vehicle_id = _integer(parameters["vehicle_id"], "vehicle_id", 1, 2**31 - 1)
    camera_id = _integer(parameters["camera_id"], "camera_id", 1, 2**31 - 1)
    resolution = _text(parameters["resolution"], "resolution")
    if not re.fullmatch(r"[1-9][0-9]{2,4}x[1-9][0-9]{2,4}", resolution):
        raise ValueError("resolution must use WIDTHxHEIGHT")
    camera_fps = _number(parameters["camera_fps"], "camera_fps", 0.1, 120.0)
    camera_fov = _number(parameters["camera_fov"], "camera_fov", 30.0, 150.0)
    expected_map = _text(parameters["expected_map"], "expected_map", allow_empty=True)
    detector = _choice(parameters["detector"], "detector", {"rtdetr", "yolo"})
    weights = _optional_workspace_path(
        workspace,
        parameters["weights"],
        "weights",
        kind="file",
    )
    model_package = _optional_workspace_path(
        workspace,
        parameters["model_package"],
        "model_package",
        kind="directory",
    )
    if model_package is None and weights is None:
        raise ValueError("live job requires weights or a model package")
    if model_package is not None and weights is not None:
        raise ValueError("weights and model_package are mutually exclusive")
    device = _text(parameters["device"], "device")
    image_size = _integer(parameters["image_size"], "image_size", 64, 4096)
    confidence = _number(parameters["confidence"], "confidence", 0.0, 1.0)
    control = _choice(parameters["control"], "control", {"none", "teacher"})
    acknowledged = _boolean(
        parameters["acknowledge_teacher_motion"],
        "acknowledge_teacher_motion",
    )
    if control == "teacher" and not acknowledged:
        raise PermissionError("teacher motion requires explicit acknowledgement")
    cruise_speed = _number(parameters["cruise_speed"], "cruise_speed", 0.1, 8.0)
    duration = _number(parameters["duration"], "duration", 1.0, 3600.0)
    stale = _number(
        parameters["max_stale_seconds"],
        "max_stale_seconds",
        0.05,
        10.0,
    )
    view = _choice(parameters["view"], "view", {"none", "overlay", "split"})
    record_video = _boolean(parameters["record_video"], "record_video")
    spectator_follow = _boolean(parameters.get("spectator_follow", False), "spectator_follow")
    shadow_policy = _choice(
        parameters["shadow_policy"],
        "shadow_policy",
        {"none", "hazard-stop"},
    )
    policy_options = parameters["policy_options"]
    if not isinstance(policy_options, Mapping):
        raise TypeError("policy_options must be an object")

    output = _unused_output(workspace / "runs" / run_id, "live run")
    tokens = [
        "--host",
        host,
        "--port",
        str(port),
        "--vehicle-id",
        str(vehicle_id),
        "--camera-id",
        str(camera_id),
        "--resolution",
        resolution,
        "--camera-fps",
        str(camera_fps),
        "--camera-fov",
        str(camera_fov),
        "--device",
        device,
        "--confidence",
        str(confidence),
        "--control",
        control,
        "--cruise-speed",
        str(cruise_speed),
        "--duration",
        str(duration),
        "--max-stale-seconds",
        str(stale),
        "--view",
        view,
        "--runs-root",
        str(workspace / "runs"),
        "--run-id",
        run_id,
    ]
    if expected_map:
        tokens.extend(("--expected-map", expected_map))
    if model_package is not None:
        tokens.extend(("--model-package", str(model_package)))
    else:
        tokens.extend(
            (
                "--detector",
                detector,
                "--weights",
                str(weights),
                "--image-size",
                str(image_size),
            )
        )
    if shadow_policy != "none":
        tokens.extend(
            (
                "--shadow-policy",
                shadow_policy,
                "--policy-options",
                json.dumps(
                    dict(policy_options),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        )
    if not record_video:
        tokens.append("--no-video")
    if spectator_follow:
        tokens.append("--spectator-follow")
    return CommandPlan(
        kind="live",
        title=f"Live RGB session {run_id}",
        command=_module_command(executable, "carla_vision.runtime", *tokens),
        expected_output=output,
        motion_authorized=control == "teacher",
        destructive=False,
        note=(
            "Teacher may move the selected vehicle; any vision-policy proposal "
            "remains non-actuating. Server spectator follow is best-effort "
            "operator visualization."
            if control == "teacher"
            else (
                "Perception-only session; no vehicle command is requested. "
                "Server spectator follow is best-effort operator visualization."
            )
        ),
    )


def _build_scenario_plan(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={"suite", "split_plan", "run_id"},
        name="scenario-plan job",
    )
    suite = _workspace_path(workspace, parameters["suite"], "suite", kind="file")
    split_plan = _workspace_path(
        workspace,
        parameters["split_plan"],
        "split_plan",
        kind="file",
    )
    run_id = _identifier(parameters["run_id"], "run_id")
    output = _unused_output(workspace / "runs" / run_id, "scenario plan")
    return CommandPlan(
        kind="scenario_plan",
        title=f"Plan situation {run_id}",
        command=_module_command(
            executable,
            "carla_vision.scenarios.planner",
            "--suite",
            str(suite),
            "--split-plan",
            str(split_plan),
            "--runs-root",
            str(workspace / "runs"),
            "--run-id",
            run_id,
        ),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note="Planning is offline and does not contact or mutate CARLA.",
    )


def _build_native_capture(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    required = {
        "scenario_plan",
        "dataset_id",
        "host",
        "port",
        "partition",
        "max_episodes",
        "timeout",
        "sensor_timeout",
        "carla_python_api",
        "dry_run",
        "acknowledge_exclusive_tick_owner",
    }
    _keys(parameters, required=required, name="native-capture job")
    plan = _workspace_path(
        workspace,
        parameters["scenario_plan"],
        "scenario_plan",
        kind="directory",
    )
    dataset_id = _identifier(parameters["dataset_id"], "dataset_id")
    host = _text(parameters["host"], "host")
    port = _integer(parameters["port"], "port", 1, 65535)
    partition = _choice(
        parameters["partition"],
        "partition",
        {
            "train",
            "val",
            "val_seen",
            "val_map_ood",
            "val_weather_ood",
            "test",
            "test_seen",
            "test_map_ood",
            "test_weather_ood",
            "unassigned",
        },
    )
    max_episodes = _integer(parameters["max_episodes"], "max_episodes", 1, 10000)
    timeout = _number(parameters["timeout"], "timeout", 1.0, 300.0)
    sensor_timeout = _number(
        parameters["sensor_timeout"],
        "sensor_timeout",
        1.0,
        300.0,
    )
    dry_run = _boolean(parameters["dry_run"], "dry_run")
    acknowledged = _boolean(
        parameters["acknowledge_exclusive_tick_owner"],
        "acknowledge_exclusive_tick_owner",
    )
    if not dry_run and not acknowledged:
        raise PermissionError(
            "real native collection requires explicit exclusive-tick-owner acknowledgement"
        )
    api_value = parameters["carla_python_api"]
    api_path: Path | None = None
    if api_value not in {None, ""}:
        raw_api = Path(_text(api_value, "carla_python_api")).expanduser()
        try:
            api_path = raw_api.resolve(strict=True)
        except OSError as error:
            raise ValueError("carla_python_api does not exist") from error

    output = _unused_output(workspace / "datasets" / dataset_id, "dataset")
    tokens = [
        "--scenario-plan",
        str(plan),
        "--dataset-id",
        dataset_id,
        "--datasets-root",
        str(workspace / "datasets"),
        "--host",
        host,
        "--port",
        str(port),
        "--partition",
        partition,
        "--max-episodes",
        str(max_episodes),
        "--timeout",
        str(timeout),
        "--sensor-timeout",
        str(sensor_timeout),
    ]
    if api_path is not None:
        tokens.extend(("--carla-python-api", str(api_path)))
    if dry_run:
        tokens.append("--dry-run")
    else:
        tokens.append("--acknowledge-exclusive-tick-owner")
    return CommandPlan(
        kind="native_capture",
        title=f"Native dataset {dataset_id}",
        command=_module_command(executable, "carla_vision.native.worker", *tokens),
        expected_output=None if dry_run else output,
        motion_authorized=not dry_run,
        destructive=not dry_run,
        note=(
            "Dry run verifies selection only and does not import or contact CARLA."
            if dry_run
            else "Real collection reloads the map, destroys existing actors, and owns world.tick()."
        ),
    )


def _build_native_preflight(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    required = {
        "scenario_plan",
        "dataset_id",
        "run_id",
        "host",
        "port",
        "partition",
        "max_episodes",
        "timeout",
        "carla_python_api",
        "confirm_world_reload",
        "confirm_exclusive_tick_owner",
    }
    _keys(parameters, required=required, name="native-preflight job")
    plan = _workspace_path(
        workspace,
        parameters["scenario_plan"],
        "scenario_plan",
        kind="directory",
    )
    dataset_id = _identifier(parameters["dataset_id"], "dataset_id")
    run_id = _identifier(parameters["run_id"], "run_id")
    host = _text(parameters["host"], "host")
    port = _integer(parameters["port"], "port", 1, 65535)
    partition = _choice(
        parameters["partition"],
        "partition",
        {
            "train",
            "val",
            "val_seen",
            "val_map_ood",
            "val_weather_ood",
            "test",
            "test_seen",
            "test_map_ood",
            "test_weather_ood",
            "unassigned",
        },
    )
    max_episodes = _integer(parameters["max_episodes"], "max_episodes", 1, 10000)
    timeout = _number(parameters["timeout"], "timeout", 0.1, 300.0)
    confirm_world_reload = _boolean(
        parameters["confirm_world_reload"],
        "confirm_world_reload",
    )
    confirm_exclusive_tick_owner = _boolean(
        parameters["confirm_exclusive_tick_owner"],
        "confirm_exclusive_tick_owner",
    )
    api_value = parameters["carla_python_api"]
    api_path: Path | None = None
    if api_value not in {None, ""}:
        raw_api = Path(_text(api_value, "carla_python_api")).expanduser()
        try:
            api_path = raw_api.resolve(strict=True)
        except OSError as error:
            raise ValueError("carla_python_api does not exist") from error
    output = _unused_output(workspace / "runs" / run_id, "native preflight run")
    tokens = [
        "--scenario-plan",
        str(plan),
        "--dataset-id",
        dataset_id,
        "--datasets-root",
        str(workspace / "datasets"),
        "--runs-root",
        str(workspace / "runs"),
        "--run-id",
        run_id,
        "--host",
        host,
        "--port",
        str(port),
        "--partition",
        partition,
        "--max-episodes",
        str(max_episodes),
        "--timeout",
        str(timeout),
    ]
    if api_path is not None:
        tokens.extend(("--carla-python-api", str(api_path)))
    if confirm_world_reload:
        tokens.append("--confirm-world-reload")
    if confirm_exclusive_tick_owner:
        tokens.append("--confirm-exclusive-tick-owner")
    return CommandPlan(
        kind="native_preflight",
        title=f"Native readiness {run_id}",
        command=_module_command(executable, "carla_vision.native.preflight", *tokens),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note=(
            "Read-only readiness assessment: verifies the plan, endpoint, "
            "versions, PythonAPI, output ID, and manual gates."
        ),
    )


def _build_dataset_qa(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={"dataset", "run_id", "montage_count"},
        name="dataset-QA job",
    )
    dataset = _workspace_path(
        workspace,
        parameters["dataset"],
        "dataset",
        kind="directory",
    )
    run_id = _identifier(parameters["run_id"], "run_id")
    montage_count = _integer(
        parameters["montage_count"],
        "montage_count",
        1,
        10000,
    )
    output = _unused_output(workspace / "runs" / run_id, "dataset-QA run")
    return CommandPlan(
        kind="dataset_qa",
        title=f"Automated dataset QA {run_id}",
        command=_module_command(
            executable,
            "carla_vision.dataset.qa",
            "--dataset",
            str(dataset),
            "--runs-root",
            str(workspace / "runs"),
            "--run-id",
            run_id,
            "--montage-count",
            str(montage_count),
        ),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note=(
            "Read-only automated dataset integrity and label QA; "
            "this is not human review or release signoff."
        ),
    )


def _build_train(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    required = {
        "config",
        "dataset",
        "weights",
        "run_id",
        "device",
        "dry_run",
        "acknowledge_real_training",
    }
    _keys(parameters, required=required, name="training job")
    config = _workspace_path(workspace, parameters["config"], "config", kind="file")
    dataset = _workspace_path(workspace, parameters["dataset"], "dataset", kind="directory")
    weights = _workspace_path(workspace, parameters["weights"], "weights", kind="file")
    run_id = _identifier(parameters["run_id"], "run_id")
    device = _text(parameters["device"], "device")
    dry_run = _boolean(parameters["dry_run"], "dry_run")
    acknowledged = _boolean(
        parameters["acknowledge_real_training"],
        "acknowledge_real_training",
    )
    if not dry_run and not acknowledged:
        raise PermissionError("real training requires explicit acknowledgement")
    output = _unused_output(workspace / "runs" / run_id, "training run")
    tokens = [
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--weights",
        str(weights),
        "--runs-root",
        str(workspace / "runs"),
        "--run-id",
        run_id,
        "--device",
        device,
    ]
    if dry_run:
        tokens.append("--dry-run")
    return CommandPlan(
        kind="train",
        title=f"Detector training {run_id}",
        command=_module_command(executable, "carla_vision.training.runner", *tokens),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note=(
            "Dry run verifies and resolves the training contract without backend training."
            if dry_run
            else "Real model training is authorized and may consume substantial compute."
        ),
    )


def _build_replay(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={
            "config",
            "dataset",
            "evaluation_config",
            "replay_id",
            "acknowledge_locked_test",
        },
        name="replay job",
    )
    config = _workspace_path(workspace, parameters["config"], "config", kind="file")
    dataset = _workspace_path(workspace, parameters["dataset"], "dataset", kind="directory")
    evaluation = _workspace_path(
        workspace,
        parameters["evaluation_config"],
        "evaluation_config",
        kind="file",
    )
    locked = _boolean(parameters["acknowledge_locked_test"], "acknowledge_locked_test")
    replay_id = _identifier(parameters["replay_id"], "replay_id").lower()
    output = _unused_output(workspace / "runs" / replay_id, "paired replay")
    derived_config = _derived_replay_config(workspace, config, replay_id)
    tokens = [
        "--config",
        str(derived_config),
        "--dataset",
        str(dataset),
        "--evaluation-config",
        str(evaluation),
        "--runs-root",
        str(workspace / "runs"),
    ]
    if locked:
        tokens.append("--acknowledge-locked-test")
    return CommandPlan(
        kind="replay",
        title=f"Paired replay {replay_id}",
        command=_module_command(executable, "carla_vision.replay.runner", *tokens),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note="All models consume the same immutable RGB order; CARLA is not contacted.",
    )


def _build_shadow_matrix(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={
            "config",
            "matrix_id",
            "execute",
            "acknowledge_teacher_motion",
        },
        name="shadow-matrix job",
    )
    config = _workspace_path(workspace, parameters["config"], "config", kind="file")
    execute = _boolean(parameters["execute"], "execute")
    acknowledged = _boolean(
        parameters["acknowledge_teacher_motion"],
        "acknowledge_teacher_motion",
    )
    matrix_id = _identifier(parameters["matrix_id"], "matrix_id").lower()
    parent_id = matrix_id if execute else f"{matrix_id}-plan"
    output = _unused_output(workspace / "runs" / parent_id, "shadow matrix")
    source_payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(source_payload, Mapping):
        raise ValueError("shadow matrix configuration must contain an object")
    source_config = ShadowMatrixConfig.from_mapping(
        source_payload,
        base=config.parent,
    )
    if execute and source_config.control == "teacher" and not acknowledged:
        raise PermissionError("live teacher matrix execution requires acknowledgement")
    derived_config, _ = _derived_shadow_config(
        workspace,
        config,
        matrix_id,
    )
    tokens = [
        "--config",
        str(derived_config),
        "--runs-root",
        str(workspace / "runs"),
    ]
    if execute:
        tokens.append("--execute")
        if acknowledged:
            tokens.append("--acknowledge-teacher-motion")
    return CommandPlan(
        kind="shadow_matrix",
        title=f"Vision shadow matrix {matrix_id}",
        command=_module_command(executable, "carla_vision.shadow.matrix", *tokens),
        expected_output=output,
        motion_authorized=execute and acknowledged,
        destructive=False,
        note=(
            "Execution may move the configured vehicle through the privileged teacher."
            if execute
            else "Plan mode is offline and does not contact CARLA."
        ),
    )


def _build_analyze(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={"source_run", "run_id"},
        name="analysis job",
    )
    source = _workspace_path(
        workspace,
        parameters["source_run"],
        "source_run",
        kind="directory",
    )
    run_id = _identifier(parameters["run_id"], "run_id")
    output = _unused_output(workspace / "runs" / run_id, "analysis run")
    return CommandPlan(
        kind="analyze",
        title=f"Post-hoc analysis {run_id}",
        command=_module_command(
            executable,
            "carla_vision.analysis",
            "--source-run",
            str(source),
            "--runs-root",
            str(workspace / "runs"),
            "--run-id",
            run_id,
        ),
        expected_output=output,
        motion_authorized=False,
        destructive=False,
        note="Read-only analysis creates a new source-linked artifact.",
    )


def _build_verify(
    parameters: Mapping[str, Any],
    workspace: Path,
    executable: str,
) -> CommandPlan:
    _keys(
        parameters,
        required={"paths", "allow_non_success", "require_clean_git"},
        name="verification job",
    )
    raw_paths = parameters["paths"]
    if isinstance(raw_paths, (str, bytes)) or not isinstance(raw_paths, Sequence):
        raise TypeError("verification paths must be an array")
    if not 1 <= len(raw_paths) <= 25:
        raise ValueError("verification requires 1-25 object paths")
    paths = [
        _workspace_path(workspace, path, f"paths[{index}]", kind="directory")
        for index, path in enumerate(raw_paths)
    ]
    allow_non_success = _boolean(parameters["allow_non_success"], "allow_non_success")
    require_clean_git = _boolean(parameters["require_clean_git"], "require_clean_git")
    tokens = [str(path) for path in paths]
    tokens.append("--reject-unregistered")
    if require_clean_git:
        tokens.append("--require-clean-git")
    if allow_non_success:
        tokens.append("--allow-non-success")
    return CommandPlan(
        kind="verify",
        title=f"Verify {len(paths)} research object(s)",
        command=_module_command(executable, "carla_vision.verification", *tokens),
        expected_output=None,
        motion_authorized=False,
        destructive=False,
        note="Verification is read-only and checks registered hashes and semantic contracts.",
    )


_BUILDERS = {
    "analyze": _build_analyze,
    "dataset_qa": _build_dataset_qa,
    "live": _build_live,
    "native_capture": _build_native_capture,
    "native_preflight": _build_native_preflight,
    "replay": _build_replay,
    "scenario_plan": _build_scenario_plan,
    "shadow_matrix": _build_shadow_matrix,
    "train": _build_train,
    "verify": _build_verify,
}


def build_command_plan(
    request: OperatorJobRequest,
    *,
    workspace: str | Path,
    executable: str = sys.executable,
) -> CommandPlan:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("operator workspace must be a directory")
    return _BUILDERS[request.kind](request.parameters, root, executable)


__all__ = ["CommandPlan", "build_command_plan"]
