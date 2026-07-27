"""Read-only, artifact-producing readiness check for native CARLA collection."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import socket
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import RunArtifactTracker
from ..bridge import CarlaRpc
from ..scenarios.splits import canonical_map_family
from ..scenarios.verified_plan import load_verified_scenario_plan
from .worker import _load_carla_module, select_episodes

NATIVE_PREFLIGHT_SCHEMA_VERSION = "1.0"


def _atomic_text(path: Path, payload: str) -> None:
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
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _safe_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__qualname__,
        "module": type(error).__module__,
        "message": str(error)[:1000],
    }


def _capture_count(episode: Any) -> int:
    capture = episode.recipe.capture
    return 1 + (capture.duration_ticks - 1) // capture.capture_every_ticks


def _add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    label: str,
    category: str,
    required: bool,
    passed: bool,
    observed: Any,
    expected: Any,
    note: str,
    status: str | None = None,
) -> None:
    resolved_status = status or ("pass" if passed else "fail")
    if resolved_status not in {"pass", "fail", "skip", "pending", "warn"}:
        raise ValueError(f"invalid preflight check status {resolved_status!r}")
    checks.append(
        {
            "check_id": check_id,
            "label": label,
            "category": category,
            "required": required,
            "passed": passed,
            "status": resolved_status,
            "observed": observed,
            "expected": expected,
            "note": note,
        }
    )


def _socket_probe(host: str, port: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {
                "reachable": True,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "error": None,
            }
    except OSError as error:
        return {
            "reachable": False,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": _safe_error(error),
        }


def _bridge_probe(host: str, port: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with CarlaRpc(host, port, timeout=timeout) as rpc:
            version = str(rpc.value_call("version"))
            map_info = rpc.value_call("get_map_info")
        if not isinstance(map_info, Sequence) or isinstance(map_info, (str, bytes)):
            raise RuntimeError("CARLA get_map_info returned an invalid payload")
        return {
            "succeeded": True,
            "server_version": version,
            "map_name": str(map_info[0]),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": None,
            "operations": ["version", "get_map_info"],
        }
    except BaseException as error:
        return {
            "succeeded": False,
            "server_version": None,
            "map_name": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": _safe_error(error),
            "operations": ["version", "get_map_info"],
        }


def _native_pythonapi_probe(
    *,
    host: str,
    port: int,
    timeout: float,
    python_api_path: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "importable": False,
        "module_path": None,
        "package_version": None,
        "client_connected": False,
        "client_version": None,
        "server_version": None,
        "map_name": None,
        "world_synchronous_mode": None,
        "world_fixed_delta_seconds": None,
        "error": None,
        "read_only_operations": [
            "Client",
            "get_client_version",
            "get_server_version",
            "get_world",
            "get_map",
            "get_settings",
        ],
    }
    try:
        carla = _load_carla_module(python_api_path)
        result["importable"] = True
        result["module_path"] = str(getattr(carla, "__file__", "") or "")
        package_version = getattr(carla, "__version__", None)
        result["package_version"] = None if package_version is None else str(package_version)

        client = carla.Client(host, port)
        client.set_timeout(timeout)
        result["client_version"] = str(client.get_client_version())
        result["server_version"] = str(client.get_server_version())
        world = client.get_world()
        result["map_name"] = str(world.get_map().name)
        settings = world.get_settings()
        result["world_synchronous_mode"] = bool(settings.synchronous_mode)
        fixed_delta = settings.fixed_delta_seconds
        result["world_fixed_delta_seconds"] = None if fixed_delta is None else float(fixed_delta)
        result["client_connected"] = True
    except BaseException as error:
        result["error"] = _safe_error(error)
    return result


def _checks(
    *,
    plan: Any,
    episodes: Sequence[Any],
    dataset_dir: Path,
    socket_probe: Mapping[str, Any],
    bridge_probe: Mapping[str, Any],
    native_probe: Mapping[str, Any],
    confirm_world_reload: bool,
    confirm_exclusive_tick_owner: bool,
) -> list[dict[str, Any]]:
    expected_version = plan.suite.carla_version
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        check_id="scenario_plan_verified",
        label="Scenario plan integrity and deterministic recomputation",
        category="automated",
        required=True,
        passed=True,
        observed=plan.run_id,
        expected="verified scenario plan",
        note="The plan loader verified hashes and recomputed all episode rows.",
    )
    _add_check(
        checks,
        check_id="episode_selection_nonempty",
        label="Selected pilot episodes",
        category="automated",
        required=True,
        passed=bool(episodes),
        observed=len(episodes),
        expected=">=1",
        note="Only the selected episodes would be collected.",
    )
    output_available = not dataset_dir.exists()
    _add_check(
        checks,
        check_id="dataset_output_id_available",
        label="Dataset output ID is unused",
        category="automated",
        required=True,
        passed=output_available,
        observed=str(dataset_dir),
        expected="path does not exist",
        note="Native collection never overwrites an existing dataset.",
    )
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    _add_check(
        checks,
        check_id="project_python_supported",
        label="Python version satisfies the project contract",
        category="automated",
        required=True,
        passed=supported_python,
        observed=platform.python_version(),
        expected=">=3.11,<3.14",
        note="The packaged CARLA wheel must additionally match this interpreter and platform.",
    )
    official_binary_platform = platform.machine().lower() in {
        "amd64",
        "x86_64",
    } and platform.system() in {"Linux", "Windows"}
    _add_check(
        checks,
        check_id="official_binary_platform",
        label="Host matches official CARLA binary platforms",
        category="advisory",
        required=False,
        passed=official_binary_platform,
        observed=f"{platform.system()} {platform.machine()}",
        expected="Linux x86_64 or Windows x86_64",
        note="A custom compatible PythonAPI build may satisfy the native checks.",
        status="pass" if official_binary_platform else "warn",
    )
    socket_ok = bool(socket_probe.get("reachable"))
    _add_check(
        checks,
        check_id="carla_rpc_tcp_reachable",
        label="CARLA RPC TCP endpoint",
        category="automated",
        required=True,
        passed=socket_ok,
        observed=socket_probe,
        expected="reachable",
        note="This opens and closes one TCP connection without issuing a simulator command.",
    )
    bridge_ok = bool(bridge_probe.get("succeeded"))
    _add_check(
        checks,
        check_id="carla_read_only_rpc",
        label="CARLA read-only version and map RPC",
        category="automated",
        required=True,
        passed=bridge_ok,
        observed=bridge_probe,
        expected={"operations": ["version", "get_map_info"]},
        note="No actor, world, weather, setting, or control mutation is requested.",
        status="pass" if bridge_ok else ("skip" if not socket_ok else "fail"),
    )
    bridge_version_ok = bridge_probe.get("server_version") == expected_version
    _add_check(
        checks,
        check_id="bridge_server_version_match",
        label="Server version matches the scenario plan",
        category="automated",
        required=True,
        passed=bridge_version_ok,
        observed=bridge_probe.get("server_version"),
        expected=expected_version,
        note="Client and server version coupling is mandatory for native collection.",
        status="pass" if bridge_version_ok else ("skip" if not bridge_ok else "fail"),
    )
    native_importable = bool(native_probe.get("importable"))
    _add_check(
        checks,
        check_id="native_pythonapi_importable",
        label="Official version-matched CARLA PythonAPI",
        category="automated",
        required=True,
        passed=native_importable,
        observed={
            "module_path": native_probe.get("module_path"),
            "package_version": native_probe.get("package_version"),
            "error": native_probe.get("error"),
        },
        expected=f"importable CARLA {expected_version} client",
        note="The raw bridge is sufficient for live perception but not for deterministic capture.",
    )
    native_connected = bool(native_probe.get("client_connected"))
    _add_check(
        checks,
        check_id="native_pythonapi_read_only_connection",
        label="PythonAPI read-only client connection",
        category="automated",
        required=True,
        passed=native_connected,
        observed={
            "client_version": native_probe.get("client_version"),
            "server_version": native_probe.get("server_version"),
            "map_name": native_probe.get("map_name"),
            "error": native_probe.get("error"),
        },
        expected="connection and read-only world query succeed",
        note="The probe does not load a world or change synchronous settings.",
        status="pass" if native_connected else ("skip" if not native_importable else "fail"),
    )
    native_client_version_ok = native_probe.get("client_version") == expected_version
    _add_check(
        checks,
        check_id="native_client_version_match",
        label="PythonAPI client version",
        category="automated",
        required=True,
        passed=native_client_version_ok,
        observed=native_probe.get("client_version"),
        expected=expected_version,
        note="The client library must match the frozen scenario plan.",
        status=(
            "pass" if native_client_version_ok else ("skip" if not native_connected else "fail")
        ),
    )
    native_server_version_ok = native_probe.get("server_version") == expected_version
    _add_check(
        checks,
        check_id="native_server_version_match",
        label="Server version observed through PythonAPI",
        category="automated",
        required=True,
        passed=native_server_version_ok,
        observed=native_probe.get("server_version"),
        expected=expected_version,
        note="This is checked again immediately before a real dataset writer opens.",
        status=(
            "pass" if native_server_version_ok else ("skip" if not native_connected else "fail")
        ),
    )
    _add_check(
        checks,
        check_id="world_reload_authorized",
        label="Operator confirms the current world may be reloaded",
        category="manual",
        required=True,
        passed=confirm_world_reload,
        observed=confirm_world_reload,
        expected=True,
        note="A real episode destroys existing actors when its map is loaded or reloaded.",
        status="pass" if confirm_world_reload else "pending",
    )
    _add_check(
        checks,
        check_id="exclusive_tick_owner_confirmed",
        label="Operator confirms exclusive world.tick ownership",
        category="manual",
        required=True,
        passed=confirm_exclusive_tick_owner,
        observed=confirm_exclusive_tick_owner,
        expected=True,
        note="No other synchronous client may tick the world during native capture.",
        status="pass" if confirm_exclusive_tick_owner else "pending",
    )
    return checks


def _check_csv(checks: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "check_id",
        "label",
        "category",
        "required",
        "passed",
        "status",
        "observed",
        "expected",
        "note",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for check in checks:
        writer.writerow(
            {
                **check,
                "observed": json.dumps(check["observed"], ensure_ascii=False, sort_keys=True),
                "expected": json.dumps(check["expected"], ensure_ascii=False, sort_keys=True),
            }
        )
    return buffer.getvalue()


def _report(summary: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(
        "| {status} | {category} | {label} | {observed} |".format(
            status=check["status"],
            category=check["category"],
            label=check["label"],
            observed=str(check["observed"]).replace("|", "\\|"),
        )
        for check in checks
    )
    readiness = "READY" if summary["ready_for_native_execution"] else "NOT READY"
    return f"""# Native CARLA collection preflight

Result: **{readiness}**  
Run ID: `{summary["run_id"]}`  
Scenario plan: `{summary["scenario_plan"]["run_id"]}`  
Dataset ID: `{summary["dataset"]["dataset_id"]}`  
Endpoint: `{summary["endpoint"]["host"]}:{summary["endpoint"]["port"]}`  
Selected episodes: {summary["selection"]["selected_episode_count"]}  
Planned captures: {summary["selection"]["planned_capture_count"]}

This preflight is read-only. It did not load a map, change world settings,
spawn or destroy actors, call `world.tick()`, apply control, or create a
dataset.

| Status | Category | Check | Observed |
|---|---|---|---|
{rows}

## Readiness interpretation

- `automated_ready` requires every automated execution prerequisite to pass.
- `manual_ready` requires explicit world-reload and exclusive-tick-owner
  confirmations.
- `ready_for_native_execution` is true only when both are true.
- A successful preflight process means the assessment artifact was created;
  it does not imply that native collection is ready.
"""


def run_preflight(
    *,
    scenario_plan: str | Path,
    dataset_id: str,
    datasets_root: str | Path,
    runs_root: str | Path,
    run_id: str,
    host: str,
    port: int,
    timeout: float,
    python_api_path: str | None,
    episode_ids: Sequence[str],
    partitions: Sequence[str],
    max_episodes: int | None,
    confirm_world_reload: bool,
    confirm_exclusive_tick_owner: bool,
    repository_root: str | Path | None = None,
    cli_args: Sequence[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = load_verified_scenario_plan(scenario_plan)
    episodes = select_episodes(
        plan,
        episode_ids=episode_ids,
        partitions=partitions,
        max_episodes=max_episodes,
    )
    dataset_dir = (Path(datasets_root).expanduser().resolve() / dataset_id).resolve()
    socket_result = _socket_probe(host, port, timeout)
    bridge_result = _bridge_probe(host, port, timeout)
    native_result = _native_pythonapi_probe(
        host=host,
        port=port,
        timeout=timeout,
        python_api_path=python_api_path,
    )
    checks = _checks(
        plan=plan,
        episodes=episodes,
        dataset_dir=dataset_dir,
        socket_probe=socket_result,
        bridge_probe=bridge_result,
        native_probe=native_result,
        confirm_world_reload=confirm_world_reload,
        confirm_exclusive_tick_owner=confirm_exclusive_tick_owner,
    )
    automated_required = [
        check for check in checks if check["category"] == "automated" and check["required"]
    ]
    manual_required = [
        check for check in checks if check["category"] == "manual" and check["required"]
    ]
    automated_ready = all(bool(check["passed"]) for check in automated_required)
    manual_ready = all(bool(check["passed"]) for check in manual_required)
    capture_count = sum(_capture_count(episode) for episode in episodes)
    selected_payload = {
        "schema_version": NATIVE_PREFLIGHT_SCHEMA_VERSION,
        "object_type": "native_preflight_selection",
        "scenario_plan": dict(plan.reference),
        "selection_request": {
            "episode_ids": list(episode_ids),
            "partitions": list(partitions),
            "max_episodes": max_episodes,
        },
        "selected_episode_count": len(episodes),
        "planned_capture_count": capture_count,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "partition": episode.split.partition,
                "recipe_id": episode.recipe.recipe_id,
                "map_name": episode.recipe.map_name,
                "map_family": canonical_map_family(episode.recipe.map_name),
                "weather_id": episode.recipe.weather.weather_id,
                "scenario_seed": episode.seeds.values["world"],
                "capture_count": _capture_count(episode),
            }
            for episode in episodes
        ],
    }
    summary: dict[str, Any] = {
        "schema_version": NATIVE_PREFLIGHT_SCHEMA_VERSION,
        "object_type": "native_collection_preflight",
        "run_id": run_id,
        "status": "ready" if automated_ready and manual_ready else "not_ready",
        "read_only": True,
        "simulator_contacted": bool(socket_result["reachable"]),
        "simulator_mutated": False,
        "automated_ready": automated_ready,
        "manual_ready": manual_ready,
        "ready_for_native_execution": automated_ready and manual_ready,
        "scenario_plan": {
            **dict(plan.reference),
            "suite_id": plan.suite.suite_id,
            "carla_version": plan.suite.carla_version,
        },
        "dataset": {
            "dataset_id": dataset_id,
            "path": str(dataset_dir),
            "output_available": not dataset_dir.exists(),
        },
        "endpoint": {
            "host": host,
            "port": port,
            "timeout_seconds": timeout,
            "socket": socket_result,
            "bridge_read_only_rpc": bridge_result,
            "native_pythonapi": native_result,
        },
        "selection": {
            "selected_episode_count": len(episodes),
            "planned_capture_count": capture_count,
            "partition_counts": dict(
                sorted(Counter(episode.split.partition for episode in episodes).items())
            ),
            "map_family_counts": dict(
                sorted(
                    Counter(
                        canonical_map_family(episode.recipe.map_name) for episode in episodes
                    ).items()
                )
            ),
        },
        "checks": {
            "count": len(checks),
            "passed": sum(check["status"] == "pass" for check in checks),
            "failed": sum(check["status"] == "fail" for check in checks),
            "pending": sum(check["status"] == "pending" for check in checks),
            "skipped": sum(check["status"] == "skip" for check in checks),
            "warnings": sum(check["status"] == "warn" for check in checks),
        },
    }
    tracker = RunArtifactTracker(
        runs_root,
        run_id=run_id,
        cli_args=cli_args or (),
        config={
            "schema_version": NATIVE_PREFLIGHT_SCHEMA_VERSION,
            "object_type": "native_collection_preflight",
            "scenario_plan": str(Path(scenario_plan).expanduser().resolve()),
            "dataset_id": dataset_id,
            "datasets_root": str(Path(datasets_root).expanduser().resolve()),
            "host": host,
            "port": port,
            "timeout": timeout,
            "python_api_path": python_api_path,
            "episode_ids": list(episode_ids),
            "partitions": list(partitions),
            "max_episodes": max_episodes,
            "confirm_world_reload": confirm_world_reload,
            "confirm_exclusive_tick_owner": confirm_exclusive_tick_owner,
            "read_only": True,
        },
        repository_root=repository_root,
        carla_endpoint={"host": host, "port": port},
        carla_version=bridge_result.get("server_version"),
        carla_map=bridge_result.get("map_name"),
        input_refs=[dict(plan.reference)],
    )
    summary["run_id"] = tracker.run_id
    with tracker:
        summary_path = tracker.artifact_path("summary.json")
        checks_path = tracker.artifact_path("checks.json")
        checks_csv_path = tracker.artifact_path("checks.csv")
        selection_path = tracker.artifact_path("selected_episodes.json")
        report_path = tracker.artifact_path("report.md")
        _write_json(summary_path, summary)
        _write_json(
            checks_path,
            {
                "schema_version": NATIVE_PREFLIGHT_SCHEMA_VERSION,
                "object_type": "native_preflight_checks",
                "checks": checks,
            },
        )
        _atomic_text(checks_csv_path, _check_csv(checks))
        _write_json(selection_path, selected_payload)
        _atomic_text(report_path, _report(summary, checks))
        metadata = {
            "ready_for_native_execution": summary["ready_for_native_execution"],
            "selected_episode_count": len(episodes),
            "planned_capture_count": capture_count,
        }
        tracker.register_artifact(
            checks_csv_path,
            role="native_preflight_check_table",
            metadata=metadata,
        )
        tracker.register_artifact(
            checks_path,
            role="native_preflight_checks",
            metadata=metadata,
        )
        tracker.register_artifact(
            report_path,
            role="native_preflight_report",
            metadata=metadata,
        )
        tracker.register_artifact(
            selection_path,
            role="native_preflight_selected_episodes",
            metadata=metadata,
        )
        tracker.register_artifact(
            summary_path,
            role="native_preflight_summary",
            metadata=metadata,
        )
    result = {
        "run_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "manifest": str(tracker.manifest_path),
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a read-only, checksum-tracked readiness assessment for "
            "official-PythonAPI native CARLA dataset collection"
        )
    )
    parser.add_argument("--scenario-plan", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--carla-python-api")
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--confirm-world-reload", action="store_true")
    parser.add_argument("--confirm-exclusive-tick-owner", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not 0.1 <= args.timeout <= 300.0:
        parser.error("--timeout must be in [0.1, 300]")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_preflight(
        scenario_plan=args.scenario_plan,
        dataset_id=args.dataset_id,
        datasets_root=args.datasets_root,
        runs_root=args.runs_root,
        run_id=args.run_id,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        python_api_path=args.carla_python_api,
        episode_ids=args.episode_id,
        partitions=args.partition,
        max_episodes=args.max_episodes,
        confirm_world_reload=args.confirm_world_reload,
        confirm_exclusive_tick_owner=args.confirm_exclusive_tick_owner,
        cli_args=vars(args),
    )
    return 0


__all__ = [
    "NATIVE_PREFLIGHT_SCHEMA_VERSION",
    "main",
    "parse_args",
    "run_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
