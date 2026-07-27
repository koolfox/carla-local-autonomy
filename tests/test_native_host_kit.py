from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from carla_vision.native.host_gate import (
    NativeHostGateError,
    validate_ready_preflight,
)
from carla_vision.native.host_kit import (
    NativeHostKitConfig,
    build_native_host_kit,
)
from carla_vision.native.preflight import run_preflight
from carla_vision.native.verified_host_kit import load_verified_native_host_kit
from carla_vision.native.verified_preflight import load_verified_native_preflight
from carla_vision.scenarios.planner import plan_scenarios
from carla_vision.verification import ArtifactIntegrityError, verify_research_object

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "native_integration_pilot_v1.json"
SPLIT_PATH = (
    REPOSITORY_ROOT / "configs" / "scenarios" / "split_plan_native_integration_pilot_v1.json"
)
WINDOWS_REQUIREMENTS = (
    REPOSITORY_ROOT / "configs" / "native_host" / "requirements-windows-py312.txt"
)
LINUX_REQUIREMENTS = REPOSITORY_ROOT / "configs" / "native_host" / "requirements-linux-py312.txt"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_plan(root: Path) -> Path:
    result = plan_scenarios(
        suite_path=SUITE_PATH,
        split_plan_path=SPLIT_PATH,
        runs_root=root / "source-runs",
        run_id="portable-kit-source-plan",
        repository_root=REPOSITORY_ROOT,
    )
    return Path(result["run_dir"])


def _make_config(root: Path, plan: Path) -> Path:
    path = root / "native-kit.json"
    _write_json(
        path,
        {
            "schema_version": "1.0",
            "kit_id": "native-host-kit-test-v1",
            "title": "Native host kit test",
            "scenario_plan": str(plan),
            "dataset_id": "ds-native-host-kit-test",
            "host": "172.20.10.7",
            "port": 2000,
            "partitions": ["train"],
            "max_episodes": 1,
            "python_version": "3.12",
            "carla_version": "0.9.16",
            "requirements": {
                "windows": str(WINDOWS_REQUIREMENTS),
                "linux": str(LINUX_REQUIREMENTS),
            },
        },
    )
    return path


def _socket_result() -> dict[str, object]:
    return {"reachable": True, "latency_ms": 1.0, "error": None}


def _bridge_result() -> dict[str, object]:
    return {
        "succeeded": True,
        "server_version": "0.9.16",
        "map_name": "Town10HD_Opt",
        "latency_ms": 2.0,
        "error": None,
        "operations": ["version", "get_map_info"],
    }


def _native_result() -> dict[str, object]:
    return {
        "importable": True,
        "module_path": "C:/carla/carla.pyd",
        "package_version": "0.9.16",
        "client_connected": True,
        "client_version": "0.9.16",
        "server_version": "0.9.16",
        "map_name": "Carla/Maps/Town10HD_Opt",
        "world_synchronous_mode": False,
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


class NativeHostKitTests(unittest.TestCase):
    def test_config_is_intentionally_limited_to_one_train_episode(self) -> None:
        base = {
            "schema_version": "1.0",
            "kit_id": "native-host-kit-test",
            "title": "Kit",
            "scenario_plan": "plan",
            "dataset_id": "dataset",
            "host": "127.0.0.1",
            "port": 2000,
            "partitions": ["train"],
            "max_episodes": 1,
            "python_version": "3.12",
            "carla_version": "0.9.16",
            "requirements": {"windows": "w.txt", "linux": "l.txt"},
        }
        config = NativeHostKitConfig.from_mapping(base)
        self.assertEqual(config.partitions, ("train",))
        changed = dict(base)
        changed["max_episodes"] = 2
        with self.assertRaisesRegex(ValueError, "one train episode"):
            NativeHostKitConfig.from_mapping(changed)

    def test_built_kit_is_standalone_deterministic_and_dry_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _make_plan(root)
            config = _make_config(root, plan)
            first = build_native_host_kit(
                config_path=config,
                kits_root=root / "kits-a",
                repository_root=REPOSITORY_ROOT,
            )
            second = build_native_host_kit(
                config_path=config,
                kits_root=root / "kits-b",
                repository_root=REPOSITORY_ROOT,
            )
            first_root = Path(first["kit_dir"])
            second_root = Path(second["kit_dir"])
            verified = load_verified_native_host_kit(first_root)
            generic = verify_research_object(
                first_root,
                reject_unregistered=True,
            )
            self.assertEqual(verified.kit_id, "native-host-kit-test-v1")
            self.assertEqual(
                verified.descriptor["selection"]["planned_capture_count"],
                50,
            )
            self.assertFalse(verified.descriptor["generation"]["simulator_mutated"])
            self.assertEqual(generic.deep_verification["kind"], "native_host_kit")
            self.assertEqual(generic.external_reference_count, 0)
            self.assertEqual(generic.unregistered_file_count, 0)
            self.assertEqual(
                (first_root / "payload" / "native-host-kit.zip").read_bytes(),
                (second_root / "payload" / "native-host-kit.zip").read_bytes(),
            )

            extraction = root / "extracted"
            with zipfile.ZipFile(
                first_root / "payload" / "native-host-kit.zip",
                "r",
            ) as archive:
                archive.extractall(extraction)
            verify_process = subprocess.run(
                [sys.executable, str(extraction / "scripts" / "verify_payload.py")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify_process.returncode, 0, verify_process.stderr)
            self.assertIn("payload verified", verify_process.stdout)
            for script in (
                "bootstrap.sh",
                "preflight.sh",
                "collect.sh",
                "postrun.sh",
            ):
                syntax = subprocess.run(
                    ["bash", "-n", str(extraction / "scripts" / script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)

            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(extraction / "source")
            dry_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carla_vision.native.worker",
                    "--scenario-plan",
                    str(extraction / "scenario-plan" / "portable-kit-source-plan"),
                    "--dataset-id",
                    "ds-native-host-kit-test",
                    "--partition",
                    "train",
                    "--max-episodes",
                    "1",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            summary = json.loads(dry_run.stdout)
            self.assertEqual(summary["planned_capture_count"], 50)
            self.assertTrue(summary["would_mutate_simulator"])

            plan.rename(root / "source-plan-removed")
            standalone = load_verified_native_host_kit(first_root)
            self.assertEqual(standalone.kit_id, "native-host-kit-test-v1")

    def test_archive_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _make_plan(root)
            config = _make_config(root, plan)
            result = build_native_host_kit(
                config_path=config,
                kits_root=root / "kits",
                repository_root=REPOSITORY_ROOT,
            )
            kit_root = Path(result["kit_dir"])
            archive = kit_root / "payload" / "native-host-kit.zip"
            archive.write_bytes(archive.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ArtifactIntegrityError, "fingerprint differs"):
                load_verified_native_host_kit(kit_root)


class NativeHostGateTests(unittest.TestCase):
    def _ready_preflight(self, root: Path, plan: Path) -> Path:
        with (
            patch(
                "carla_vision.native.preflight._socket_probe",
                return_value=_socket_result(),
            ),
            patch(
                "carla_vision.native.preflight._bridge_probe",
                return_value=_bridge_result(),
            ),
            patch(
                "carla_vision.native.preflight._native_pythonapi_probe",
                return_value=_native_result(),
            ),
            patch("sys.stdout", new=io.StringIO()),
        ):
            result = run_preflight(
                scenario_plan=plan,
                dataset_id="ds-native-host-kit-test",
                datasets_root=root / "datasets",
                runs_root=root / "runs",
                run_id="ready-preflight",
                host="172.20.10.7",
                port=2000,
                timeout=3.0,
                python_api_path=None,
                episode_ids=(),
                partitions=("train",),
                max_episodes=1,
                confirm_world_reload=True,
                confirm_exclusive_tick_owner=True,
                repository_root=REPOSITORY_ROOT,
            )
        return Path(result["run_dir"])

    def test_ready_preflight_must_match_kit_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _make_plan(root)
            preflight_path = self._ready_preflight(root, plan)
            preflight = load_verified_native_preflight(preflight_path)
            selected = preflight.selected_episodes
            kit_plan_path = root / "kit-plan.json"
            _write_json(
                kit_plan_path,
                {
                    "schema_version": "1.0",
                    "object_type": "native_host_kit_payload",
                    "kit_id": "native-host-kit-test-v1",
                    "dataset": {"dataset_id": "ds-native-host-kit-test"},
                    "endpoint": {"host": "172.20.10.7", "port": 2000},
                    "scenario_plan": {
                        key: selected["scenario_plan"][key]
                        for key in ("run_id", "sha256", "size_bytes")
                    },
                    "selection": {
                        "episode_ids": [episode["episode_id"] for episode in selected["episodes"]],
                        "planned_capture_count": selected["planned_capture_count"],
                    },
                },
            )
            result = validate_ready_preflight(
                kit_plan_path=kit_plan_path,
                preflight_path=preflight_path,
            )
            self.assertEqual(result["status"], "authorized")
            self.assertEqual(result["planned_capture_count"], 50)

            changed = json.loads(kit_plan_path.read_text(encoding="utf-8"))
            changed["dataset"]["dataset_id"] = "different-dataset"
            _write_json(kit_plan_path, changed)
            with self.assertRaisesRegex(NativeHostGateError, "dataset ID differs"):
                validate_ready_preflight(
                    kit_plan_path=kit_plan_path,
                    preflight_path=preflight_path,
                )

    def test_current_not_ready_preflight_cannot_authorize_collection(self) -> None:
        kit_root = REPOSITORY_ROOT / "native_kits" / "native-host-kit-pilot-20260727-v001"
        if not kit_root.is_dir():
            self.skipTest("retained development native host kit is absent")
        with tempfile.TemporaryDirectory() as temporary:
            extraction = Path(temporary)
            with zipfile.ZipFile(
                kit_root / "payload" / "native-host-kit.zip",
                "r",
            ) as archive:
                archive.extract("kit-plan.json", extraction)
            with self.assertRaisesRegex(NativeHostGateError, "not ready"):
                validate_ready_preflight(
                    kit_plan_path=extraction / "kit-plan.json",
                    preflight_path=(
                        REPOSITORY_ROOT / "runs" / "native-preflight-pilot-20260726-v1"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
