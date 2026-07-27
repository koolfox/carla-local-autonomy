from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from carla_vision.native.preflight import run_preflight
from carla_vision.native.verified_preflight import load_verified_native_preflight
from carla_vision.scenarios.planner import plan_scenarios
from carla_vision.verification import verify_research_object

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "thesis_pilot_v1.json"
SPLIT_PATH = REPOSITORY_ROOT / "configs" / "scenarios" / "split_plan_thesis_pilot_v1.json"


def make_plan(root: Path) -> Path:
    result = plan_scenarios(
        suite_path=SUITE_PATH,
        split_plan_path=SPLIT_PATH,
        runs_root=root / "source-runs",
        run_id="preflight-source-plan",
        repository_root=REPOSITORY_ROOT,
    )
    return Path(result["run_dir"])


def socket_result() -> dict[str, object]:
    return {"reachable": True, "latency_ms": 1.2, "error": None}


def bridge_result() -> dict[str, object]:
    return {
        "succeeded": True,
        "server_version": "0.9.16",
        "map_name": "Town10HD_Opt",
        "latency_ms": 2.4,
        "error": None,
        "operations": ["version", "get_map_info"],
    }


def native_result(*, ready: bool) -> dict[str, object]:
    return {
        "importable": ready,
        "module_path": "/opt/carla/carla.so" if ready else None,
        "package_version": "0.9.16" if ready else None,
        "client_connected": ready,
        "client_version": "0.9.16" if ready else None,
        "server_version": "0.9.16" if ready else None,
        "map_name": "Town10HD_Opt" if ready else None,
        "world_synchronous_mode": False if ready else None,
        "world_fixed_delta_seconds": None,
        "error": None if ready else {"type": "ImportError", "message": "carla missing"},
        "read_only_operations": [
            "Client",
            "get_client_version",
            "get_server_version",
            "get_world",
            "get_map",
            "get_settings",
        ],
    }


class NativePreflightTests(unittest.TestCase):
    def _run(self, root: Path, *, native_ready: bool, manual_ready: bool) -> Path:
        plan = make_plan(root)
        with (
            patch(
                "carla_vision.native.preflight._socket_probe",
                return_value=socket_result(),
            ),
            patch(
                "carla_vision.native.preflight._bridge_probe",
                return_value=bridge_result(),
            ),
            patch(
                "carla_vision.native.preflight._native_pythonapi_probe",
                return_value=native_result(ready=native_ready),
            ),
            patch("sys.stdout", new=io.StringIO()),
        ):
            result = run_preflight(
                scenario_plan=plan,
                dataset_id="ds-native-pilot",
                datasets_root=root / "datasets",
                runs_root=root / "runs",
                run_id="native-preflight-test",
                host="172.20.10.7",
                port=2000,
                timeout=3.0,
                python_api_path=None,
                episode_ids=(),
                partitions=("train",),
                max_episodes=1,
                confirm_world_reload=manual_ready,
                confirm_exclusive_tick_owner=manual_ready,
                repository_root=REPOSITORY_ROOT,
            )
        return Path(result["run_dir"])

    def test_missing_pythonapi_is_retained_as_successful_not_ready_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight_dir = self._run(root, native_ready=False, manual_ready=False)
            preflight = load_verified_native_preflight(preflight_dir)
            generic = verify_research_object(
                preflight_dir,
                reject_unregistered=True,
            )
            self.assertFalse(preflight.summary["automated_ready"])
            self.assertFalse(preflight.summary["manual_ready"])
            self.assertFalse(preflight.summary["ready_for_native_execution"])
            self.assertFalse(preflight.summary["simulator_mutated"])
            self.assertEqual(
                preflight.selected_episodes["selected_episode_count"],
                1,
            )
            self.assertEqual(
                preflight.selected_episodes["planned_capture_count"],
                150,
            )
            self.assertEqual(
                generic.deep_verification["kind"],
                "native_collection_preflight",
            )
            self.assertEqual(generic.external_reference_count, 1)
            self.assertEqual(generic.unregistered_file_count, 0)

    def test_matching_pythonapi_and_manual_gates_produce_ready_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preflight_dir = self._run(
                Path(temporary),
                native_ready=True,
                manual_ready=True,
            )
            preflight = load_verified_native_preflight(preflight_dir)
            self.assertTrue(preflight.summary["automated_ready"])
            self.assertTrue(preflight.summary["manual_ready"])
            self.assertTrue(preflight.summary["ready_for_native_execution"])
            self.assertEqual(preflight.summary["status"], "ready")
            self.assertEqual(preflight.summary["checks"]["failed"], 0)
            self.assertEqual(preflight.summary["checks"]["pending"], 0)


if __name__ == "__main__":
    unittest.main()
