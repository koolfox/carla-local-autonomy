from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from carla_vision.operator.catalog import build_catalog
from carla_vision.operator.commands import CommandPlan, build_command_plan
from carla_vision.operator.contracts import OperatorJobRequest
from carla_vision.operator.jobs import JobManager
from carla_vision.operator.server import create_server
from carla_vision.operator.situations import (
    SituationSpec,
    build_scenario_suite,
    save_situation_suite,
)
from carla_vision.verification import verify_research_object


def valid_situation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "situation_id": "town10-ui-poc",
        "map_name": "Town10HD_Opt",
        "weather_preset": "soft-rain-sunset",
        "vehicle_count": 35,
        "walker_count": 30,
        "pedestrian_crossing_factor": 0.1,
        "speed_difference_percent": 20.0,
        "following_distance_metres": 2.5,
        "prop_preset": "construction",
        "ego_blueprint": "vehicle.tesla.model3",
        "ego_spawn_index": 0,
        "duration_seconds": 30,
        "capture_fps": 5,
        "repetitions": 2,
        "master_seed": 20260726,
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fov": 90.0,
    }
    payload.update(overrides)
    return payload


def request(kind: str, parameters: dict[str, object]) -> OperatorJobRequest:
    return OperatorJobRequest.from_mapping(
        {
            "schema_version": "1.0",
            "kind": kind,
            "parameters": parameters,
        }
    )


def write_research_object(
    root: Path,
    root_kind: str,
    object_id: str,
    artifacts: dict[str, tuple[str, bytes | None, str]],
) -> Path:
    directory = root / root_kind / object_id
    directory.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    for relative, (role, content, mime_type) in artifacts.items():
        declared = content if content is not None else b"declared-but-missing"
        if content is not None:
            output = directory / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "role": role,
                "mime_type": mime_type,
                "size_bytes": len(declared),
                "sha256": hashlib.sha256(declared).hexdigest(),
                "metadata": {"fixture": True, "root": root_kind},
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": object_id,
                "status": "success",
                "timestamps": {
                    "created_at": "2026-07-27T00:00:00Z",
                    "finished_at": "2026-07-27T00:00:01Z",
                },
                "invocation": {
                    "config": {
                        "object_type": f"{root_kind}_fixture",
                    }
                },
                "artifacts": entries,
            }
        ),
        encoding="utf-8",
    )
    return directory


class SituationBuilderTests(unittest.TestCase):
    def test_operator_spec_builds_existing_strict_scenario_contract(self) -> None:
        spec = SituationSpec.from_mapping(valid_situation())
        suite = build_scenario_suite(spec)
        recipe = suite.recipes[0]
        self.assertEqual(suite.suite_id, "suite-operator-town10-ui-poc-v1")
        self.assertEqual(recipe.map_name, "Town10HD_Opt")
        self.assertEqual(recipe.traffic.vehicle_count, 35)
        self.assertEqual(recipe.traffic.walker_count, 30)
        self.assertEqual(recipe.capture.duration_ticks, 600)
        self.assertEqual(recipe.capture.capture_every_ticks, 4)
        self.assertEqual(len(recipe.props), 3)
        self.assertEqual(recipe.weather.weather_id, "operator-soft-rain-sunset")

    def test_situation_validation_and_non_overwriting_save(self) -> None:
        with self.assertRaisesRegex(ValueError, "vehicle_count"):
            SituationSpec.from_mapping(valid_situation(vehicle_count=201))
        with self.assertRaisesRegex(ValueError, "capture_fps"):
            SituationSpec.from_mapping(valid_situation(capture_fps=3))
        with self.assertRaisesRegex(ValueError, "situation_id"):
            SituationSpec.from_mapping(valid_situation(situation_id="../escape"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            spec = SituationSpec.from_mapping(valid_situation())
            output = save_situation_suite(spec, workspace=root)
            self.assertEqual(
                output.relative_to(root).as_posix(),
                "operator_configs/situations/town10-ui-poc.json",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["recipes"][0]["traffic"]["vehicle_count"], 35)
            with self.assertRaises(FileExistsError):
                save_situation_suite(spec, workspace=root)


class OperatorCatalogTests(unittest.TestCase):
    def test_native_host_kits_are_discoverable_research_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kit = root / "native_kits" / "native-host-kit-test-v001"
            kit.mkdir(parents=True)
            (kit / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "native-host-kit-test-v001",
                        "status": "success",
                        "artifacts": [
                            {
                                "role": "native_host_kit_release_manifest",
                                "path": "kit.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = build_catalog(
                root,
                carla_host="127.0.0.1",
                carla_port=1,
            )

            self.assertEqual(
                catalog["native_host_kits"],
                ["native_kits/native-host-kit-test-v001"],
            )
            row = catalog["research_objects"][0]
            self.assertEqual(row["root_kind"], "native_kits")
            self.assertEqual(row["artifact_count"], 1)
            self.assertEqual(row["manifest_source"], "declared")
            self.assertEqual(row["verification_status"], "not_checked")
            self.assertEqual(catalog["research_object_counts"]["total"], 1)
            self.assertEqual(
                catalog["research_object_counts"]["by_root"]["native_kits"],
                1,
            )


class OperatorCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "runs").mkdir()
        (self.root / "datasets").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "weights.pt").write_bytes(b"weights")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def live_parameters(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": "ui-live-test",
            "host": "172.20.10.7",
            "port": 2000,
            "vehicle_id": 24,
            "camera_id": 25,
            "resolution": "640x384",
            "camera_fps": 10.0,
            "camera_fov": 90.0,
            "expected_map": "Town10HD_Opt",
            "detector": "rtdetr",
            "weights": "weights.pt",
            "model_package": "",
            "device": "cpu",
            "image_size": 640,
            "confidence": 0.2,
            "control": "none",
            "cruise_speed": 2.0,
            "duration": 10.0,
            "max_stale_seconds": 0.75,
            "view": "none",
            "record_video": True,
            "shadow_policy": "hazard-stop",
            "policy_options": {"confidence": 0.35, "close_bottom": 0.72},
            "acknowledge_teacher_motion": False,
        }
        payload.update(overrides)
        return payload

    def test_live_command_is_tokenized_and_teacher_motion_is_gated(self) -> None:
        plan = build_command_plan(
            request("live", self.live_parameters()),
            workspace=self.root,
            executable="/python",
        )
        self.assertEqual(plan.command[:2], ("/python", "-c"))
        self.assertIn("carla_vision.runtime", plan.command[2])
        self.assertIn("--shadow-policy", plan.command)
        self.assertFalse(plan.motion_authorized)
        self.assertEqual(
            plan.expected_output,
            self.root.resolve() / "runs" / "ui-live-test",
        )

        with self.assertRaisesRegex(PermissionError, "teacher motion"):
            build_command_plan(
                request(
                    "live",
                    self.live_parameters(control="teacher"),
                ),
                workspace=self.root,
            )
        teacher = build_command_plan(
            request(
                "live",
                self.live_parameters(
                    run_id="ui-live-teacher",
                    control="teacher",
                    acknowledge_teacher_motion=True,
                ),
            ),
            workspace=self.root,
        )
        self.assertTrue(teacher.motion_authorized)

    def test_paths_cannot_escape_workspace_and_outputs_are_never_overwritten(self) -> None:
        outside = self.root.parent / "outside.pt"
        outside.write_bytes(b"outside")
        try:
            with self.assertRaisesRegex(ValueError, "workspace"):
                build_command_plan(
                    request(
                        "live",
                        self.live_parameters(weights=str(outside)),
                    ),
                    workspace=self.root,
                )
            (self.root / "runs" / "ui-live-test").mkdir()
            with self.assertRaises(FileExistsError):
                build_command_plan(
                    request("live", self.live_parameters()),
                    workspace=self.root,
                )
        finally:
            outside.unlink(missing_ok=True)

    def test_native_dry_run_and_real_execution_have_distinct_gates(self) -> None:
        plan_dir = self.root / "runs" / "plan"
        plan_dir.mkdir()
        parameters: dict[str, object] = {
            "scenario_plan": "runs/plan",
            "dataset_id": "ds-ui-native",
            "host": "172.20.10.7",
            "port": 2000,
            "partition": "train",
            "max_episodes": 1,
            "timeout": 30.0,
            "sensor_timeout": 10.0,
            "carla_python_api": "",
            "dry_run": True,
            "acknowledge_exclusive_tick_owner": False,
        }
        dry = build_command_plan(
            request("native_capture", parameters),
            workspace=self.root,
        )
        self.assertIn("--dry-run", dry.command)
        self.assertFalse(dry.destructive)
        self.assertIsNone(dry.expected_output)

        parameters["dry_run"] = False
        with self.assertRaisesRegex(PermissionError, "exclusive-tick-owner"):
            build_command_plan(
                request("native_capture", parameters),
                workspace=self.root,
            )
        parameters["acknowledge_exclusive_tick_owner"] = True
        real = build_command_plan(
            request("native_capture", parameters),
            workspace=self.root,
        )
        self.assertTrue(real.destructive)
        self.assertIn("--acknowledge-exclusive-tick-owner", real.command)

        preflight = build_command_plan(
            request(
                "native_preflight",
                {
                    "scenario_plan": "runs/plan",
                    "dataset_id": "ds-ui-native",
                    "run_id": "native-preflight-ui-test",
                    "host": "172.20.10.7",
                    "port": 2000,
                    "partition": "train",
                    "max_episodes": 1,
                    "timeout": 3.0,
                    "carla_python_api": "",
                    "confirm_world_reload": False,
                    "confirm_exclusive_tick_owner": False,
                },
            ),
            workspace=self.root,
        )
        self.assertFalse(preflight.destructive)
        self.assertFalse(preflight.motion_authorized)
        self.assertNotIn("--confirm-world-reload", preflight.command)
        self.assertEqual(
            preflight.expected_output,
            (self.root / "runs" / "native-preflight-ui-test").resolve(),
        )

    def test_dataset_qa_is_read_only_bounded_and_non_overwriting(self) -> None:
        dataset = self.root / "datasets" / "qa-source"
        dataset.mkdir()
        parameters: dict[str, object] = {
            "dataset": "datasets/qa-source",
            "run_id": "qa-source-automated-qa",
            "montage_count": 50,
        }
        plan = build_command_plan(
            request("dataset_qa", parameters),
            workspace=self.root,
            executable="/python",
        )
        self.assertIn("carla_vision.dataset.qa", plan.command[2])
        self.assertIn(str(dataset.resolve()), plan.command)
        self.assertIn("--montage-count", plan.command)
        self.assertEqual(plan.command[plan.command.index("--montage-count") + 1], "50")
        self.assertEqual(
            plan.expected_output,
            (self.root / "runs" / "qa-source-automated-qa").resolve(),
        )
        self.assertFalse(plan.motion_authorized)
        self.assertFalse(plan.destructive)
        self.assertIn("not human review or release signoff", plan.note)

        with self.assertRaisesRegex(ValueError, "montage_count"):
            build_command_plan(
                request("dataset_qa", {**parameters, "montage_count": 0}),
                workspace=self.root,
            )

        assert plan.expected_output is not None
        plan.expected_output.mkdir()
        with self.assertRaises(FileExistsError):
            build_command_plan(
                request("dataset_qa", parameters),
                workspace=self.root,
            )

    def test_verification_clean_git_gate_is_explicit(self) -> None:
        target = self.root / "runs" / "verify-target"
        target.mkdir()
        parameters: dict[str, object] = {
            "paths": ["runs/verify-target"],
            "allow_non_success": False,
            "require_clean_git": False,
        }
        development = build_command_plan(
            request("verify", parameters),
            workspace=self.root,
        )
        self.assertIn("--reject-unregistered", development.command)
        self.assertNotIn("--require-clean-git", development.command)
        self.assertNotIn("--no-deep", development.command)
        self.assertNotIn("--no-references", development.command)

        confirmatory = build_command_plan(
            request(
                "verify",
                {
                    **parameters,
                    "allow_non_success": True,
                    "require_clean_git": True,
                },
            ),
            workspace=self.root,
        )
        self.assertIn("--require-clean-git", confirmatory.command)
        self.assertIn("--allow-non-success", confirmatory.command)
        self.assertFalse(confirmatory.motion_authorized)
        self.assertFalse(confirmatory.destructive)

    def test_replay_and_shadow_templates_receive_new_operator_ids(self) -> None:
        dataset = self.root / "datasets" / "dataset"
        dataset.mkdir()
        evaluation = self.root / "configs" / "evaluation.json"
        evaluation.write_text("{}", encoding="utf-8")
        replay = self.root / "configs" / "replay.json"
        replay.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "replay_id": "template-replay",
                    "title": "Template replay",
                    "purpose": "development",
                    "master_seed": 7,
                    "reference_model_id": "model-a",
                    "bootstrap_replicates": 10,
                    "bootstrap_confidence": 0.95,
                    "latency_exclude_first_images": 1,
                    "montage_count": 1,
                    "models": [
                        {
                            "model_id": "model-a",
                            "label": "A",
                            "device": "cpu",
                            "model_package": None,
                            "detector": {
                                "backend": "rtdetr",
                                "weights": "../weights.pt",
                                "image_size": 640,
                                "factory": None,
                                "options": {},
                            },
                        },
                        {
                            "model_id": "model-b",
                            "label": "B",
                            "device": "cpu",
                            "model_package": None,
                            "detector": {
                                "backend": "yolo",
                                "weights": "../weights.pt",
                                "image_size": 640,
                                "factory": None,
                                "options": {},
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        replay_plan = build_command_plan(
            request(
                "replay",
                {
                    "config": "configs/replay.json",
                    "replay_id": "replay-ui-new",
                    "dataset": "datasets/dataset",
                    "evaluation_config": "configs/evaluation.json",
                    "acknowledge_locked_test": False,
                },
            ),
            workspace=self.root,
        )
        derived_replay = self.root / "operator_configs" / "replay" / "replay-ui-new.json"
        self.assertTrue(derived_replay.is_file())
        self.assertIn(str(derived_replay.resolve()), replay_plan.command)
        self.assertEqual(json.loads(replay.read_text())["replay_id"], "template-replay")

        shadow = self.root / "configs" / "shadow.json"
        detector = {
            "backend": "rtdetr",
            "weights": "../weights.pt",
            "model_package": None,
            "factory": None,
            "device": "cpu",
            "image_size": 640,
            "confidence": 0.2,
        }
        policy = {
            "backend": "hazard-stop",
            "factory": None,
            "checkpoint": None,
            "device": "cpu",
            "options": {},
        }
        shadow.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "matrix_id": "template-shadow",
                    "title": "Template shadow",
                    "purpose": "development",
                    "host": "172.20.10.7",
                    "port": 2000,
                    "vehicle_id": 24,
                    "camera_id": 25,
                    "expected_map": "Town10HD_Opt",
                    "resolution": [640, 384],
                    "camera_fps": 10.0,
                    "camera_fov": 90.0,
                    "duration_seconds": 5.0,
                    "max_stale_seconds": 0.75,
                    "control": "teacher",
                    "view": "none",
                    "record_video": False,
                    "continue_on_failure": True,
                    "cells": [
                        {
                            "cell_id": "cell-a",
                            "label": "Cell A",
                            "detector": detector,
                            "policy": policy,
                            "repetitions": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PermissionError, "teacher matrix"):
            build_command_plan(
                request(
                    "shadow_matrix",
                    {
                        "config": "configs/shadow.json",
                        "matrix_id": "shadow-ui-new",
                        "execute": True,
                        "acknowledge_teacher_motion": False,
                    },
                ),
                workspace=self.root,
            )
        self.assertFalse(
            (self.root / "operator_configs" / "shadow" / "shadow-ui-new.json").exists()
        )
        shadow_plan = build_command_plan(
            request(
                "shadow_matrix",
                {
                    "config": "configs/shadow.json",
                    "matrix_id": "shadow-ui-new",
                    "execute": True,
                    "acknowledge_teacher_motion": True,
                },
            ),
            workspace=self.root,
        )
        derived_shadow = self.root / "operator_configs" / "shadow" / "shadow-ui-new.json"
        self.assertTrue(derived_shadow.is_file())
        self.assertIn(str(derived_shadow.resolve()), shadow_plan.command)
        self.assertTrue(shadow_plan.motion_authorized)


class OperatorJobManagerTests(unittest.TestCase):
    def test_job_session_retains_request_command_status_logs_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = JobManager(workspace=root)
            job_request = request(
                "verify",
                {"paths": ["placeholder"], "allow_non_success": False},
            )
            plan = CommandPlan(
                kind="verify",
                title="Synthetic operator smoke",
                command=(
                    sys.executable,
                    "-c",
                    "print('operator-job-ok', flush=True)",
                ),
                expected_output=None,
                motion_authorized=False,
                destructive=False,
                note="unit test",
            )
            snapshot = manager.submit(job_request, plan)
            self.assertRegex(
                snapshot["job_id"],
                r"^op-\d{8}t\d{6}z-verify-[a-f0-9]{8}$",
            )
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                snapshot = manager.get(snapshot["job_id"])
                if snapshot["status"] in {"success", "failed", "stopped"}:
                    break
                time.sleep(0.02)
            self.assertEqual(snapshot["status"], "success")
            self.assertIn("operator-job-ok", manager.log_tail(snapshot["job_id"]))
            verified = verify_research_object(
                snapshot["session_path"],
                reject_unregistered=True,
            )
            self.assertEqual(verified.status, "success")
            self.assertEqual(verified.artifact_count, 5)
            self.assertEqual(verified.unregistered_file_count, 0)
            manager.shutdown()


class EvidenceExplorerServerTests(unittest.TestCase):
    ROOTS = (
        "datasets",
        "runs",
        "reports",
        "bundles",
        "models",
        "native_kits",
        "operator_sessions",
    )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for root_kind in self.ROOTS:
            (self.root / root_kind).mkdir()

        self.objects: dict[str, Path] = {}
        for root_kind in self.ROOTS:
            object_id = f"{root_kind}-evidence-v001"
            artifacts: dict[str, tuple[str, bytes | None, str]] = {
                "evidence.json": (
                    f"{root_kind}_evidence",
                    json.dumps({"root": root_kind}).encode(),
                    "application/json",
                )
            }
            if root_kind == "runs":
                artifacts = {
                    "images/actual-frame.png": (
                        "latest_overlay",
                        b"registered-png",
                        "image/png",
                    ),
                    "video/actual-session.mp4": (
                        "annotated_video",
                        b"registered-mp4",
                        "video/mp4",
                    ),
                    "linked.txt": (
                        "unsafe_link_fixture",
                        None,
                        "text/plain",
                    ),
                }
            elif root_kind == "reports":
                artifacts["tables/missing.csv"] = (
                    "missing_fixture",
                    None,
                    "text/csv",
                )
            elif root_kind == "models":
                artifacts = {
                    "weights/model.pt": (
                        "model_weights",
                        b"registered-model",
                        "application/octet-stream",
                    ),
                    "model.yaml": (
                        "model_configuration",
                        b"backend: rtdetr\n",
                        "application/yaml",
                    ),
                    "checksums.sha256": (
                        "model_checksum_index",
                        b"abc  weights/model.pt\n",
                        "text/plain",
                    ),
                }
            elif root_kind == "native_kits":
                artifacts = {
                    "payload/native-host-kit.zip": (
                        "native_host_kit_payload_archive",
                        b"PK-registered-kit",
                        "application/zip",
                    )
                }
            self.objects[root_kind] = write_research_object(
                self.root,
                root_kind,
                object_id,
                artifacts,
            )

        unregistered = self.objects["runs"] / "unregistered.txt"
        unregistered.write_text("must not be exposed", encoding="utf-8")
        outside = self.root / "symlink-target.txt"
        outside.write_text("must not be exposed through a symlink", encoding="utf-8")
        (self.objects["runs"] / "linked.txt").symlink_to(outside)
        (self.root / "runs" / "linked-object").symlink_to(
            self.objects["runs"],
            target_is_directory=True,
        )

        self.server = create_server(
            workspace=self.root,
            port=0,
            carla_host="127.0.0.1",
            carla_port=1,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server.application.jobs.shutdown()
        self.thread.join(timeout=5)
        self.temporary.cleanup()

    def api_url(self, endpoint: str, **query: str) -> str:
        return f"{self.base}{endpoint}?{urllib.parse.urlencode(query)}"

    def get_json(self, endpoint: str, **query: str) -> dict[str, object]:
        with urllib.request.urlopen(self.api_url(endpoint, **query), timeout=5) as response:
            return json.loads(response.read())

    def assert_http_error(
        self,
        expected: set[int],
        endpoint: str,
        **query: str,
    ) -> None:
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(self.api_url(endpoint, **query), timeout=5)
        self.assertIn(context.exception.code, expected)

    def test_catalog_and_evidence_api_cover_all_roots_without_verification_claim(self) -> None:
        with urllib.request.urlopen(f"{self.base}/api/bootstrap", timeout=5) as response:
            bootstrap = json.loads(response.read())
        catalog = bootstrap["catalog"]
        self.assertEqual(catalog["research_object_counts"]["total"], len(self.ROOTS))
        self.assertEqual(
            catalog["research_object_counts"]["by_root"],
            {root_kind: 1 for root_kind in self.ROOTS},
        )
        self.assertTrue(
            all(row["verification_status"] == "not_checked" for row in catalog["research_objects"])
        )

        run_path = "runs/runs-evidence-v001"
        payload = self.get_json("/api/evidence", path=run_path)
        self.assertEqual(payload["source"], "manifest_declarations")
        self.assertEqual(
            payload["verification"],
            {
                "performed": False,
                "status": "not_checked",
                "note": (
                    "This view lists manifest declarations and file availability; "
                    "use Verify for hash and semantic validation."
                ),
            },
        )
        artifact_paths = {row["path"] for row in payload["artifacts"]}
        self.assertEqual(
            artifact_paths,
            {
                "images/actual-frame.png",
                "video/actual-session.mp4",
                "linked.txt",
            },
        )
        self.assertNotIn("manifest.json", artifact_paths)
        self.assertNotIn("unregistered.txt", artifact_paths)
        by_path = {row["path"]: row for row in payload["artifacts"]}
        self.assertEqual(by_path["images/actual-frame.png"]["preview_kind"], "image")
        self.assertEqual(by_path["video/actual-session.mp4"]["preview_kind"], "video")
        self.assertFalse(by_path["linked.txt"]["available"])
        self.assertEqual(by_path["linked.txt"]["availability"], "unsafe_symlink")

        report = self.get_json("/api/evidence", path="reports/reports-evidence-v001")
        missing = next(row for row in report["artifacts"] if row["path"] == "tables/missing.csv")
        self.assertFalse(missing["available"])
        self.assertEqual(missing["availability"], "missing")

    def test_registered_zip_model_yaml_and_checksum_files_are_downloadable(self) -> None:
        downloads = (
            (
                "native_kits/native_kits-evidence-v001",
                "payload/native-host-kit.zip",
                b"PK-registered-kit",
            ),
            ("models/models-evidence-v001", "weights/model.pt", b"registered-model"),
            ("models/models-evidence-v001", "model.yaml", b"backend: rtdetr\n"),
            (
                "models/models-evidence-v001",
                "checksums.sha256",
                b"abc  weights/model.pt\n",
            ),
        )
        for object_path, artifact_path, expected in downloads:
            with self.subTest(artifact_path=artifact_path):
                with urllib.request.urlopen(
                    self.api_url(
                        "/api/artifact",
                        object=object_path,
                        path=artifact_path,
                    ),
                    timeout=5,
                ) as response:
                    self.assertEqual(response.read(), expected)
                    self.assertTrue(
                        response.headers["Content-Disposition"].startswith("attachment;")
                    )
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                    self.assertIn(
                        "script-src 'none'",
                        response.headers["Content-Security-Policy"],
                    )

    def test_traversal_symlinks_missing_and_unregistered_files_are_rejected(self) -> None:
        self.assert_http_error({400}, "/api/evidence", path="../outside")
        self.assert_http_error({400}, "/api/evidence", path="/runs/runs-evidence-v001")
        self.assert_http_error(
            {400},
            "/api/evidence",
            path="runs/runs-evidence-v001/nested",
        )
        self.assert_http_error({400}, "/api/evidence", path="runs/linked-object")
        self.assert_http_error(
            {404},
            "/api/artifact",
            object="runs/runs-evidence-v001",
            path="unregistered.txt",
        )
        self.assert_http_error(
            {404},
            "/api/artifact",
            object="reports/reports-evidence-v001",
            path="tables/missing.csv",
        )
        self.assert_http_error(
            {400},
            "/api/artifact",
            object="runs/runs-evidence-v001",
            path="linked.txt",
        )
        self.assert_http_error(
            {400},
            "/api/artifact",
            object="runs/runs-evidence-v001",
            path="../symlink-target.txt",
        )
        self.assert_http_error(
            {400},
            "/api/artifact",
            object="runs/runs-evidence-v001",
            path="C:/Windows/secret.txt",
        )
        self.assert_http_error(
            {400},
            "/api/artifact",
            path="runs/runs-evidence-v001/unregistered.txt",
        )

        script = (
            Path(__file__).parents[1] / "carla_vision" / "operator" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertNotIn("images/latest_overlay.png", script)
        self.assertNotIn("video/overlay.mp4", script)


class OperatorServerTests(unittest.TestCase):
    def test_local_http_bootstrap_situation_and_csrf_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("runs", "datasets", "reports", "bundles", "models"):
                (root / directory).mkdir()
            server = create_server(
                workspace=root,
                port=0,
                carla_host="127.0.0.1",
                carla_port=1,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address[:2]
            base = f"http://{host}:{port}"
            try:
                with urllib.request.urlopen(f"{base}/", timeout=5) as response:
                    html = response.read().decode("utf-8")
                self.assertIn("CARLA Vision Operator", html)
                with urllib.request.urlopen(f"{base}/api/bootstrap", timeout=5) as response:
                    bootstrap = json.loads(response.read())
                self.assertFalse(bootstrap["catalog"]["capabilities"]["carla_tcp_reachable"])
                token = bootstrap["token"]

                body = json.dumps(valid_situation()).encode("utf-8")
                unauthorized = urllib.request.Request(
                    f"{base}/api/situations",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(unauthorized, timeout=5)
                self.assertEqual(context.exception.code, 403)

                authorized = urllib.request.Request(
                    f"{base}/api/situations",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-Operator-Token": token,
                    },
                )
                with urllib.request.urlopen(authorized, timeout=5) as response:
                    saved = json.loads(response.read())
                self.assertEqual(saved["status"], "saved")
                self.assertTrue((root / saved["path"]).is_file())
            finally:
                server.shutdown()
                server.server_close()
                server.application.jobs.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
