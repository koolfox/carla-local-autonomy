from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from carla_vision.artifacts import RunArtifactTracker
from carla_vision.reproduction import (
    build_reproduction_bundle,
    load_verified_reproduction_bundle,
)
from carla_vision.verification import ArtifactIntegrityError, verify_research_object


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixed_hardware() -> dict[str, object]:
    return {
        "cpu": {"architecture": "test", "model": "test", "logical_count": 2},
        "memory": {"total_bytes": 1024},
        "accelerators": {
            "available": False,
            "version": None,
            "cuda": {
                "available": False,
                "runtime_version": None,
                "device_count": 0,
                "devices": [],
            },
            "cudnn": {"available": False, "version": None},
            "mps": {"available": False, "built": False},
            "rocm_version": None,
            "determinism": {
                "deterministic_algorithms": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": False,
            },
        },
        "tools": {"python_compiler": "test", "git": "test", "uv": "test"},
        "container": {"detected": False, "image_digest": None},
    }


def _rehash_artifact(bundle: Path, relative: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = bundle / relative
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    size = target.stat().st_size
    for artifact in manifest["artifacts"]:
        if artifact["path"] == relative:
            artifact["sha256"] = digest
            artifact["size_bytes"] = size
            break
    else:
        raise AssertionError(f"artifact is not registered: {relative}")

    checksum_path = bundle / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    rewritten = [
        f"{digest}  {relative}" if line.endswith(f"  {relative}") else line for line in lines
    ]
    checksum_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    checksum_digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "checksums.sha256":
            artifact["sha256"] = checksum_digest
            artifact["size_bytes"] = checksum_path.stat().st_size
            break
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ReproductionBundleTests(unittest.TestCase):
    def _build_fixture(
        self,
        root: Path,
        *,
        source_role: str = "other",
    ) -> tuple[Path, Path]:
        repository = root / "repository"
        (repository / "src").mkdir(parents=True)
        (repository / "configs" / "reproduction").mkdir(parents=True)
        (repository / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (repository / "pyproject.toml").write_text(
            "[project]\nname='fixture'\nversion='1.0.0'\n",
            encoding="utf-8",
        )
        (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (repository / "src" / "demo.py").write_text(
            "print('reproduced')\n",
            encoding="utf-8",
        )

        source_tracker = RunArtifactTracker(
            repository / "runs",
            run_id="source-a",
            config={"kind": "fixture"},
            repository_root=repository,
            package_names=(),
            hardware_probe=_fixed_hardware,
        )
        with source_tracker:
            metrics = source_tracker.artifact_path("metrics.json")
            _write_json(metrics, {"score": 1.0})
            source_tracker.register_artifact(metrics, role="metrics_table")

        config_path = repository / "configs" / "reproduction" / "bundle.json"
        _write_json(
            config_path,
            {
                "schema_version": "1.0",
                "bundle_id": "bundle-fixture-v1",
                "title": "Fixture reproduction bundle",
                "authors": ["Test Author"],
                "purpose": "development",
                "repository_root": "../..",
                "sources": [
                    {
                        "source_id": "source-a",
                        "role": source_role,
                        "path": "../../runs/source-a",
                        "label": "Fixture source",
                    }
                ],
                "source_paths": [
                    "README.md",
                    "pyproject.toml",
                    "uv.lock",
                    "src",
                    "configs/reproduction/bundle.json",
                ],
                "commands": [
                    ["uv", "sync", "--frozen"],
                    ["uv", "run", "python", "src/demo.py"],
                ],
                "limitations": ["The fixture contains no CARLA binary or trained model weights."],
            },
        )
        result = build_reproduction_bundle(
            config_path=config_path,
            bundles_root=repository / "bundles",
            cli_args={"config": str(config_path)},
        )
        return repository, Path(result["bundle_dir"])

    def test_evidence_registry_source_role_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _repository, bundle = self._build_fixture(
                Path(temporary_directory),
                source_role="evidence-registry",
            )

            verified = load_verified_reproduction_bundle(bundle)
            self.assertEqual(verified.config.sources[0].role, "evidence-registry")
            self.assertEqual(
                verified.source_graph["sources"][0]["role"],
                "evidence-registry",
            )

    def test_bundle_is_semantically_verified_and_independent_of_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository, bundle = self._build_fixture(Path(temporary_directory))
            verified = load_verified_reproduction_bundle(bundle)
            self.assertEqual(verified.bundle_id, "bundle-fixture-v1")
            self.assertEqual(verified.descriptor["source_count"], 1)
            self.assertEqual(verified.descriptor["dependency_lock_count"], 1)
            self.assertEqual(
                verified.source_inventory["file_count"],
                5,
            )

            generic = verify_research_object(
                bundle,
                reject_unregistered=True,
            )
            self.assertEqual(
                generic.deep_verification["kind"],
                "reproduction_bundle",
            )
            self.assertEqual(generic.external_reference_count, 0)
            self.assertEqual(generic.unregistered_file_count, 0)

            (repository / "runs" / "source-a").rename(repository / "removed-original-source")
            moved_bundle = Path(temporary_directory) / "standalone-bundle"
            bundle.rename(moved_bundle)
            standalone = load_verified_reproduction_bundle(moved_bundle)
            self.assertEqual(standalone.bundle_id, "bundle-fixture-v1")

    def test_semantic_verifier_rejects_rehashed_command_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            _repository, bundle = self._build_fixture(Path(temporary_directory))
            commands_path = bundle / "commands.json"
            commands = json.loads(commands_path.read_text(encoding="utf-8"))
            commands["commands"][1][-1] = "src/other.py"
            _write_json(commands_path, commands)
            _rehash_artifact(bundle, "commands.json")

            shallow = verify_research_object(
                bundle,
                deep=False,
                reject_unregistered=True,
            )
            self.assertEqual(shallow.deep_verification["kind"], "skipped")
            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "commands fingerprint",
            ):
                load_verified_reproduction_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
