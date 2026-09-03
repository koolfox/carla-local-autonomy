from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from test_drive_console import CARLA_HOST, CARLA_PORT, valid_start
from test_operator_configuration import _session

from carla_vision.operator.catalog import build_catalog
from carla_vision.operator.configuration import (
    build_garage_preview_request,
    build_legacy_drive_request,
    session_defaults,
)
from carla_vision.operator.drive_contracts import DriveStartConfig

ROOT = Path(__file__).parents[1]


def test_researcher_checkpoint_is_selectable_without_moving_it_to_root(tmp_path, monkeypatch):
    monkeypatch.setattr("carla_vision.operator.catalog.probe_endpoint", lambda *_args: False)
    folder = tmp_path / "models" / "my-rtdetr"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"catalog-only-do-not-load")
    (tmp_path / "legacy.pt").write_bytes(b"catalog-only-do-not-load")
    catalog = build_catalog(tmp_path, carla_host="127.0.0.1", carla_port=2000)
    assert catalog["weights"] == ["legacy.pt", "models/my-rtdetr/best.pt"]


def test_existing_session_without_voxel_field_defaults_off_without_mutation() -> None:
    session = _session()
    del session["perception"]["voxelEnabled"]
    original = deepcopy(session)

    request = build_legacy_drive_request(
        session,
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        worker_connected=True,
        capabilities={"autopilot": True},
    )

    assert request["voxel_enabled"] is False
    assert session == original
    assert session_defaults(detector_enabled=True)["perception"]["voxelEnabled"] is False
    assert session_defaults(detector_enabled=False)["perception"]["voxelEnabled"] is False


@pytest.mark.parametrize("control_mode", ["manual", "autopilot"])
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_voxel_is_independent_of_detector_and_manual_or_tm_control(
    tmp_path: Path,
    control_mode: str,
    device: str,
) -> None:
    request = build_legacy_drive_request(
        _session(
            control__mode=control_mode,
            perception__enabled=False,
            perception__voxelEnabled=True,
            perception__device=device,
            perception__weights="",
        ),
        carla_host=CARLA_HOST,
        carla_port=CARLA_PORT,
        worker_connected=True,
        capabilities={"autopilot": True},
    )

    assert request["control_mode"] == "manual"
    assert request["initial_control_mode"] == control_mode
    assert request["voxel_enabled"] is True
    assert request["detector_enabled"] is False
    assert request["device"] == device
    assert request["policy_checkpoint"] == ""
    assert "model_package_id" not in request
    config = DriveStartConfig.from_mapping(
        valid_start(
            detector_enabled=False,
            weights="",
            voxel_enabled=True,
            device=device,
            initial_control_mode=control_mode,
        ),
        workspace=tmp_path,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
        world_worker_configured=True,
    )
    assert config.voxel_enabled
    assert not config.detector_enabled
    assert config.device == device
    assert config.initial_control_mode == control_mode
    assert config.manifest_config()["voxel_enabled"] is True
    assert config.manifest_config()["model_output_actuated"] is False


def test_legacy_drive_request_does_not_require_voxel_field(tmp_path: Path) -> None:
    config = DriveStartConfig.from_mapping(
        valid_start(detector_enabled=False, weights=""),
        workspace=tmp_path,
        expected_host=CARLA_HOST,
        expected_port=CARLA_PORT,
    )

    assert config.voxel_enabled is False
    assert config.manifest_config()["voxel_enabled"] is False


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_voxel_opt_in_requires_a_boolean(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(TypeError, match="voxelEnabled must be a boolean"):
        build_garage_preview_request(_session(perception__voxelEnabled=invalid))

    with pytest.raises(TypeError, match="voxel_enabled must be a boolean"):
        DriveStartConfig.from_mapping(
            valid_start(detector_enabled=False, weights="", voxel_enabled=invalid),
            workspace=tmp_path,
            expected_host=CARLA_HOST,
            expected_port=CARLA_PORT,
        )


def test_voxel_display_does_not_change_garage_preview_request() -> None:
    disabled = build_garage_preview_request(_session(perception__voxelEnabled=False))
    enabled = build_garage_preview_request(_session(perception__voxelEnabled=True))

    assert enabled == disabled
    assert "voxel_enabled" not in enabled
    assert "voxelEnabled" not in enabled


def test_svelte_voxel_view_is_opt_in_and_preserves_raw_default() -> None:
    cockpit = (ROOT / "web/src/lib/components/DriveCockpit.svelte").read_text(encoding="utf-8")
    vision = (ROOT / "web/src/lib/components/VisionSettings.svelte").read_text(encoding="utf-8")
    preview = (ROOT / "web/src/lib/components/GaragePreview.svelte").read_text(encoding="utf-8")
    config = (ROOT / "web/src/lib/domain/config.ts").read_text(encoding="utf-8")
    runtime = (ROOT / "web/src/lib/domain/runtime.ts").read_text(encoding="utf-8")

    assert "let view: 'raw' | 'overlay' | 'voxel' = 'raw'" in cockpit
    assert "view = 'voxel'" not in cockpit
    assert "disabled={!voxelEnabled}" in cockpit
    assert "chooseView('voxel')" in cockpit
    assert "voxel?.status === 'failed'" in cockpit
    assert "voxel.error" in cockpit
    assert "voxel?.latency_ms" in cockpit
    assert "voxel?.source_frame" in cockpit
    assert "Voxel (RGB depth)" in vision
    assert "100 MB" in vision
    assert "does not infer lanes or steer" in vision
    assert "garagePreviewSignature($sessionConfig)" in preview
    assert "voxelEnabled: false" in config
    assert "voxel?: DriveVoxelState" in runtime
    assert "actuated: false" in runtime
