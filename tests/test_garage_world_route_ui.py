from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_scene_world_fields_expose_capability_driven_spawn_and_destination_controls() -> None:
    source = (ROOT / "web/src/lib/components/SceneWorldFields.svelte").read_text(encoding="utf-8")
    assert "Start point" in source
    assert "Selected destination" in source
    assert "spawn_point_selection" in source
    assert "selected_route" in source
    assert "no fallback if occupied" in source


def test_drive_cockpit_names_runtime_control_owner_and_hides_manual_pad_for_policies() -> None:
    source = (ROOT / "web/src/lib/components/DriveCockpit.svelte").read_text(encoding="utf-8")
    for label in (
        "CARLA Traffic Manager",
        "CARLA BehaviorAgent",
        "Imitation policy",
        "Voxel supervisor",
        "Registered model",
        "Browser manual",
    ):
        assert label in source
    assert "policyOwnsControl" in source
    assert "Control owner" in source
