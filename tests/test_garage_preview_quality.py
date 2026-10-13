from __future__ import annotations

import re
from pathlib import Path

from carla_vision.operator.garage_preview import GaragePreviewConfig

ROOT = Path(__file__).parents[1]
SVELTE = ROOT / "web/src/lib/components/GaragePreview.svelte"
STYLES = ROOT / "web/src/lib/styles/app.css"


def preview_request(**patch: object) -> dict[str, object]:
    value: dict[str, object] = {
        "map_name": "Town10HD_Opt",
        "weather_preset": "clear-day",
        "vehicle_blueprint": "vehicle.tesla.model3",
        "color": "",
        "seed": 42,
        "traffic_count": 0,
        "walker_count": 0,
        "prop_preset": "none",
    }
    value.update(patch)
    return value


def test_preview_defaults_to_production_balanced_camera_profile() -> None:
    config = GaragePreviewConfig.from_mapping(preview_request())

    assert config.profile == "balanced"
    assert (config.width, config.height, config.fps) == (1280, 720, 30.0)
    assert config.fov == 65.0


def test_svelte_preview_uses_double_buffered_stream_instead_of_clearing_ready_frame() -> None:
    source = SVELTE.read_text(encoding="utf-8")

    assert "createGarageStreamBuffer" in source
    assert "stageGarageStream" in source
    assert "confirmGarageStream" in source
    assert "failGarageStream" in source
    assert "streamBuffer.sources" in source
    assert "streamBuffer.ready" in source
    assert "streamReady = false" not in source
    assert "last decoded frame" in source.lower()


def _block(styles: str, header: str) -> str:
    match = re.search(re.escape(header) + r"\s*\{(?P<body>.*?)\n\}", styles, re.DOTALL)
    assert match is not None, header
    return match.group("body")


def test_toolbar_has_stable_desktop_tablet_and_phone_layout_regions() -> None:
    styles = STYLES.read_text(encoding="utf-8")

    # Desktop and laptop: one floating row centred under the artboard.
    desktop = _block(styles, "\n.garage-preview-toolbar")
    assert "display: flex" in desktop
    assert "justify-self: center" in desktop
    assert "flex-wrap: wrap" in desktop

    # Narrow laptop canvas: the same row compacts instead of dropping controls.
    laptop = _block(styles, "@container canvas (min-width: 861px) and (max-width: 1000px)")
    assert ".vehicle-carousel" in laptop
    assert ".garage-camera-presets" in laptop

    # Phone and tablet: full-width toolbar, inspector becomes a bottom sheet.
    phone = re.search(
        r"@media \(max-width: 860px\)\s*\{(?P<body>.*?)(?:\n\}\n\n@media|\Z)",
        styles,
        re.DOTALL,
    )
    assert phone is not None
    body = phone.group("body")
    assert ".garage-preview-toolbar" in body
    assert ".garage-preview-actions" in body
    assert ".panels" in body and "position: fixed" in body


def test_orbit_surface_is_a_real_button_for_pointer_and_keyboard_semantics() -> None:
    source = SVELTE.read_text(encoding="utf-8")
    assert re.search(r'<button\s+type="button"\s+class="garage-orbit-surface"', source)
    assert 'aria-label="Garage orbit camera"' in source
