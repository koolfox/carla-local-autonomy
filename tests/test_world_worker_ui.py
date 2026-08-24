from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
STATIC_ROOT = ROOT / "carla_vision" / "operator" / "static"


class _ElementParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.elements[element_id] = (tag, attributes)


class WorldWorkerGarageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        cls.script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        cls.styles = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
        cls.parser = _ElementParser()
        cls.parser.feed(cls.html)

    def select_values(self, element_id: str) -> list[str]:
        match = re.search(
            rf'<select\s+id="{re.escape(element_id)}"[^>]*>(.*?)</select>',
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(match, element_id)
        return re.findall(r'<option\s+value="([^"]*)"', match.group(1))

    def test_garage_has_bounded_world_worker_choices(self) -> None:
        for element_id in (
            "drive-map-choice",
            "drive-traffic-choice",
            "drive-walkers-choice",
            "drive-starting-choice",
        ):
            with self.subTest(element_id=element_id):
                tag, attributes = self.parser.elements[element_id]
                self.assertEqual(tag, "select")
                self.assertIn("disabled", attributes)

        self.assertEqual(self.select_values("drive-traffic-choice"), ["0", "5", "10", "20", "40"])
        self.assertEqual(self.select_values("drive-walkers-choice"), ["0", "5", "10", "20", "40"])
        self.assertEqual(
            self.select_values("drive-starting-choice"),
            ["free", "random_destination"],
        )

        mode_tag, mode_attributes = self.parser.elements["drive-control-mode"]
        self.assertEqual(mode_tag, "div")
        self.assertEqual(mode_attributes.get("role"), "radiogroup")
        for element_id, mode in (
            ("drive-mode-manual", "manual"),
            ("drive-mode-autopilot", "autopilot"),
        ):
            with self.subTest(element_id=element_id):
                tag, attributes = self.parser.elements[element_id]
                self.assertEqual(tag, "button")
                self.assertEqual(attributes.get("type"), "button")
                self.assertEqual(attributes.get("data-drive-mode"), mode)

    def test_catalog_capabilities_gate_each_world_control(self) -> None:
        for capability in (
            '"native_worker"',
            '"map_reload"',
            '"random_route"',
            '"traffic_manager"',
            '"walkers"',
            '"autopilot"',
            '"driving_guidance"',
        ):
            with self.subTest(capability=capability):
                self.assertIn(capability, self.script)

        for binding in (
            '$("drive-map-choice").disabled = active || !capabilities.mapReload;',
            '["drive-traffic-field", "drive-traffic-choice", capabilities.traffic]',
            '["drive-walkers-field", "drive-walkers-choice", capabilities.walkers]',
            '["drive-route-field", "drive-starting-choice", capabilities.randomRoute]',
            '$("drive-control-mode-field").hidden = !capabilities.autopilot;',
            "configureDriveWorldControls();",
        ):
            with self.subTest(binding=binding):
                self.assertIn(binding, self.script)

        self.assertIn(".garage-group [hidden]", self.styles)
        self.assertIn("display: none !important;", self.styles)

    def test_start_request_uses_typed_world_fields(self) -> None:
        expected = (
            "map_name: driveWorldCapabilities().mapReload",
            '? $("drive-map-choice").value || state.drive.catalog?.map || "current"',
            'traffic_count: number("drive-traffic-choice")',
            'walker_count: number("drive-walkers-choice")',
            'route_mode: $("drive-starting-choice").value || "free"',
            "initial_control_mode: state.drive.initialControlMode",
        )
        for field in expected:
            with self.subTest(field=field):
                self.assertIn(field, self.script)

    def test_single_shell_uses_one_canonical_worker_payload(self) -> None:
        self.assertIn("function driveStartPayload()", self.script)
        self.assertEqual(self.script.count('request("/api/drive/start"'), 1)
        self.assertNotIn("fullStartPayload", self.script)
        self.assertNotIn("startGarageDrive", self.script)
        self.assertNotIn("drive-garage-control-mode", self.html)

    def test_autonomy_blocks_base_manual_input_without_global_extension_shim(self) -> None:
        self.assertIn("function driveExtensionBlocksManualControl()", self.script)
        self.assertGreaterEqual(self.script.count("driveExtensionBlocksManualControl()"), 5)
        self.assertIn('state.drive.session?.garage_mode || "manual"', self.script)
        self.assertNotIn("carlaGarageManualControlBlocked", self.script)

    def test_autopilot_takeover_precedes_manual_commands(self) -> None:
        self.assertIn('request("/api/drive/mode", {', self.script)
        self.assertIn(
            'body: JSON.stringify({ session_id: driveSessionId(), mode: "manual" })', self.script
        )
        self.assertIn("if (!safety && driveIsAutopilot()) return;", self.script)
        self.assertIn(
            "if (driveIsAutopilot() && !(await requestDriveManualMode())) return;", self.script
        )
        self.assertIn("void requestDriveManualMode().then((taken) => {", self.script)
        self.assertIn('? "Take Control"', self.script)

        # Existing deadman and lifecycle hooks remain part of the worker-enabled cockpit.
        for safety_hook in (
            'viewport.addEventListener("blur", () => releaseDriveControl("Viewport focus lost"));',
            'window.addEventListener("blur", () => releaseDriveControl("Browser focus lost"));',
            'window.addEventListener("pagehide", () => releaseDriveControl("Page closing"));',
            "window.setInterval(() => void sendDriveControl(), 67);",
        ):
            with self.subTest(safety_hook=safety_hook):
                self.assertIn(safety_hook, self.script)

    def test_browser_never_collects_world_worker_endpoint_or_secret(self) -> None:
        lowered_html = self.html.lower()
        lowered_script = self.script.lower()
        for forbidden in (
            "drive-worker-url",
            "drive-worker-token",
            "world_worker_url",
            "world_worker_token",
            "world-worker-authorization",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered_html)
                self.assertNotIn(forbidden, lowered_script)

        self.assertIn('request("/api/drive/catalog")', self.script)

    def test_privileged_path_guide_is_exact_frame_stream_and_explicitly_sourced(self) -> None:
        button_tag, button_attributes = self.parser.elements["drive-guidance-toggle"]
        self.assertEqual(button_tag, "button")
        self.assertEqual(button_attributes.get("aria-pressed"), "true")
        self.assertIn('guidance.label || "CARLA PATH"', self.script)
        self.assertIn("· PRIVILEGED · ${sampleLabel} · ACTUAL STEER", self.script)
        self.assertIn("session.guidance_views?.[state.drive.view]", self.script)
        self.assertIn('pathEnabled ? "path" : "clean"', self.script)
        self.assertIn("guidance=${pathEnabled ? 1 : 0}", self.script)
        self.assertNotIn("drive-guidance-canvas", self.html)
        self.assertNotIn("debug.draw", self.script.lower())


if __name__ == "__main__":
    unittest.main()
