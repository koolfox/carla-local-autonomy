from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
STATIC_ROOT = ROOT / "carla_vision" / "operator" / "static"


class _GarageDocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.elements: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.metas: dict[str, str] = {}
        self.tabs: list[dict[str, str | None]] = []
        self.tablist_count = 0
        self.nested_forms: list[tuple[str, str]] = []
        self._form_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
            self.elements[element_id] = (tag, attributes)

        if tag == "meta" and attributes.get("name") and attributes.get("content"):
            self.metas[attributes["name"]] = attributes["content"]

        if attributes.get("role") == "tablist":
            self.tablist_count += 1
        if tag == "button" and attributes.get("role") == "tab":
            self.tabs.append(attributes)

        if tag == "form":
            form_id = element_id or "<anonymous>"
            if self._form_stack:
                self.nested_forms.append((self._form_stack[-1], form_id))
            self._form_stack.append(form_id)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form_stack:
            self._form_stack.pop()


def _balanced_block(text: str, marker: str) -> str:
    marker_position = text.find(marker)
    if marker_position < 0:
        raise AssertionError(f"CSS marker not found: {marker}")
    open_brace = text.find("{", marker_position + len(marker))
    if open_brace < 0:
        raise AssertionError(f"CSS block has no opening brace: {marker}")

    return _balanced_block_from(text, open_brace, marker)


def _balanced_block_from(text: str, open_brace: int, label: str) -> str:
    depth = 0
    for position in range(open_brace, len(text)):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : position]
    raise AssertionError(f"CSS block is not closed: {label}")


def _selector_block(text: str, *selectors: str) -> str:
    selector_pattern = r"\s*,\s*".join(re.escape(selector) for selector in selectors)
    match = re.search(rf"(?m)^\s*{selector_pattern}\s*\{{", text)
    if match is None:
        raise AssertionError(f"CSS selector not found: {', '.join(selectors)}")
    return _balanced_block_from(text, match.end() - 1, ", ".join(selectors))


def _property(block: str, name: str) -> str:
    match = re.search(rf"(?:^|;)\s*{re.escape(name)}\s*:\s*([^;]+)", block)
    if match is None:
        raise AssertionError(f"CSS property not found: {name}")
    return match.group(1).strip()


def _lightness(hex_color: str) -> float:
    value = hex_color.removeprefix("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if len(value) != 6:
        raise AssertionError(f"Expected a hex color, got {hex_color!r}")
    channels = [int(value[offset : offset + 2], 16) for offset in (0, 2, 4)]
    return sum(channels) / len(channels)


class GarageVisualContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        cls.styles = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
        cls.parser = _GarageDocumentParser()
        cls.parser.feed(cls.html)

    def test_page_declares_light_chrome_and_edge_to_edge_safe_area_support(self) -> None:
        viewport_tokens = {
            token.strip() for token in self.parser.metas["viewport"].split(",")
        }
        self.assertIn("width=device-width", viewport_tokens)
        self.assertIn("initial-scale=1", viewport_tokens)
        self.assertIn("viewport-fit=cover", viewport_tokens)
        self.assertEqual(self.parser.metas["color-scheme"], "light")
        self.assertRegex(self.parser.metas["theme-color"], r"^#[0-9a-fA-F]{6}$")

        root_theme = _balanced_block(self.styles, ":root")
        self.assertGreater(_lightness(_property(root_theme, "--bg")), 220)
        self.assertGreater(_lightness(_property(root_theme, "--surface")), 220)

    def test_tab_controls_expose_aria_state_and_control_real_tabpanels(self) -> None:
        self.assertGreaterEqual(self.parser.tablist_count, 2)
        self.assertGreaterEqual(len(self.parser.tabs), 7)

        for tab in self.parser.tabs:
            controlled_id = tab.get("aria-controls")
            with self.subTest(controlled_id=controlled_id):
                self.assertEqual(tab.get("type"), "button")
                self.assertIn(tab.get("aria-selected"), {"true", "false"})
                self.assertIsNotNone(controlled_id)
                self.assertIn(controlled_id, self.parser.elements)
                target_tag, target_attributes = self.parser.elements[controlled_id]
                self.assertEqual(target_tag, "section")
                self.assertEqual(target_attributes.get("role"), "tabpanel")

    def test_safe_area_insets_protect_cockpit_mobile_shell_and_sticky_action(self) -> None:
        cockpit = _balanced_block(self.styles, ".drive-immersive .drive-viewer-card")
        for edge in ("top", "right", "bottom", "left"):
            with self.subTest(cockpit_edge=edge):
                self.assertIn(f"env(safe-area-inset-{edge})", cockpit)

        mobile = _balanced_block(self.styles, "@media (max-width: 480px)")
        mobile_shell = _balanced_block(mobile, ".app-shell")
        self.assertIn("env(safe-area-inset-right)", mobile_shell)
        self.assertIn("env(safe-area-inset-left)", mobile_shell)

        adaptive_action = _balanced_block(self.styles, "@media (max-width: 780px),")
        sticky_action = _balanced_block(
            adaptive_action,
            "body.drive-tab-active:not(.drive-immersive) .drive-session-actions",
        )
        for edge in ("right", "bottom", "left"):
            with self.subTest(sticky_edge=edge):
                self.assertIn(f"env(safe-area-inset-{edge})", sticky_action)

    def test_mobile_sticky_action_and_tablet_form_remain_structurally_responsive(self) -> None:
        adaptive_action = _balanced_block(self.styles, "@media (max-width: 780px),")
        sticky_action = _balanced_block(
            adaptive_action,
            "body.drive-tab-active:not(.drive-immersive) .drive-session-actions",
        )
        self.assertEqual(_property(sticky_action, "position"), "fixed")
        self.assertEqual(_property(sticky_action, "bottom"), "0")
        self.assertEqual(_property(sticky_action, "left"), "0")
        self.assertEqual(_property(sticky_action, "right"), "0")

        mobile_shell = _balanced_block(
            adaptive_action,
            "body.drive-tab-active:not(.drive-immersive) .app-shell",
        )
        self.assertIn("safe-area-inset-bottom", _property(mobile_shell, "padding-bottom"))

        tablet = _balanced_block(
            self.styles,
            "@media (min-width: 600px) and (max-width: 780px)",
        )
        tablet_fields = _balanced_block(tablet, ".drive-setup-form .fields.two")
        self.assertRegex(
            _property(tablet_fields, "grid-template-columns"),
            r"repeat\(2,\s*minmax\(0,\s*1fr\)\)",
        )

    def test_flexible_grid_children_and_controls_can_shrink_without_overflow(self) -> None:
        card = _selector_block(self.styles, ".card")
        fields = _selector_block(self.styles, ".fields")
        garage_group = _selector_block(self.styles, ".garage-group")
        self.assertEqual(_property(card, "min-width"), "0")
        self.assertEqual(_property(garage_group, "min-width"), "0")
        self.assertIn("minmax(0, 1fr)", _property(fields, "grid-template-columns"))

        self.assertRegex(
            self.styles,
            re.compile(
                r"^\s*input\s*,\s*select\s*\{[^}]*min-width:\s*0\s*;",
                re.MULTILINE | re.DOTALL,
            ),
        )

        responsive = _balanced_block(self.styles, "@media (max-width: 780px)")
        shrinking_children = _selector_block(
            responsive,
            ".evidence-filters > *",
            ".form-grid > *",
            ".fields > *",
        )
        self.assertEqual(_property(shrinking_children, "min-width"), "0")

    def test_cockpit_uses_legacy_viewport_fallback_before_dynamic_viewport(self) -> None:
        for selector in (
            ".drive-immersive .app-shell",
            ".drive-immersive .drive-viewer-card",
        ):
            with self.subTest(selector=selector):
                block = _balanced_block(self.styles, selector)
                self.assertRegex(block, r"height:\s*100vh\s*;\s*height:\s*100dvh\s*;")

    def test_reduced_motion_contract_disables_nonessential_animation(self) -> None:
        reduced_motion = _balanced_block(
            self.styles,
            "@media (prefers-reduced-motion: reduce)",
        )
        self.assertIn("*::before", reduced_motion)
        self.assertIn("*::after", reduced_motion)
        self.assertRegex(reduced_motion, r"animation-duration:\s*0\.01ms\s*!important")
        self.assertRegex(reduced_motion, r"animation-iteration-count:\s*1\s*!important")
        self.assertRegex(reduced_motion, r"transition-duration:\s*0\.01ms\s*!important")

    def test_document_structure_has_unique_ids_and_no_nested_forms(self) -> None:
        duplicate_ids = sorted(
            element_id
            for element_id in set(self.parser.ids)
            if self.parser.ids.count(element_id) > 1
        )
        self.assertEqual(duplicate_ids, [])
        self.assertEqual(self.parser.nested_forms, [])


if __name__ == "__main__":
    unittest.main()
