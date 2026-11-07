"""Check that public workshop placeholders remain usable and reference local assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / "docs" / "workshop"


def test_media_slots_match_slide_references_and_use_local_assets() -> None:
    slides = (WORKSHOP / "docs" / "class-presentation.fa.md").read_text(encoding="utf-8")
    references = re.findall(r"\]\(#slot-([a-z-]+)\)", slides)
    slots = json.loads((WORKSHOP / "data" / "metadata" / "media-slots.json").read_text())
    assert len(references) == len(set(references))
    assert set(references) == set(slots)
    for slot in slots.values():
        assert slot["purpose"]
        if slot["src"] is None:
            assert slot["type"] is None
            continue
        assert slot["type"] in {"image", "video"} and slot["alt"]
        asset = (WORKSHOP / slot["src"]).resolve()
        assert asset.is_relative_to((WORKSHOP / "assets").resolve()) and asset.is_file()


def test_learning_diagrams_are_present_and_distinct_from_recordings() -> None:
    slides = (WORKSHOP / "docs" / "class-presentation.fa.md").read_text(encoding="utf-8")
    script = (WORKSHOP / "assets" / "presentation.js").read_text(encoding="utf-8")
    diagrams = re.findall(r"\]\(#diagram-([a-z-]+)\)", slides)
    assert diagrams == ["workflow", "splits", "metrics", "comparison"]
    assert all(f"'{name}':" in script for name in diagrams)
    assert "مثال آموزشی" in script
