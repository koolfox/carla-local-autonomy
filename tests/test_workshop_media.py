"""Check that public workshop placeholders and stated sample counts stay traceable."""

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


def test_retained_pilot_counts_are_supported_by_the_dataset() -> None:
    dataset = json.loads(
        (ROOT / "samples/data/ds-carla0916-town10-pilot-v001/dataset.json").read_text()
    )
    coco = json.loads(
        (
            ROOT / "samples/data/ds-carla0916-town10-pilot-v001/annotations/instances.coco.json"
        ).read_text()
    )
    assert dataset["sample_count"] == len(dataset["samples"]) == len(coco["images"]) == 3
    assert dataset["annotation_count"] == len(coco["annotations"]) == 27
    assert {sample["split"] for sample in dataset["samples"]} == {"unassigned"}
    assert len({sample["episode_id"] for sample in dataset["samples"]}) == 1
