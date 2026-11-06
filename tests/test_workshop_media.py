"""Keep the public workshop's media placeholders and animation assets coherent."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / 'docs' / 'workshop'


def test_every_workshop_media_slot_has_a_manifest_entry_and_valid_asset() -> None:
    slides = (WORKSHOP / 'docs' / 'class-presentation.fa.md').read_text(encoding='utf-8')
    references = re.findall(r'\]\(#slot-([a-z-]+)\)', slides)
    slots = json.loads((WORKSHOP / 'data' / 'metadata' / 'media-slots.json').read_text(encoding='utf-8'))
    assert len(references) == len(set(references))
    assert set(references) == set(slots)
    for slot in slots.values():
        assert slot['purpose']
        if slot['src'] is None:
            assert slot['type'] is None
            continue
        assert slot['type'] in {'image', 'video'} and slot['alt']
        asset = (WORKSHOP / slot['src']).resolve()
        assert asset.is_relative_to((WORKSHOP / 'assets').resolve()) and asset.is_file()


def test_illustrative_animations_have_subtitles_posters_and_matching_hashes() -> None:
    for name in ('frame-journey', 'perception-tasks'):
        assert (WORKSHOP / 'assets' / 'video' / f'{name}.mp4').is_file()
        assert (WORKSHOP / 'assets' / 'video' / f'{name}.jpg').is_file()
        assert (WORKSHOP / 'assets' / 'video' / f'{name}.fa.vtt').read_text(encoding='utf-8').startswith('WEBVTT\n')
    for line in (WORKSHOP / 'animations' / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        digest, path = line.split('  ', 1)
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
