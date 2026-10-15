"""Test-only helpers: re-sign deliberate corruptions to test semantic checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def ref(path: Path, relative: str) -> dict:
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def rehash_dataset(root: Path) -> None:
    """Not a repair utility. Deliberately removes hash signals in adversarial tests."""
    path = root / "dataset.json"
    data = json.loads(path.read_text())
    for sample in data["samples"]:
        for name in ("rgb", "metadata", "instance_mask", "yolo_label"):
            relative = sample[name]["path"]
            sample[name].update(ref(root / relative, relative))
    index = root / "checksums.sha256"
    relatives = [line.split("  ", 1)[1] for line in index.read_text().splitlines() if line]
    # Restore all canonical references when a test replaced the old index.
    for relative in ("annotations/instances.coco.json", "ontology.json", "data.yaml"):
        if (root / relative).is_file():
            relatives.append(relative)
    for sample in data["samples"]:
        relatives.extend(
            sample[name]["path"] for name in ("rgb", "metadata", "instance_mask", "yolo_label")
        )
    index.write_text(
        "".join(
            f"{ref(root / relative, relative)['sha256']}  {relative}\n"
            for relative in sorted(set(relatives))
        )
    )
    data["checksum_index"].update(ref(index, "checksums.sha256"))
    write_json(path, data)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        artifact.update(ref(root / artifact["path"], artifact["path"]))
    write_json(manifest_path, manifest)
