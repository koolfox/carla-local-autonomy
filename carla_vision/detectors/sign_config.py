"""Lightweight configuration for the optional, RGB-only DeiT sign stage."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SignClassifierConfig:
    checkpoint: Path
    ontology: Path
    confidence: float = 0.7
    crop_scale: float = 4.0
    show_rejection_status: bool = False

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, workspace: Path | None = None,
    ) -> SignClassifierConfig:
        if not isinstance(raw, Mapping):
            raise ValueError("sign_classifier must be an object")
        if set(raw) - {"checkpoint", "ontology", "confidence", "crop_scale", "show_rejection_status"}:
            raise ValueError("Unknown sign_classifier settings")
        show_status = raw.get("show_rejection_status", False)
        if not isinstance(show_status, bool):
            raise ValueError("sign_classifier.show_rejection_status must be boolean")
        paths: dict[str, Path] = {}
        for key, suffix in (("checkpoint", ".pt"), ("ontology", ".csv")):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"sign_classifier.{key} must be a local file path")
            path = ((workspace or Path.cwd()) / value).resolve(strict=True)
            if workspace is not None:
                path.relative_to(workspace.resolve())
            if not path.is_file() or path.suffix.lower() != suffix:
                raise ValueError(f"sign_classifier.{key} must be a {suffix} file")
            paths[key] = path
        numbers: dict[str, float] = {}
        for key, default, lower, upper in (
            ("confidence", 0.7, 0.0, 1.0), ("crop_scale", 4.0, 1.0, 4.0),
        ):
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"sign_classifier.{key} must be numeric")
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"sign_classifier.{key} must be in [{lower}, {upper}]")
            numbers[key] = float(value)
        return cls(**paths, **numbers, show_rejection_status=show_status)

    def as_dict(self) -> dict[str, Any]:
        return {"checkpoint": str(self.checkpoint), "ontology": str(self.ontology),
                "confidence": self.confidence, "crop_scale": self.crop_scale,
                "show_rejection_status": self.show_rejection_status}


def read_sign_ontology(path: Path) -> tuple[str, ...]:
    """Use canonical IDs, never CSV row order or alphabetic class order."""
    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"canonical_id", "canonical_name"} <= set(reader.fieldnames or ()):
            raise ValueError("DeiT ontology requires canonical_id and canonical_name columns")
        for row in reader:
            index = int(row["canonical_id"])
            name = (row["canonical_name"] or "").strip()
            if index in labels or not name:
                raise ValueError("DeiT ontology IDs must be unique and names nonempty")
            labels[index] = name
    if len(labels) not in {64, 68} or set(labels) != set(range(len(labels))):
        raise ValueError("DeiT ontology must contain exactly canonical IDs 0–63 (64 classes) or 0–67 (68 classes)")
    return tuple(labels[index] for index in range(len(labels)))
