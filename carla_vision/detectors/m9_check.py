"""Run the notebook M9 adapter on one image before connecting to CARLA."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from ..contracts import DetectorConfig, PerceptionResult
from ..display import OverlayRenderer
from .factory import create_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--sign-checkpoint", type=Path, help="Optional DeiT-64 state_dict checkpoint")
    parser.add_argument("--sign-ontology", type=Path, help="DeiT canonical_id/canonical_name CSV")
    parser.add_argument("--sign-confidence", type=float, default=0.7)
    parser.add_argument("--sign-crop-scale", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory for overlay and detections")
    args = parser.parse_args()
    if bool(args.sign_checkpoint) != bool(args.sign_ontology):
        parser.error("--sign-checkpoint and --sign-ontology must be supplied together")
    options = {}
    if args.sign_checkpoint:
        options["sign_classifier"] = {
            "checkpoint": str(args.sign_checkpoint), "ontology": str(args.sign_ontology),
            "confidence": args.sign_confidence, "crop_scale": args.sign_crop_scale,
        }
    source = cv2.imread(str(args.image))
    if source is None:
        parser.error(f"cannot read image: {args.image}")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    model = create_detector(DetectorConfig(
        backend="m9-hierarchical", weights=args.weights, device=args.device,
        image_size=800, confidence=args.confidence, options=options,
    ))
    try:
        started = time.monotonic()
        detections = model.infer(source)
        elapsed = time.monotonic() - started
        overlay = OverlayRenderer().render(PerceptionResult(
            sequence=0, carla_frame=0, source_timestamp=0.0,
            source_received_monotonic=started, completed_monotonic=started + elapsed,
            detections=detections, source_bgr=source, detector_name=model.name,
        ), stale=False, hud={"MODEL": model.name})
        args.output.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(args.output / "overlay.jpg"), overlay):
            raise RuntimeError("could not save overlay")
        report = {"model": model.metadata.as_dict(), "image": str(args.image.resolve()),
                  "source_shape": list(source.shape), "inference_seconds": elapsed,
                  "detections": [asdict(item) for item in detections]}
        (args.output / "detections.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"M9: {len(detections)} detections in {elapsed:.3f}s; saved {args.output}")
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
