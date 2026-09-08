"""Run the notebook M9 adapter on one image before connecting to CARLA."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from ..contracts import DetectorConfig
from ..display import detection_display_label
from .factory import create_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory for overlay and detections")
    args = parser.parse_args()
    source = cv2.imread(str(args.image))
    if source is None:
        parser.error(f"cannot read image: {args.image}")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    model = create_detector(DetectorConfig(
        backend="m9-hierarchical", weights=args.weights, device=args.device,
        image_size=800, confidence=args.confidence,
    ))
    try:
        started = time.monotonic()
        detections = model.infer(source)
        elapsed = time.monotonic() - started
        overlay = source.copy()
        for detection in detections:
            x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 210, 240), 2)
            cv2.putText(overlay, detection_display_label(detection),
                        (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (50, 210, 240), 1, cv2.LINE_AA)
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
