"""Verified recorded RGB episode previews; no CARLA replay or vehicle actuation."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ..artifacts import RunArtifactTracker, fingerprint_file
from ..native.teacher_verify import verify_teacher_dataset
from .camera_views import view_path


def replay_teacher_episode(
    dataset: str | Path,
    *,
    episode_id: str | None,
    runs_root: str | Path,
    run_id: str,
    tile_width: int = 480,
) -> Path:
    """Create an indexed mosaic MP4 for the existing Garage recordings player."""
    if not 160 <= tile_width <= 1280 or tile_width % 2:
        raise ValueError("tile_width must be even and in [160, 1280]")
    root = Path(dataset).expanduser().resolve(strict=True)
    report = verify_teacher_dataset(root)
    if report["status"] != "passed":
        raise ValueError("teacher dataset verification failed: " + "; ".join(report["errors"]))
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    episode_ids = sorted({sample["episode_id"] for sample in manifest["samples"]})
    if episode_id is None:
        if len(episode_ids) != 1:
            raise ValueError("choose --episode-id when the dataset has multiple episodes")
        episode_id = episode_ids[0]
    samples = [sample for sample in manifest["samples"] if sample["episode_id"] == episode_id]
    if len(samples) < 2:
        raise ValueError("replay requires a selected episode with at least two samples")
    intervals = np.diff([sample["source_timestamp"] for sample in samples])
    period = float(intervals[0])
    if period <= 0 or not np.allclose(intervals, period, atol=1e-6, rtol=0):
        raise ValueError(
            "replay requires uniformly spaced timestamps; irregular gaps are not hidden"
        )
    camera_ids = manifest.get("rgb_camera_ids", ["front"])
    # Left/front/right is the viewing order; it is independent of storage order.
    order = {"front_left": 0, "front": 1, "front_right": 2}
    camera_ids = sorted(camera_ids, key=lambda name: (order.get(name, 3), name))
    columns = min(3, len(camera_ids))
    tile_height = 2 * round(tile_width * 9 / 16 / 2)
    height = math.ceil(len(camera_ids) / columns) * (tile_height + 32)
    size = (columns * tile_width, height)
    with RunArtifactTracker(
        runs_root,
        run_id=run_id,
        config={
            "object_type": "teacher_episode_replay",
            "dataset_dir": str(root),
            "dataset_manifest": fingerprint_file(root / "dataset.json"),
            "episode_id": episode_id,
            "camera_ids": camera_ids,
            "source": "recorded_rgb",
            "fps": 1.0 / period,
            "resized_for_preview_only": True,
        },
    ) as tracker:
        target = tracker.artifact_path("camera-rig.mp4")
        video = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 1 / period, size)
        if not video.isOpened():
            raise RuntimeError("OpenCV could not open the episode video encoder")
        frame_index = []
        try:
            for sample in samples:
                metadata = json.loads(view_path(root, sample["metadata"]["path"]).read_text())
                views = metadata.get("rgb_views", {"front": {"image": sample["rgb"]}})
                canvas = np.zeros((height, size[0], 3), dtype=np.uint8)
                for index, name in enumerate(camera_ids):
                    image = cv2.imread(str(view_path(root, views[name]["image"]["path"])))
                    if image is None:
                        raise ValueError(f"RGB image disappeared during replay: {name}")
                    scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
                    resized = cv2.resize(
                        image,
                        (
                            max(1, round(image.shape[1] * scale)),
                            max(1, round(image.shape[0] * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    x, y = index % columns * tile_width, index // columns * (tile_height + 32)
                    left = x + (tile_width - resized.shape[1]) // 2
                    top = y + 32 + (tile_height - resized.shape[0]) // 2
                    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
                    cv2.putText(
                        canvas,
                        f"{name} | frame {sample['carla_frame']}",
                        (x + 8, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                video.write(canvas)
                frame_index.append(
                    {
                        "video_frame": len(frame_index),
                        "sample_id": sample["sample_id"],
                        "carla_frame": sample["carla_frame"],
                        "source_timestamp": sample["source_timestamp"],
                        "rgb_views": {name: views[name]["image"] for name in camera_ids},
                    }
                )
        finally:
            video.release()
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("episode video encoder produced no output")
        index_path = tracker.artifact_path("frames.json")
        index_path.write_text(
            json.dumps({"schema_version": "1.0", "frames": frame_index}, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        report_path = tracker.artifact_path("verification.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tracker.register_artifact(target, role="teacher_multicamera_video")
        tracker.register_artifact(index_path, role="recorded_frame_index")
        tracker.register_artifact(report_path, role="source_dataset_verification")
    return target.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tile-width", type=int, default=480)
    args = parser.parse_args(argv)
    print(replay_teacher_episode(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
