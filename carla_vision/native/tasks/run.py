"""Internal built-in task entry point; run with the existing Worker interpreter."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Direct-file startup keeps optional collector dependencies out of the bridge.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if request["task"]["id"] != "teacher_capture":
            raise ValueError("unsupported built-in native task")
        required = {"numpy": "numpy", "cv2": "opencv-python-headless", "msgpack": "msgpack"}
        if not request["parameters"].get("dry_run", False):
            required.update(
                {
                    "carla": "matching CARLA wheel",
                    "agents.navigation.behavior_agent": "matching CARLA agents",
                }
            )
        for module, package in required.items():
            try:
                importlib.import_module(module)
            except ImportError as error:
                raise RuntimeError(
                    f"The Worker's Python is missing {module}: {error}. "
                    f"Add {package} to that same environment; no second environment is needed."
                ) from error
        from carla_vision.native.tasks.teacher_capture import execute
    except Exception as error:
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "cleanup_confirmed": True,
            "error": f"Preflight: {error}",
        }
    else:
        result = execute(request, args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
