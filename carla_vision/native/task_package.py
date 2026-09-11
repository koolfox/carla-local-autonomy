"""Export a trusted teacher task package without copying models or local secrets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .research_jobs import digest, write_json

_ENTRY = """from pathlib import Path
import argparse, json, sys
p = argparse.ArgumentParser()
p.add_argument('--request', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
try:
    from carla_vision.native.tasks.teacher_capture import execute
    request = json.loads(args.request.read_text(encoding='utf-8'))
except Exception as error:
    result = {'schema_version': '1.0', 'status': 'failed',
              'cleanup_confirmed': True, 'error': f'Preflight: {type(error).__name__}: {error}'}
else:
    result = execute(request, args.output)
(args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
raise SystemExit(0 if result['status'] == 'succeeded' else 1)
"""


def build_teacher_task(destination: Path, *, version: str) -> Path:
    if not version.strip() or len(version) > 80:
        raise ValueError("provide a short nonempty task version")
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    (destination / "run.py").write_text(_ENTRY, encoding="utf-8")
    for path in sorted(source.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("source package contains a symlink")
        target = destination / "source" / "carla_vision" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # Source only: no weights, frontend bundle, credentials, or auto-installation.
    files = {
        path.relative_to(destination).as_posix(): digest(path)
        for path in sorted(destination.rglob("*.py"))
    }
    write_json(
        destination / "task.json",
        {
            "schema_version": "1.0",
            "id": "teacher_capture",
            "version": version,
            "entrypoint": "run.py",
            "world_access": "exclusive",
            "max_seconds": 900,
            "files": files,
            "requires": [
                "carla==0.9.16",
                "agents.navigation.behavior_agent",
                "numpy",
                "opencv-python",
                "msgpack",
                "networkx",
                "shapely",
            ],
        },
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="New directory named teacher_capture under the host task registry",
    )
    parser.add_argument("--version", required=True, help="Use a source commit or release version")
    args = parser.parse_args()
    path = build_teacher_task(args.destination, version=args.version)
    print(json.dumps({"task_directory": str(path), "manifest_sha256": digest(path / "task.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
