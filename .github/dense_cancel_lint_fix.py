from pathlib import Path

# Keep the workspace existence check without retaining an otherwise-unused local.
path = Path(__file__).resolve().parents[1] / "carla_vision/operator/garage_density_acceptance.py"
text = path.read_text(encoding="utf-8")
old = '        workspace = Path(args.workspace).expanduser().resolve(strict=True)\n        worker = WorldWorkerClient(args.world_worker_url, token, timeout=5.0)\n'
new = '        Path(args.workspace).expanduser().resolve(strict=True)\n        worker = WorldWorkerClient(args.world_worker_url, token, timeout=5.0)\n'
if old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError("density runner lint target not found")
