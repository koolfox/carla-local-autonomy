"""Direct-file worker startup must not depend on incidental site imports."""
import subprocess
import sys
from pathlib import Path


def test_encoder_discovery_in_clean_interpreter(tmp_path):
    # Discovery intentionally checks specs without importing the native packages.
    for name in ("numpy", "cv2"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('must remain lazy')\n")
    worker = Path(__file__).resolve().parents[1] / "carla_vision/native/world_worker.py"
    script = """
import sys, types
sys.path.insert(0, sys.argv[2])
module = types.ModuleType('worker_probe')
sys.modules[module.__name__] = module
with open(sys.argv[1], encoding='utf-8') as source:
    exec(compile(source.read(), sys.argv[1], 'exec'), module.__dict__)
assert module._camera_encoder_modules_present(), 'encoder falsely unavailable'
assert 'numpy' not in sys.modules and 'cv2' not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", script, str(worker), str(tmp_path)],
                   check=True, capture_output=True, text=True, timeout=15)
