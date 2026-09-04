from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"finalize target not found in {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "carla_vision/operator/garage_density_acceptance.py",
    '            "responsive_prepare_health": capabilities.get("responsive_prepare_health") is True,\n'
    '            "clean_worker_scene": clean,\n',
    '            "responsive_prepare_health": capabilities.get("responsive_prepare_health") is True,\n'
    '            "prepare_cancellation": capabilities.get("prepare_cancellation") is True,\n'
    '            "clean_worker_scene": clean,\n',
)

replace_once(
    "tests/test_garage_density_acceptance.py",
    "        self.crossing = 0.0\n",
    "        self.crossing = 0.0\n        self.prepare_cancellation = True\n",
)
replace_once(
    "tests/test_garage_density_acceptance.py",
    '                "prepare_cancellation": True,\n',
    '                "prepare_cancellation": self.prepare_cancellation,\n',
)
replace_once(
    "tests/test_garage_density_acceptance.py",
    '    with pytest.raises(ValueError, match="\\[0, 250\\]"):\n',
    '    with pytest.raises(ValueError, match=r"\\[0, 250\\]"):\n',
)
replace_once(
    "tests/test_garage_density_acceptance.py",
    "\ndef test_density_matrix_records_progress_dynamics_camera_cleanup_and_capacity() -> None:\n",
    "\ndef test_preflight_requires_prepare_cancellation_capability() -> None:\n"
    "    worker = FakeWorker()\n"
    "    worker.prepare_cancellation = False\n"
    "    manager = FakeManager(worker)\n"
    "    report = make_runner(manager, worker, tiers=(DensityTier(25, 20),)).run()\n\n"
    "    assert report[\"status\"] == \"fail\"\n"
    "    assert report[\"preflight\"][\"checks\"][\"prepare_cancellation\"] is False\n"
    "    assert report[\"tiers\"] == []\n\n\n"
    "def test_density_matrix_records_progress_dynamics_camera_cleanup_and_capacity() -> None:\n",
)

replace_once(
    ".github/workflows/ci.yml",
    "          tests/test_garage_acceptance.py\n          tests/test_garage_async.py\n",
    "          tests/test_garage_acceptance.py\n          tests/test_garage_density_acceptance.py\n          tests/test_garage_async.py\n",
)
replace_once(
    ".github/workflows/ci.yml",
    "          tests/test_observable_world_worker.py\n          tests/test_operator.py\n",
    "          tests/test_observable_world_worker.py\n          tests/test_prepare_cancellation.py\n          tests/test_operator.py\n",
)
replace_once(
    ".github/workflows/ci.yml",
    "          uv run carla-garage-acceptance --help\n          uv run carla-vision --help\n",
    "          uv run carla-garage-acceptance --help\n          uv run carla-garage-density-acceptance --help\n          uv run carla-vision --help\n",
)
