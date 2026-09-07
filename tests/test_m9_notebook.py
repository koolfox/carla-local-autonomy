"""Regression checks for the corrected M9 notebook inference recipe."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from carla_vision.detectors.m9_hierarchical import (  # noqa: E402
    M9_ALPHA,
    M9_BETA,
    M9_GAMMA,
    _boxes_to_original_xyxy,
    _fused_score,
    _image_tensor,
    _load_certified_checkpoint,
)


def test_preprocessing_matches_notebook_pil_rgb_resize():
    image = np.random.default_rng(4).integers(0, 256, (81, 137, 3), dtype=np.uint8)
    before = image.copy()
    expected = np.array(Image.fromarray(image[:, :, ::-1]).resize((800, 800)), copy=True)
    expected = torch.from_numpy(expected).float().permute(2, 0, 1).unsqueeze(0) / 255
    actual = _image_tensor(image, device=torch.device("cpu"))
    assert torch.equal(actual, expected)
    assert np.array_equal(image, before)


def test_locked_formula_and_rectangular_frame_coordinates():
    assert (M9_ALPHA, M9_BETA, M9_GAMMA) == (1.2, 0.5, 1.1)
    fine, coarse, quality = [torch.tensor([v]) for v in (0.8, 0.7, 0.6)]
    assert _fused_score(fine, coarse, quality).item() == pytest.approx(0.8**1.2 * 0.7**0.5 * 0.6**1.1)
    box = _boxes_to_original_xyxy(torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
                                 source_width=1280, source_height=720)
    assert box.tolist() == [[[320, 180, 960, 540]]]


def test_unknown_checkpoint_rejected_before_deserialization(tmp_path: Path, monkeypatch):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not the certified checkpoint")
    def forbidden(*args, **kwargs):
        pytest.fail("unknown checkpoint must never reach torch.load")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="not the certified M9"):
        _load_certified_checkpoint(path)


def test_session_accepts_m9_and_rejects_wrong_image_size(tmp_path):
    from test_drive_console import CARLA_HOST, CARLA_PORT, valid_start

    from carla_vision.operator.drive_contracts import DriveStartConfig

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"contract validation does not deserialize")
    request = valid_start(detector="m9-hierarchical", weights="model.pt", image_size=800)
    options = dict(workspace=tmp_path, expected_host=CARLA_HOST, expected_port=CARLA_PORT)
    parsed = DriveStartConfig.from_mapping(request, **options)
    assert parsed.detector == "m9-hierarchical"
    assert parsed.weights == checkpoint
    with pytest.raises(ValueError, match="image_size=800"):
        DriveStartConfig.from_mapping({**request, "image_size": 640}, **options)
