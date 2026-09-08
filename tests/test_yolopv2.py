from unittest.mock import patch

import numpy as np
import pytest

from carla_vision.segmentation.contracts import SegmentationConfig
from carla_vision.segmentation.yolopv2 import YoloPv2Segmenter, preprocess


def test_preprocess_rgb_unit_range_and_stride_padding():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, :, 2] = 255
    tensor, crop = preprocess(image)
    assert tensor.shape == (1, 3, 384, 640)
    assert crop == (0, 12, 640, 360)
    assert tensor[0, 0, 12, 0] == 1
    assert tensor[0, 2, 12, 0] == 0
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)


def test_untrusted_checkpoint_rejected_before_load(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not official")
    with patch("torch.jit.load") as loader:
        with pytest.raises(ValueError, match="checksum"):
            YoloPv2Segmenter(SegmentationConfig(backend="yolopv2", checkpoint=path))
        loader.assert_not_called()


def test_head_restoration_lane_priority_and_close():
    torch = pytest.importorskip("torch")
    model = object.__new__(YoloPv2Segmenter)
    model.name = "test"
    model._device = "cpu"
    def forward(tensor):
        _, _, h, w = tensor.shape
        road = torch.zeros(1, 2, h, w)
        road[:, 1] = 1
        lane = torch.ones(1, 1, h, w)
        return None, road, lane
    model._model = forward
    source = np.zeros((180, 320, 3), np.uint8)
    result = model.infer(source)
    assert result.class_ids.shape == (180, 320)
    assert np.all(result.class_ids == 2)
    assert np.all(result.confidence == 1)
    assert not source.any()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer(source)
