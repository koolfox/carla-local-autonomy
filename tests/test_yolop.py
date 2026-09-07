from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from carla_vision.segmentation.contracts import SegmentationConfig
from carla_vision.segmentation.yolop import YoloPSegmenter, preprocess


def test_rgb_normalization_and_padding():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    tensor, crop = preprocess(image, 4, 4)
    assert crop == (0, 1, 4, 2)
    assert tensor.shape == (1, 3, 4, 4)
    assert tensor[0, 0, 1, 0] == pytest.approx((1 - .485) / .229)
    assert tensor[0, 2, 1, 0] == pytest.approx(-.406 / .225)
    assert tensor[0, 0, 0, 0] == pytest.approx((114 / 255 - .485) / .229)


def test_lane_priority_and_original_frame_shape(tmp_path):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "sample.onnx"
    path.write_bytes(b"session mocked")
    road = np.zeros((1, 2, 4, 4), np.float32)
    road[:, 1] = 1
    lane = road.copy()
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 4, 4])]
        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in ("drive_area_seg", "lane_line_seg")]
        def run(self, names, feeds):
            return [road, lane]
    with patch("onnxruntime.InferenceSession", return_value=Session()):
        model = YoloPSegmenter(SegmentationConfig(backend="yolop", checkpoint=path))
    image = np.zeros((2, 4, 3), np.uint8)
    result = model.infer(image)
    assert result.class_ids.shape == (2, 4)
    assert np.all(result.class_ids == 2)
    assert not image.any()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer(image)
