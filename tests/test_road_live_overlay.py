from dataclasses import replace

import numpy as np
import pytest
from test_drive_console import valid_start

from carla_vision.contracts import PerceptionResult
from carla_vision.display import OverlayRenderer
from carla_vision.operator.drive_contracts import DriveStartConfig
from carla_vision.segmentation.contracts import SegmentationResult
from carla_vision.segmentation.overlay import render_segmentation_overlay
from carla_vision.segmentation.worker import SegmentationFrameInput


def test_legend_leaves_bottom_left_clear_for_driving_hud():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = SegmentationResult(np.ones((720, 1280), dtype=np.uint8),
                              np.ones((720, 1280), dtype=np.float32), "road")
    plain = render_segmentation_overlay(image, mask, draw_legend=False)
    legend = render_segmentation_overlay(image, mask)
    assert np.array_equal(plain[-120:, :300], legend[-120:, :300])
    assert not np.array_equal(plain[340:380, :220], legend[340:380, :220])


def test_road_and_detection_render_preserves_exact_source():
    image = np.zeros((120, 320, 3), dtype=np.uint8)
    detection = PerceptionResult(7, 42, 1.0, 2.0, 2.1, (), image, "custom-model")
    submitted = SegmentationFrameInput.from_perception(detection)
    assert submitted.perception is detection
    assert submitted.frame == 42
    mask = SegmentationResult(np.ones((120, 320), dtype=np.uint8),
                              np.ones((120, 320), dtype=np.float32), "road-model")
    road = render_segmentation_overlay(image, mask)
    output = OverlayRenderer().render(replace(detection, source_bgr=road),
                                      hud={"NAME": "Hesam Shani", "MODEL": "custom-model"})
    assert output.shape == image.shape
    assert output.any()
    assert not image.any()
    assert detection.source_bgr is image


def test_road_only_session_and_unsupported_backend(tmp_path):
    raw = valid_start()
    raw.update(road_enabled=True, road_backend="segformer", detector_enabled=False)
    config = DriveStartConfig.from_mapping(raw, workspace=tmp_path,
                                          expected_host=raw["host"], expected_port=raw["port"])
    assert config.road_enabled and not config.detector_enabled
    raw.update(road_backend="yolopv2", road_device="cpu")
    config = DriveStartConfig.from_mapping(raw, workspace=tmp_path,
                                          expected_host=raw["host"], expected_port=raw["port"])
    assert config.road_backend == "yolopv2"
    assert config.road_checkpoint is None
    raw["road_backend"] = "unknown"
    with pytest.raises(ValueError, match="road_backend"):
        DriveStartConfig.from_mapping(raw, workspace=tmp_path,
                                     expected_host=raw["host"], expected_port=raw["port"])
