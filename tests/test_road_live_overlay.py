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
    raw["road_backend"] = "unknown"
    with pytest.raises(ValueError, match="road_backend"):
        DriveStartConfig.from_mapping(raw, workspace=tmp_path,
                                     expected_host=raw["host"], expected_port=raw["port"])
