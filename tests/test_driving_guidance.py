from __future__ import annotations

import cv2
import numpy as np
import pytest

from carla_vision.operator.driving_guidance import (
    project_driving_guidance,
    render_driving_guidance_jpeg,
)


def _guidance() -> dict[str, object]:
    return {
        "available": True,
        "source": "assigned_route",
        "label": "ASSIGNED ROUTE",
        "lookahead_m": 20.0,
        "points": [
            {
                "center": [distance, 0.0, -1.5],
                "left": [distance, -1.75, -1.5],
                "right": [distance, 1.75, -1.5],
                "distance_m": distance,
            }
            for distance in (3.0, 8.0, 12.0, 20.0)
        ],
    }


def test_projects_carla_coordinates_into_the_rgb_camera() -> None:
    result = project_driving_guidance(
        _guidance(),
        camera_transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        width=640,
        height=360,
        fov=90.0,
        camera_sequence=9,
    )

    assert result["available"] is True
    assert result["source"] == "assigned_route"
    assert result["privileged"] is True
    assert result["feeds_control"] is False
    assert result["camera_sequence"] == 9
    assert len(result["center"]) == 4
    assert all(point[0] == pytest.approx(320.0) for point in result["center"])
    assert result["left"][0][0] < result["center"][0][0] < result["right"][0][0]
    assert result["center"][0][1] > result["center"][-1][1]
    assert result["steering_target"] == result["center"][2]


def test_discards_points_behind_the_camera() -> None:
    guidance = _guidance()
    guidance["points"] = [
        {
            "center": [-5.0, 0.0, -1.5],
            "left": [-5.0, -1.75, -1.5],
            "right": [-5.0, 1.75, -1.5],
            "distance_m": 0.0,
        },
        *guidance["points"],
    ]

    result = project_driving_guidance(
        guidance,
        camera_transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        width=640,
        height=360,
        fov=90.0,
        camera_sequence=2,
    )

    assert result["available"] is True
    assert len(result["center"]) == 4


def test_unavailable_guidance_remains_explicitly_non_actuating() -> None:
    result = project_driving_guidance(
        {"available": False, "reason": "no active scene"},
        camera_transform=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        width=640,
        height=360,
        fov=90.0,
        camera_sequence=-1,
    )

    assert result["available"] is False
    assert result["reason"] == "no active scene"
    assert result["privileged"] is True
    assert result["feeds_control"] is False


def test_browser_derivative_draws_on_the_exact_jpeg_without_mutating_raw_bytes() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    raw_jpeg = encoded.tobytes()
    guidance = {
        "available": True,
        "source": "traffic_manager_action_buffer",
        "frame_size": [320, 180],
        "center": [[160.0, 170.0], [160.0, 105.0], [160.0, 65.0]],
        "left": [[95.0, 170.0], [125.0, 105.0], [145.0, 65.0]],
        "right": [[225.0, 170.0], [195.0, 105.0], [175.0, 65.0]],
        "steering_target": [160.0, 105.0],
    }

    rendered = render_driving_guidance_jpeg(raw_jpeg, guidance)
    decoded = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert rendered != raw_jpeg
    assert decoded is not None
    assert int(decoded.sum()) > 0
    assert render_driving_guidance_jpeg(raw_jpeg, {"available": False}) == raw_jpeg
