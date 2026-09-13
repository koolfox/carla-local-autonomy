from carla_vision.contracts import Detection
from carla_vision.display import detection_display_label
from carla_vision.operator.drive import overlay_identity


def test_independent_head_scores_not_fused_confidence():
    detection = Detection(0, "vehicles", .25, (0, 0, 20, 20), attributes={
        "fine_label": "person", "coarse_confidence": .8, "fine_confidence": .9,
    })
    assert detection_display_label(detection) == "F:person\nPf:0.90\nC:vehicles\nPc:0.80\nQ:n/a\nS:0.25"
    assert detection.confidence == .25


def test_standard_detector_label_unchanged():
    assert detection_display_label(Detection(0, "car", .75, (0, 0, 20, 20))) == "car 75%"


def test_neighboring_notebook_captions_choose_free_space_and_stay_in_frame():
    import numpy as np

    from carla_vision.display import OverlayRenderer

    image = np.full((384, 640, 3), 73, np.uint8)
    renderer = OverlayRenderer()
    occupied = []
    label = "F:traffic_signs\nPf:0.80 C:traffic_controls\nPc:0.90\nQ:0.60 S:0.40\nSign:STOP\nDeiT:0.70"
    renderer._draw_detection_label(image, label, x1=300, y1=180, x2=320, y2=200, occupied=occupied)
    renderer._draw_detection_label(image, label, x1=315, y1=190, x2=335, y2=210, occupied=occupied)
    a, b = occupied
    assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    for left, top, right, bottom in occupied:
        assert 0 <= left < right <= 640 and 0 <= top < bottom <= 384
    assert np.any(np.all(image == [255, 255, 255], axis=2))
    assert np.any(np.all(image == [0, 0, 0], axis=2))


def test_one_author_source_for_all_overlay_modes():
    expected = "Marjan Shahchera at University of Kashan"
    for models in [("M9", None), ("M9", "YOLOP"), (None, "YOLOP")]:
        hud = overlay_identity(*models)
        assert hud["Author"] == expected
        assert set(hud) == {"Author", "MODEL"}
    assert overlay_identity("M9", "YOLOP")["MODEL"] == "M9 + YOLOP"
