from carla_vision.contracts import Detection
from carla_vision.display import detection_display_label
from carla_vision.operator.drive import overlay_identity


def test_independent_head_scores_not_fused_confidence():
    detection = Detection(0, "vehicles", .25, (0, 0, 20, 20), attributes={
        "fine_label": "person", "coarse_confidence": .8, "fine_confidence": .9,
    })
    assert detection_display_label(detection) == "Coarse: vehicles 80% | Fine: person 90%"
    assert detection.confidence == .25


def test_standard_detector_label_unchanged():
    assert detection_display_label(Detection(0, "car", .75, (0, 0, 20, 20))) == "car 75%"


def test_one_author_source_for_all_overlay_modes():
    expected = "Marjan Shahchera at University of Kashan"
    for models in [("M9", None), ("M9", "YOLOP"), (None, "YOLOP")]:
        hud = overlay_identity(*models)
        assert hud["Author"] == expected
        assert set(hud) == {"Author", "MODEL"}
    assert overlay_identity("M9", "YOLOP")["MODEL"] == "M9 + YOLOP"
