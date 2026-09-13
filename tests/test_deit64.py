from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from carla_vision.contracts import Detection, DetectorConfig, DetectorMetadata, PerceptionResult
from carla_vision.detectors.deit64 import DeiT64Classifier, SignRecognitionDetector, sign_crop_box
from carla_vision.detectors.factory import create_detector
from carla_vision.detectors.sign_config import SignClassifierConfig, read_sign_ontology
from carla_vision.display import OverlayRenderer, detection_display_label


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "weights.pt").touch()
    _write_ontology(tmp_path / "labels.csv")
    return {"checkpoint": str(tmp_path / "weights.pt"), "ontology": str(tmp_path / "labels.csv")}


def _write_ontology(path, num_classes=64):
    stage_c = {64: "back", 65: "speed_limit_30", 66: "speed_limit_40", 67: "speed_limit_60"}
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["canonical_id", "canonical_name"])
        # Deliberately reversed to catch accidental row-order mappings.
        writer.writerows((i, stage_c.get(i, f"sign-{i}")) for i in reversed(range(num_classes)))


def test_config_and_ontology(settings, tmp_path):
    config = SignClassifierConfig.from_mapping(settings, workspace=tmp_path)
    assert config.confidence == 0.7 and config.crop_scale == 4
    assert config.show_rejection_status is False
    assert SignClassifierConfig.from_mapping({**settings, "show_rejection_status": True}).as_dict()["show_rejection_status"] is True
    assert read_sign_ontology(config.ontology)[35] == "sign-35"
    assert SignClassifierConfig.from_mapping(config.as_dict()) == config
    with pytest.raises(ValueError):
        SignClassifierConfig.from_mapping(settings, workspace=tmp_path / "child")


@pytest.mark.parametrize("patch", [
    {"confidence": float("nan")}, {"confidence": True}, {"confidence": 1.01},
    {"crop_scale": 0}, {"crop_scale": 5}, {"crop_scale": "4"},
    {"checkpoint": ""}, {"ontology": None}, {"runtime": "pickle"},
    {"show_rejection_status": "false"}, {"show_rejection_status": 1},
])
def test_invalid_config_rejected(settings, patch):
    with pytest.raises((ValueError, TypeError)):
        SignClassifierConfig.from_mapping({**settings, **patch})


@pytest.mark.parametrize("rows", [
    "canonical_id,canonical_name\n0,Stop\n",
    "canonical_id,canonical_name\n0,Stop\n0,Other\n",
    "canonical_id,canonical_name\n0,\n",
    "id,name\n0,Stop\n",
])
def test_invalid_ontology_rejected(tmp_path, rows):
    path = tmp_path / "bad.csv"
    path.write_text(rows)
    with pytest.raises(ValueError):
        read_sign_ontology(path)


def test_stage_c_ontology_keeps_all_four_new_classes(tmp_path):
    path = tmp_path / "labels.csv"
    _write_ontology(path, 68)
    labels = read_sign_ontology(path)
    assert labels[:64] == tuple(f"sign-{i}" for i in range(64))
    assert labels[64:] == ("back", "speed_limit_30", "speed_limit_40", "speed_limit_60")


@pytest.mark.parametrize("class_count", [63, 65, 67, 69])
def test_other_class_counts_rejected(tmp_path, class_count):
    path = tmp_path / "labels.csv"
    _write_ontology(path, class_count)
    with pytest.raises(ValueError, match="canonical IDs"):
        read_sign_ontology(path)


def _sign():
    # Coarse/fine heads may disagree; only the fine sign label gates DeiT.
    return Detection(0, "vehicles", 0.5, (30, 30, 40, 40), source_class_id=8,
        attributes={"fine_label": "traffic_signs", "fine_confidence": 0.8,
                    "coarse_confidence": 0.6, "quality": 0.4})


def _cascade(detections):
    detector = Mock(name="base")
    detector.name = "M9"
    detector.metadata = DetectorMetadata("M9", "m9-hierarchical", None, "cpu", 800, 0.1)
    detector.infer.return_value = detections
    classifier = Mock()
    classifier.config = SimpleNamespace(crop_scale=4.0)
    classifier.identity = {"name": "DeiT-64", "confidence": 0.7}
    classifier.predict.return_value = [{"class_id": 35, "label": "STOP", "confidence": 0.95, "accepted": True}]
    return SignRecognitionDetector(detector, classifier), detector, classifier


def test_cascade_preserves_m9_scores_and_uses_source_rgb_crop():
    sign = _sign()
    car = Detection(0, "vehicles", 0.9, (1, 2, 3, 4))
    model, detector, classifier = _cascade((car, sign))
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    before = image.copy()
    output = model.infer(image)
    assert output[0] is car
    assert replace(output[1], attributes=sign.attributes) == sign
    assert output[1].attributes["sign_classification"]["crop_xyxy"] == [15, 15, 55, 55]
    crop = classifier.predict.call_args.args[0][0]
    assert crop.shape == (40, 40, 3)
    assert crop[0, 0].tolist() == [255, 0, 0]
    assert np.array_equal(before, image)
    assert detection_display_label(output[1]).endswith("\nSign:STOP\nDeiT:0.95")
    assert "->" not in detection_display_label(output[1])
    assert model.metadata.extra["sign_classifier"]["name"] == "DeiT-64"
    model.close()
    model.close()
    detector.close.assert_called_once()
    classifier.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer(image)


def test_stage_c_cascade_identity_and_back_are_real_model_outputs():
    _, base, classifier = _cascade((_sign(),))
    classifier.identity = {"name": "DeiT-68", "num_classes": 68}
    classifier.predict.return_value = [{"class_id": 64, "label": "back", "confidence": .95, "accepted": True}]
    model = SignRecognitionDetector(base, classifier)
    assert model.name == model.metadata.name == "M9 + DeiT-68"
    assert model.metadata.extra["sign_classifier"]["num_classes"] == 68
    assert detection_display_label(model.infer(np.zeros((80, 100, 3), np.uint8))[0]).endswith(
        "Sign:back\nDeiT:0.95")  # Never reinterpret back as unknown/unaccepted.


def test_no_sign_does_not_run_classifier_and_invalid_crop_is_unknown():
    outside = replace(_sign(), xyxy=(90, 90, 100, 100))
    model, _, classifier = _cascade((outside,))
    output = model.infer(np.zeros((50, 50, 3), np.uint8))
    assert output[0].attributes["sign_classification"]["accepted"] is False
    classifier.predict.assert_not_called()
    model._detector.infer.return_value = ()
    assert model.infer(np.zeros((50, 50, 3), np.uint8)) == ()
    classifier.predict.assert_not_called()


def test_detection_only_camera_skips_deit_even_when_sign_boxes_are_present():
    sign = _sign()
    model, detector, classifier = _cascade((sign,))
    image = np.zeros((80, 100, 3), np.uint8)
    assert model.infer_without_signs(image) == (sign,)
    classifier.predict.assert_not_called()
    assert model.detection_only_name == detector.name
    assert "sign_classification" in model.infer(image)[0].attributes
    classifier.predict.assert_called_once()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer_without_signs(image)


def test_notebook_per_box_labels_preserve_each_detection_and_separate_scores():
    model, _, _ = _cascade((_sign(),))
    result = model.infer(np.zeros((80, 100, 3), np.uint8))[0]
    weaker = replace(result, confidence=.2, attributes={**result.attributes,
        "sign_classification": {"label": "OTHER", "confidence": .99, "accepted": True}})
    assert detection_display_label(result) == (
        "F:traffic_signs\nPf:0.80 C:vehicles\nPc:0.60\nQ:0.40 S:0.50\nSign:STOP\nDeiT:0.95")
    assert detection_display_label(weaker).endswith("S:0.20\nSign:OTHER\nDeiT:0.99")
    assert "->" not in detection_display_label(result)
    assert detection_display_label(_sign()) == (
        "F:traffic_signs\nPf:0.80\nC:vehicles\nPc:0.60\nQ:0.40\nS:0.50")


def test_all_detections_get_local_notebook_labels_without_corner_summary(monkeypatch):
    model, _, _ = _cascade((_sign(),))
    image = np.zeros((720, 1280, 3), np.uint8)
    first = replace(model.infer(image)[0], xyxy=(100, 420, 140, 460))
    second = replace(first, xyxy=(700, 420, 740, 460), confidence=.2)
    car = replace(first, xyxy=(1050, 420, 1200, 600), attributes={
        "fine_label": "car", "fine_confidence": .8, "coarse_confidence": .6, "quality": .4})
    frame = PerceptionResult(1, 2, 0, 0, 0, (first, second, car), image, model.name)
    renderer = OverlayRenderer()
    original = renderer._draw_detection_label
    labels = []
    def draw(image, text, **box):
        labels.append((text, box))
        original(image, text, **box)
    monkeypatch.setattr(renderer, "_draw_detection_label", draw)
    with_sign = renderer.render(frame, stale=False)
    without_sign = renderer.render(replace(frame, detections=()), stale=False)
    assert len(labels) == 3 and [box["x1"] for _, box in labels] == [100, 700, 1050]
    assert labels[1][0].endswith("S:0.20\nSign:STOP\nDeiT:0.95")
    assert "F:car" in labels[2][0] and "DeiT" not in labels[2][0]
    assert with_sign[420, 100].tolist() == [0, 255, 0]
    assert with_sign[270:420, 100:340].any() and with_sign[270:420, 700:940].any()
    assert not with_sign[:220, -500:].any()  # No separate Best sign panel.
    assert not without_sign[:220, -500:].any()
    assert not image.any()


def test_crop_bounds_and_low_confidence_overlay():
    assert sign_crop_box((0, 0, 10, 10), 20, 20, 4) == (0, 0, 20, 20)
    assert sign_crop_box((1, 1, 1, 2), 20, 20, 4) is None
    sign = replace(_sign(), attributes={**_sign().attributes,
        "sign_classification": {"accepted": False, "label": "STOP", "confidence": 0.2}})
    before = json.dumps(sign.attributes, sort_keys=True)
    assert detection_display_label(sign).endswith("\nSign:STOP\nDeiT:0.20")
    assert detection_display_label(sign, show_rejection_status=True).endswith("\nSign:unknown (unaccepted)\nDeiT:0.20")
    assert "STOP" not in detection_display_label(sign, show_rejection_status=True)
    assert json.dumps(sign.attributes, sort_keys=True) == before
    invalid = replace(sign, attributes={**sign.attributes, "sign_classification": {"accepted": False, "reason": "invalid_or_tiny_crop"}})
    assert "Sign:" not in detection_display_label(invalid)
    assert detection_display_label(invalid, show_rejection_status=True).endswith("Sign:unknown\nDeiT:unavailable")
    result = PerceptionResult(1, 2, 0, 0, 0, (sign,), np.zeros((80, 100, 3), np.uint8), "M9 + DeiT")
    rendered = OverlayRenderer().render(result, stale=False)
    assert rendered.any() and not result.source_bgr.any()


def test_factory_rejects_wrong_detector_without_loading_weights(settings):
    with pytest.raises(ValueError, match="requires M9"):
        create_detector(DetectorConfig("yolo", options={"sign_classifier": settings}))


def test_drive_serialization_retains_sign_and_both_head_scores():
    from carla_vision.operator.drive import DriveSession

    model, _, _ = _cascade((_sign(),))
    image = np.zeros((80, 100, 3), np.uint8)
    detections = model.infer(image)
    result = PerceptionResult(1, 2, 0, 0, 0, detections, image, model.name)
    stream = io.StringIO()
    owner = SimpleNamespace(_detections_written=0)
    DriveSession._write_detections(owner, stream, result)
    attrs = json.loads(stream.getvalue())["detections"][0]["attributes"]
    assert attrs["sign_classification"]["label"] == "STOP"
    assert attrs["fine_confidence"] == 0.8 and attrs["coarse_confidence"] == 0.6


def test_session_settings_reach_drive_and_old_sessions_still_work(settings, tmp_path):
    from carla_vision.operator.configuration import build_legacy_drive_request, session_defaults
    from carla_vision.operator.garage_drive import GarageDriveStartConfig

    session = session_defaults(detector_enabled=True)
    session["identity"]["runId"] = "deit-test"
    session["vehicle"]["blueprint"] = "vehicle.audi.tt"
    session["control"]["mode"] = "manual"
    session["perception"].update(detector="m9-hierarchical", imageSize=800,
        weights=settings["checkpoint"], signClassifier={**settings, "show_rejection_status": True})
    kwargs = dict(carla_host="127.0.0.1", carla_port=2000, worker_connected=True, capabilities={})
    request = build_legacy_drive_request(session, **kwargs)
    config = GarageDriveStartConfig.from_mapping(request, workspace=tmp_path,
        expected_host="127.0.0.1", expected_port=2000, world_worker_configured=True).base
    assert config.sign_classifier["crop_scale"] == 4.0
    assert config.sign_classifier["show_rejection_status"] is True
    assert config.manifest_config()["sign_classifier"]["checkpoint"] == settings["checkpoint"]
    from test_drive_cameras import rig

    session["recording"].update(cameraRig=rig(), cameraPerception={"rear": "signs", "front": "detections"})
    request = build_legacy_drive_request(session, **kwargs)
    multi = GarageDriveStartConfig.from_mapping(request, workspace=tmp_path,
        expected_host="127.0.0.1", expected_port=2000, world_worker_configured=True).base
    assert multi.recording_perception == {"rear": "signs", "front": "detections"}
    assert multi.manifest_config()["recording_perception"] == multi.recording_perception
    session["perception"].pop("signClassifier")
    assert "sign_classifier" not in build_legacy_drive_request(session, **kwargs)


@pytest.mark.parametrize("class_count", [64, 68])
def test_tensor_only_load_strict_head_and_preprocessing(settings, monkeypatch, class_count):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from PIL import Image
    from torch import nn
    from torchvision import transforms

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(384, 512), nn.SiLU(), nn.Dropout(.3), nn.Linear(512, class_count))

        def forward(self, inputs):
            self.inputs = inputs
            return self.head(torch.zeros((len(inputs), 384)))

    prototype = SmallModel()
    with torch.no_grad():
        prototype.head[3].weight.zero_()
        prototype.head[3].bias.zero_()
        prototype.head[3].bias[-1] = 1
    from pathlib import Path
    _write_ontology(Path(settings["ontology"]), class_count)
    # The Stage C training loop saves metrics and optimizer state too, not only
    # weights. NumPy 1.x/2.x scalar metadata must not force unsafe pickle loading.
    torch.save({"model_state_dict": prototype.state_dict(), "epoch": 13,
                "carla_macro_f1": np.float64(.95), "history": [{"train_loss": np.float32(.1)}],
                "optimizer_state_dict": {"state": {}}, "scaler_state_dict": {}}, settings["checkpoint"])
    allowed_before = set(torch.serialization.get_safe_globals())
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load)
    create = Mock(side_effect=lambda *args, **kwargs: SmallModel())
    monkeypatch.setattr(timm, "create_model", create)
    model = DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    create.assert_called_once_with("deit_small_patch16_224", pretrained=False, num_classes=class_count)
    assert load.call_args.kwargs["weights_only"] is True
    assert set(torch.serialization.get_safe_globals()) == allowed_before
    assert model.identity["num_classes"] == class_count
    assert model.identity["name"] == f"DeiT-{class_count}"
    assert len(model.identity["checkpoint_sha256"]) == 64
    crop = np.random.default_rng(7).integers(0, 255, (57, 91, 3), dtype=np.uint8)
    output = model.predict([crop] * 17)
    expected = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(Image.fromarray(crop))
    torch.testing.assert_close(model._model.inputs[0], expected, rtol=0, atol=0)
    assert len(output) == 17 and model._model.inputs.shape[0] == 1
    assert all(row["accepted"] is False for row in output)
    assert all(row["class_id"] == class_count - 1 for row in output)
    assert output[0]["label"] == ("speed_limit_60" if class_count == 68 else "sign-63")
    assert model.predict([]) == []
    torch.save({"model_state_dict": {"head.weight": torch.zeros(2, 2)}}, settings["checkpoint"])
    with pytest.raises(ValueError, match="state_dict"):
        DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    incomplete = prototype.state_dict()
    del incomplete["head.0.weight"]
    torch.save({"model_state_dict": incomplete}, settings["checkpoint"])
    with pytest.raises(RuntimeError, match="Missing key"):
        DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")


@pytest.mark.parametrize("class_count,ontology_count", [(64, 68), (68, 64)])
def test_checkpoint_ontology_mismatch_fails_before_model_creation(settings, monkeypatch, class_count, ontology_count):
    from pathlib import Path
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    _write_ontology(Path(settings["ontology"]), ontology_count)
    torch.save({"model_state_dict": {"head.3.weight": torch.zeros(class_count, 512),
                                    "head.3.bias": torch.zeros(class_count)}}, settings["checkpoint"])
    create = Mock()
    monkeypatch.setattr(timm, "create_model", create)
    with pytest.raises(ValueError, match=f"checkpoint has {class_count} classes but ontology has {ontology_count}"):
        DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    create.assert_not_called()


def test_numpy_metric_support_does_not_allow_arbitrary_pickle_globals(settings):
    import pickle
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    allowed_before = set(torch.serialization.get_safe_globals())
    torch.save({"model_state_dict": {}, "executable": eval}, settings["checkpoint"])
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    assert set(torch.serialization.get_safe_globals()) == allowed_before


def test_stage_c_real_timm_checkpoint_roundtrip(settings):
    """Architecture/serialization smoke test, not accuracy evidence for trained Stage C."""
    from pathlib import Path
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    nn = torch.nn
    _write_ontology(Path(settings["ontology"]), 68)
    prototype = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=68)
    prototype.head = nn.Sequential(nn.Linear(384, 512), nn.SiLU(), nn.Dropout(.3), nn.Linear(512, 68))
    with torch.no_grad():
        prototype.head[3].weight.zero_()
        prototype.head[3].bias.zero_()
        prototype.head[3].bias[67] = 12
    torch.save({"model_state_dict": prototype.state_dict(), "epoch": 13}, settings["checkpoint"])
    del prototype
    model = DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    try:
        prediction, = model.predict([np.zeros((48, 80, 3), np.uint8)])
        assert prediction["class_id"] == 67
        assert prediction["label"] == "speed_limit_60"
        assert prediction["confidence"] > .99 and prediction["accepted"] is True
        assert model._model.training is False
    finally:
        model.close()


def test_no_unsafe_fallback_for_unknown_checkpoint(settings, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    loader = Mock(side_effect=RuntimeError("unsafe checkpoint refused"))
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(RuntimeError, match="unsafe checkpoint"):
        DeiT64Classifier(SignClassifierConfig.from_mapping(settings), device="cpu")
    assert loader.call_count == 1 and loader.call_args.kwargs["weights_only"] is True


def test_factory_closes_classifier_if_m9_fails(settings, monkeypatch):
    pytest.importorskip("torch")
    from carla_vision.detectors import deit64, m9_hierarchical

    classifier = Mock()
    monkeypatch.setattr(deit64, "DeiT64Classifier", Mock(return_value=classifier))
    monkeypatch.setattr(m9_hierarchical, "M9HierarchicalDetector", Mock(side_effect=ValueError("bad M9")))
    with pytest.raises(ValueError, match="bad M9"):
        create_detector(DetectorConfig("m9-hierarchical", options={"sign_classifier": settings}))
    classifier.close.assert_called_once()


def test_sign_checkpoint_not_offered_as_detector_and_single_hud_identity(tmp_path, monkeypatch):
    from carla_vision.operator import catalog
    from carla_vision.operator.drive import overlay_identity

    for directory in ("deit64", "deit68"):
        (tmp_path / "models" / directory).mkdir(parents=True)
        (tmp_path / "models" / directory / "sign.pt").touch()
    (tmp_path / "models" / "m9.pt").touch()
    monkeypatch.setattr(catalog, "probe_endpoint", lambda *args, **kwargs: False)
    result = catalog.build_catalog(tmp_path, carla_host="127.0.0.1", carla_port=2000)
    assert result["weights"] == ["models/m9.pt"]
    for name in ("DeiT-64", "DeiT-68"):
        hud = overlay_identity("m9.pt", "YOLOPv2", sign_classifier=name)
        assert hud == {"Author": "Marjan Shahchera at University of Kashan", "MODEL": f"m9.pt + {name} + YOLOPv2"}
    assert overlay_identity("m9.pt")["MODEL"] == "m9.pt"
    assert overlay_identity(None, "YOLOPv2", sign_classifier="DeiT-68")["MODEL"] == "YOLOPv2"
