from carla_vision.model_storage import model_directory
from carla_vision.segmentation.contracts import DEFAULT_SEGFORMER_B0_CHECKPOINT, SegmentationConfig
from carla_vision.segmentation.segformer import _checkpoint_options


def test_workspace_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert model_directory() == tmp_path / "models"
    other = tmp_path / "other"
    assert model_directory(other) == other / "models"


def test_segformer_uses_workspace_cache(tmp_path):
    _, options = _checkpoint_options(SegmentationConfig(
        checkpoint=DEFAULT_SEGFORMER_B0_CHECKPOINT,
        options={"workspace": str(tmp_path)},
    ))
    assert options["cache_dir"] == str(tmp_path / "models" / "huggingface")
