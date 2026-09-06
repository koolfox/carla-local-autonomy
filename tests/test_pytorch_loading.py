from __future__ import annotations

from pathlib import Path

import pytest

from carla_vision.pytorch_loading import (
    PyTorchCheckpointError,
    PyTorchStateDictError,
    extract_state_dict,
    load_state_dict_for_inference,
    load_weights_only_checkpoint,
    prepare_module_for_inference,
    resolve_torch_device,
)

torch = pytest.importorskip("torch")


def _linear(*, out_features: int = 2) -> torch.nn.Module:
    return torch.nn.Linear(3, out_features)


def test_load_direct_state_dict_round_trip_and_prepare_eval(tmp_path: Path) -> None:
    source = _linear()
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(-0.5)
    checkpoint = tmp_path / "weights.pt"
    torch.save(source.state_dict(), checkpoint)

    target = _linear()
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    target.train()

    device = load_state_dict_for_inference(target, checkpoint, device="cpu")

    assert str(device) == "cpu"
    assert target.training is False
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_load_envelope_requires_explicit_state_dict_key(tmp_path: Path) -> None:
    source = _linear()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": source.state_dict(), "epoch": 7}, checkpoint)

    target = _linear()
    load_state_dict_for_inference(
        target,
        checkpoint,
        device="cpu",
        state_dict_key="model_state_dict",
    )

    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_loader_does_not_guess_state_dict_key(tmp_path: Path) -> None:
    source = _linear()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": source.state_dict(), "epoch": 7}, checkpoint)

    with pytest.raises(PyTorchStateDictError, match="state-dict keys do not match"):
        load_state_dict_for_inference(_linear(), checkpoint, device="cpu")


def test_missing_explicit_state_dict_key_reports_available_keys() -> None:
    payload = {"model_state_dict": {"weight": torch.ones(1)}, "epoch": 7}

    with pytest.raises(
        PyTorchStateDictError,
        match="does not contain state_dict_key='state_dict'",
    ):
        extract_state_dict(payload, state_dict_key="state_dict")


def test_whole_module_pickle_has_no_unsafe_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "whole-module.pt"
    checkpoint.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    def fail_load(*_args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        raise RuntimeError("weights-only rejection")

    monkeypatch.setattr(torch, "load", fail_load)

    with pytest.raises(PyTorchCheckpointError, match="weights_only=True"):
        load_weights_only_checkpoint(checkpoint)

    assert len(calls) == 1
    assert calls[0]["weights_only"] is True
    assert str(calls[0]["map_location"]) == "cpu"


def test_strict_key_mismatch_fails_before_model_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"wrong.weight": torch.ones((2, 3)), "wrong.bias": torch.ones(2)}, checkpoint)

    with pytest.raises(PyTorchStateDictError, match="missing: bias, weight"):
        load_state_dict_for_inference(_linear(), checkpoint, device="cpu")


def test_shape_mismatch_is_wrapped_as_state_dict_error(tmp_path: Path) -> None:
    source = _linear(out_features=4)
    checkpoint = tmp_path / "weights.pt"
    torch.save(source.state_dict(), checkpoint)

    with pytest.raises(PyTorchStateDictError, match="state dict is incompatible"):
        load_state_dict_for_inference(_linear(out_features=2), checkpoint, device="cpu")


def test_prepare_module_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        prepare_module_for_inference(object(), {}, device="cpu")


def test_prepare_module_rejects_malformed_state_dict() -> None:
    with pytest.raises(PyTorchStateDictError, match="must contain a state-dict mapping"):
        prepare_module_for_inference(_linear(), [], device="cpu")
    with pytest.raises(PyTorchStateDictError, match="keys must be non-empty strings"):
        prepare_module_for_inference(_linear(), {1: torch.ones(1)}, device="cpu")


def test_unavailable_cuda_fails_before_checkpoint_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save(_linear().state_dict(), checkpoint)
    called = False

    def record_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "load", record_load)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        load_state_dict_for_inference(_linear(), checkpoint, device="cuda")

    assert called is False


def test_cpu_device_resolves_without_accelerator() -> None:
    assert str(resolve_torch_device("cpu")) == "cpu"
