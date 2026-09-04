from carla_vision.models.adapter import ModelMetadata, TorchModelAdapter


class DummyTorchModel:
    def __init__(self):
        self.mode = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.mode = "eval"
        return self

    def __call__(self, observation):
        return {"received": observation}


def test_torch_model_adapter_keeps_model_boundary():
    adapter = TorchModelAdapter(
        DummyTorchModel(),
        metadata=ModelMetadata(name="dummy"),
    )

    assert adapter.metadata.name == "dummy"
    assert adapter.infer({"rgb": "frame"}) == {"received": {"rgb": "frame"}}

    adapter.close()

    try:
        adapter.infer(None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed adapter must reject inference")
