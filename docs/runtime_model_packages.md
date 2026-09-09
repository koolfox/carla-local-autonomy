# Runtime model packages

This page covers **registered driving-policy packages**, not every perception
adapter. Built-in RT-DETR/M9 detectors and road segmenters have their own
configuration/checkpoint paths; see [M9](m9_detector.md) and [road models](road_models.md).

The Garage does not execute an arbitrary policy checkpoint by filename. A runnable
policy package is a directory under `models/` containing the artifact and a validated
`model.json` contract:

```text
models/
  my-policy/
    model.json
    policy.pt
```

This boundary makes the preprocessing, output contract, executable adapter,
artifact identity, supported devices, and operator trust decision explicit.
Discovery reads and validates JSON without importing the adapter or
deserializing a checkpoint. Stable hashes are verified when a package is
resolved for a run and again around model setup.

## Standard TorchScript control model

`torchscript_control_v1` receives a normalized `NCHW` image tensor and,
optionally, a `1x1` speed tensor. It must return exactly three finite controls
in `[throttle, steer, brake]` order. Throttle and brake are in `[0, 1]`; steer
is in `[-1, 1]`.

```json
{
  "schema_version": "1.0",
  "object_type": "runtime_model_package",
  "id": "my-torchscript-policy",
  "name": "My TorchScript Policy",
  "version": "1.0.0",
  "role": "driving_policy",
  "runtime": "torchscript_control_v1",
  "artifact": "policy.pt",
  "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
  "devices": ["cpu", "cuda", "mps"],
  "inputs": {
    "image": {
      "width": 320,
      "height": 180,
      "color": "rgb",
      "mean": [0.0, 0.0, 0.0],
      "std": [1.0, 1.0, 1.0]
    },
    "speed": {"enabled": true, "unit": "mps"}
  },
  "outputs": {"kind": "vehicle_control_v1"},
  "source": "Describe the training run, repository, and license here"
}
```

## Custom eager PyTorch or Python model

Use `python_factory` when a state dict, eager PyTorch module, custom temporal
model, or nonstandard preprocessing needs adapter code. The factory receives a
`ModelDriverConfig` and returns an object with `reset()`, `predict(observation)`,
and `close()` methods. `predict` receives `ModelObservation` containing only the
front BGR image, ego speed, frame identity, timestamp, and step duration. It
must return the common `ModelControl` shape.

```json
{
  "schema_version": "1.0",
  "object_type": "runtime_model_package",
  "id": "my-python-policy",
  "name": "My Python Policy",
  "version": "1.0.0",
  "role": "driving_policy",
  "runtime": "python_factory",
  "artifact": "checkpoint.pth",
  "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
  "factory": "my_models.policy:create_driver",
  "devices": ["cpu", "cuda"],
  "inputs": {
    "kind": "model_observation_v1",
    "options": {"history_frames": 4}
  },
  "outputs": {"kind": "vehicle_control_v1"},
  "source": "Describe the training run, repository, and license here"
}
```

For state-dict checkpoints, construct the architecture in the trusted factory
and use the shared conservative loader. The loader does not guess model
architecture, checkpoint keys, preprocessing, or output semantics:

```python
from carla_vision.model_driver import ModelDriverConfig
from carla_vision.pytorch_loading import load_state_dict_for_inference

from .architecture import MyPolicy


class Driver:
    def __init__(self, config: ModelDriverConfig) -> None:
        if config.checkpoint is None:
            raise ValueError("policy requires a checkpoint")

        self.model = MyPolicy(...)
        self.device = load_state_dict_for_inference(
            self.model,
            config.checkpoint,
            device=config.device,
            # Omit this argument when the checkpoint itself is the state dict.
            state_dict_key="model_state_dict",
        )
```

The shared loader deserializes on CPU with `weights_only=True`, requires an
exact state-dict key match for the model the factory constructed, moves the
module only after state validation, and switches it to evaluation mode. It
never retries with `weights_only=False`. A checkpoint that stores a whole
pickled module or requires custom pickle globals therefore needs a deliberate,
model-specific trusted adapter instead of an implicit generic fallback.

The factory still owns observation preprocessing, temporal state, the forward
signature, and conversion of model output to `ModelControl`. This is
intentional: arbitrary PyTorch models do not share one image size, color
transform, auxiliary-input signature, or control head.

The module containing the factory must be importable in the Operator Python
environment and backed by a regular fingerprintable Python source file. The
adapter source, manifest, and artifact are all recorded in the run lineage.

## Register and verify

Compute the artifact digest without loading it:

```bash
shasum -a 256 models/my-policy/policy.pt
```

On Windows PowerShell:

```powershell
Get-FileHash .\models\my-policy\policy.pt -Algorithm SHA256
```

Put the lowercase digest in `model.json`, restart or reload the Operator, and
inspect the registered model library. An invalid package appears with its exact
contract error and cannot be selected.

All current `.pt`, `.pth`, and Python-factory paths are executable trusted
content. The Garage requires a separate trust acknowledgement in addition to
the autonomy acknowledgement. A changed artifact, manifest, or adapter fails
closed; setup is attempted once; slow inference and stale frames resolve to a
service brake.

The current package contract covers direct driving policies. General object,
traffic-light/sign, drivable-area, and lane perception packages use the next
canonical scene-perception contract rather than pretending that every PyTorch
model has the same inputs and outputs.
