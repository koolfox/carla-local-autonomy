# Runtime model packages

The Garage does not execute an arbitrary checkpoint by filename. A runnable
model is a directory under `models/` containing the artifact and a validated
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

The module containing the factory must be importable in the Operator Python
environment and backed by a regular fingerprintable Python source file. The adapter source,
manifest, and artifact are all recorded in the run lineage.

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
