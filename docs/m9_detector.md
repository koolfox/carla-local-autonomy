# Run the trained M9 detector

The adapter implements `stage3cm5new.ipynb` cell 93, the corrected locked
test pipeline. It uses the certified `hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt`
checkpoint, PIL RGB bicubic resize to 800×800, float32 /255, original fine logits,
Query-FiLM coarse and quality heads, and score `fine^1.20 * coarse^0.50 * quality^1.10`.
The four output categories are vehicles, road_users, two_wheelers, traffic_controls.
Fine class IDs and individual score components remain in detection attributes.

Install on the **Operator computer**:

```sh
uv sync --extra vision --extra m9 --group dev
```

This pins Ultralytics 8.4.137, matching training. Restart the Operator after
changing its dependencies. No Windows World Worker change is required.

Test the checkpoint without CARLA:

```sh
uv run --extra vision --extra m9 python -m carla_vision.detectors.m9_check \
  --weights models/hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt \
  --image /absolute/path/to/your/camera-image.jpg \
  --device cpu --output runs/m9-image-check
```

Use an existing camera image and a new output directory. The command saves
`overlay.jpg` and `detections.json`, including model identity and inference time.

In Garage → Vision, enable Detection overlay, select **M9 Hierarchical RT-DETR**,
then select the checkpoint. Image size must be 800. A fused score filter of zero
reproduces the notebook candidate gate (fine >0.10, top100); increasing the filter
changes the display operating point. Start with CPU; MPS/CUDA can be selected when
available. Choose Traffic Manager for vehicle control and Detections for the view.
Your detector observes RGB while Traffic Manager supplies driving controls.

Do not select the ordinary RT-DETR backend for this checkpoint: it does not run
the custom hierarchical heads. The file contains a legacy Python model object;
the adapter only deserializes the exact SHA-256 recorded by the training notebook.
Other checkpoints need an explicitly supported architecture and inference recipe.

Lane markings and road-surface segmentation are separate from this detector.

## Independent labels and road overlay

M9 labels show both independent head predictions and confidences, e.g.
`Coarse: vehicles 80% | Fine: car 90%`. There is no hierarchy arrow or claim
that the two head predictions necessarily agree.
Canonical coarse class IDs and fused confidence remain unchanged; `fine_label`
and `coarse_label` are also saved in detection attributes. Head disagreements
are not silently relabeled. The certified notebook/checkpoint has nine fine
classes: car, bus, truck, person, rider, bike, motor, traffic_lights, traffic_signs.
It has no TrafficBoard class.

Road/lane perception is independent of M9. See [Road models](road_models.md)
for YOLOP, YOLOPv2 and SegFormer selection, supported devices, preprocessing
and model storage. Combined masks and boxes use the same camera frame; raw
streaming is independent. Loading and model errors appear in the overlay toolbar.

Install on the operator/ML computer with `uv sync --extra vision --extra m9 --extra segmentation --group dev`.
No Windows World Worker update is needed. Restart the operator after updating.
The recorded overlay includes small top-left name and model labels. Raw recordings
remain untouched. Road-model identity is retained as `road-model-metadata.json`.

## Optional traffic-sign recognition: M9 + DeiT-64 / DeiT-68

In **Garage → Vision**, enable Detection overlay, select M9 and its detector
checkpoint, then enable **Read traffic signs · DeiT**. Start a new session
and select the Detections view. Settings apply at session start, not by reloading
models on every slider movement. The default is off; existing sessions still work.

Keep these files on the **Operator/ML computer**, not the Windows Worker:

```text
models/deit64/deit64_stageB_blocks10_11_best.pt
models/deit64/ontology_final_64.csv
```

The classifier checkpoint is not a detector: select it in the DeiT settings,
not the main M9 Weights selector. Files inside `models/deit64/` and `models/deit68/` are excluded from
that detector dropdown. Model binaries and personal ontologies remain local,
ignored assets; they are not included in Git commits.

Install the optional dependency into your **existing** environment:

```sh
uv pip install --python .venv/bin/python -e '.[m9,deit64]'
```

Restart the Operator afterward. When using `uv sync`, append `--extra deit64`
to your existing extras so other enabled model runtimes stay installed. The
lockfile records the tested timm version. No Worker update or CARLA API change
is needed for this classifier.

If a running Operator says `session.perception has unknown fields: signClassifier`,
its old Python process is serving the new static UI. Stop/restart the **Mac/ML
Operator only** with your usual command and reload the page. Garage preview never
sends the sign stage, and disabled stages are omitted for compatibility; actually
running DeiT requires the updated backend process.

### Updated CARLA Stage C checkpoint (68 classes)

`carla-on-deit-ptsd.ipynb` fine-tunes the same DeiT Small architecture, with
**68 trained outputs**, and saves `deit68_carla_stageC_best.pt` plus per-epoch
checkpoints. Both formats are supported, including the supplied
`deit68_carla_stageC_epoch15.pt`. Keep the model
paired with its **verified** 68-class ontology. In that notebook, after
**BUILD VERIFIED 68-CLASS ONTOLOGY**, export:

```python
pd.DataFrame(
    sorted(official_names.items()), columns=["canonical_id", "canonical_name"]
).to_csv("ontology_final_68.csv", index=False)
```

Use `official_names`, which preserves IDs 0–63 from the original CSV. Do **not**
use the earlier handwritten `CLASS_NAMES_68` dictionary: several original names
there differ from the training ontology. The four added classes are:

| ID | Label |
| --- | --- |
| 64 | `back` |
| 65 | `speed_limit_30` |
| 66 | `speed_limit_40` |
| 67 | `speed_limit_60` |

`back` is a real sign-back class, not an unknown/unaccepted UI status. Existing
`Maximum Speed 30/60/90` and `STOP` classes retain IDs 10/13/16/35; we do not merge
or rename any class at inference.

Place the exported files on the Operator computer:

```text
models/deit68/deit68_carla_stageC_epoch15.pt
models/deit68/ontology_final_68.csv
```

Under **Vision → Read traffic signs → DeiT model files & crop context**, set
**both** paths to these files, then start a new session. The same two paths work
with `--sign-checkpoint` / `--sign-ontology` in the image-check command below.
The detector dropdown still selects **M9**, not DeiT. Existing DeiT-64 paths
remain the default so updating the app does not break working installations.
Select the actual checkpoint you want to evaluate; epoch 15 is not automatically
treated as the notebook's best-validation checkpoint.
Model metadata and live/recorded HUDs identify the loaded variant as `DeiT-68`.
Selected rig cameras share that same classifier; no Windows Worker update is needed.

Class count is checked against `head.3.weight` and `head.3.bias` before building
the model. Pairing 64-class labels with a 68-class checkpoint (or vice versa)
fails clearly; no outputs are dropped and no head is expanded/randomly initialized.
The notebook's training checkpoint envelope (`model_state_dict`, optimizer,
history, etc.) is supported. Loading remains `weights_only=True`; only NumPy
floating scalar/dtype metadata is explicitly allowlisted for sklearn metrics.
Unknown executable globals still fail, without an unsafe retry.

### What is preserved and what is added

- Existing M9 inference, fine gate, weighted score, boxes, and independent
  coarse/fine confidences are unchanged. Only `fine_label == traffic_signs`
  detections are classified, even when the coarse head disagrees.
- Crops come from the exact original RGB source frame. Defaults follow the
  final integration cells of `thesis-m9-rtdetr-to-deit64.ipynb`: expand width
  and height by 4, round and clip to image bounds, resize with PIL bilinear
  to 224×224, and apply ImageNet normalization. 4× in each dimension can
  include roughly 16× the box area; this is configurable, not a claim that
  this context is optimal for CARLA.
- The fixed model is `deit_small_patch16_224` with a
  `384 → 512 → SiLU → Dropout(0.3) → 64 or 68` head. Loading uses
  `pretrained=False`, `weights_only=True`, and strict state-dict validation.
  No hub weights are downloaded and no unsafe load fallback is allowed.
- Ontology class IDs must be exactly 0–63 or 0–67, matching the checkpoint. Names are mapped by `canonical_id`,
  not CSV row order. This cannot prove that a different CSV belongs to a
  checkpoint; keep the training ontology paired with its weights. Both notebooks
  use RGB → bilinear 224×224 → ImageNet normalization at validation. Stage C's
  training-only horizontal flips and color jitter are never used for inference;
  that notebook does not specify a new M9 crop multiplier, so the existing
  configurable context is preserved.
- Every displayed M9 detection gets the notebook's per-box label: lime outline,
  black caption and white multiline `F`, `Pf`, `C`, `Pc`, `Q`, `S` values,
  plus `Sign` and `DeiT` when a sign prediction exists. Scores use two decimals
  and the line grouping of `draw_prediction_with_deit`. Captions are attached
  above/below the object, with edge-aware placement. There is no separate
  **Best sign** panel. Coarse and fine heads remain independent, without arrows.
  The existing detection confidence slider is still the only detection filter;
  the notebook's hard-coded 0.50 gate is not added.
- **Vision → Read traffic signs → Show unknown / unaccepted statuses** controls
  display only and defaults **off**. Off shows the actual predicted sign name
  and DeiT confidence even below the sign threshold. On substitutes
  `unknown (unaccepted)` below that threshold. When a crop is invalid/too small
  to classify, off omits the sign lines; on shows `unknown` / `unavailable`.
  No invented label or confidence is produced. The setting applies next session
  to both live and recorded overlays, including selected rig cameras.
- `detections.jsonl` retains `attributes.sign_classification` (predicted ID,
  label, confidence, acceptance, crop coordinates) and both M9 head scores.
  The acceptance flag is unchanged by hiding UI statuses: displaying a predicted
  name does **not** certify it as accepted. `detector-metadata.json` records file SHA-256 hashes,
  preprocessing, thresholds, device and library versions. Source boxes remain
  unchanged and raw RGB recording remains unannotated.

Sign confidence is an **uncalibrated classifier softmax**, not a joint
detection-and-recognition probability. Confident classifications of false M9
detections are still possible; measure recognition on labelled sign crops and
the complete cascade separately. This stage is advisory and cannot brake,
steer, or change Traffic Manager behavior.

### Test a saved image without CARLA

```sh
.venv/bin/python -m carla_vision.detectors.m9_check \
  --weights models/hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt \
  --image /absolute/path/to/camera-image.jpg \
  --sign-checkpoint models/deit64/deit64_stageB_blocks10_11_best.pt \
  --sign-ontology models/deit64/ontology_final_64.csv \
  --sign-confidence 0.7 --sign-crop-scale 4 \
  --device cpu --output runs/m9-deit-image-check
```

Choose a new output directory each time. It saves an exact-frame overlay,
detections (including rejected sign predictions), model identity, and total
cascade inference time. This single-image timing is not a throughput benchmark.
Add `--show-sign-statuses` to opt into unknown/unaccepted captions; scripts can
use `OverlayRenderer(show_rejection_status=True)` for the same behavior.

For your own scripts, use the existing `create_detector(DetectorConfig(...))`
contract and pass the stage through `options`:

```python
from pathlib import Path
import cv2
from carla_vision.contracts import DetectorConfig
from carla_vision.detectors import create_detector

detector = create_detector(DetectorConfig(
    backend="m9-hierarchical",
    weights=Path("models/hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt"),
    image_size=800, confidence=0.25, device="cpu",
    options={"sign_classifier": {
        "checkpoint": "models/deit64/deit64_stageB_blocks10_11_best.pt",
        "ontology": "models/deit64/ontology_final_64.csv",
        "confidence": 0.7, "crop_scale": 4,
    }},
))
try:
    image = cv2.imread("camera-image.jpg")  # full-resolution BGR uint8
    if image is None:
        raise ValueError("Cannot read camera-image.jpg")
    for detection in detector.infer(image):
        print(detection.label, detection.attributes.get("sign_classification"))
finally:
    detector.close()
```

Implementation: `detectors/sign_config.py` validates configuration and labels;
`detectors/deit64.py` owns classifier loading, crops and enrichment. The existing
factory, session configuration and renderer remain the integration boundaries.
The front feed and [selected Drive rig cameras](native_research_jobs.md#detections-and-sign-reading-on-selected-cameras)
use this same renderer and cascade. Select Detections or Detections + sign reading
per camera; all selected cameras share one serial model scheduler. The cascade
serializes calls because M9's hook capture is mutable, and batches at most 16 sign
crops at a time. Raw streaming is independent of inference; more cameras do not
guarantee real-time overlay FPS. Cross-camera object fusion remains separate work.
