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
