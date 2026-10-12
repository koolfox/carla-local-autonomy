# Historical samples

Published at the maintainer's request on 2026-10-12 from local material that was
previously excluded by the runtime-directory ignore rules. This archive contains
123 original payload files, totalling 10,563,700 bytes (about 10.1 MiB).
[`archive.json`](archive.json) records each original workspace path, byte size
and SHA-256 digest. [`checksums.sha256`](checksums.sha256) also covers the archive
descriptor and this guide. Original payload bytes and manifests are unchanged;
macOS `.DS_Store` files are omitted.

These are **historical development examples**. The simulator was unavailable
during publication; no fresh live-CARLA acceptance was performed. Names containing
`autopilot` are preserved filenames, not proof of trained vision-based autonomy.
The publication baseline `ad8e00b7` is not asserted to be the generating commit.
Original manifests retain their recorded Git identity and dirty state.

## Recordings and camera images

All five original MP4s use MPEG-4 Part 2 at 640 × 384 and 10 encoded frames per
second. Download them and open them in a player such as VLC if browser playback
is unavailable. Encoded duration is not the run's wall-clock or simulation
duration, and the legacy summaries do not retain exact-frame timestamps or a
complete model/checkpoint identity.

| Recording | Encoded duration | Retained summary | Associated image |
| --- | ---: | --- | --- |
| [Perception smoke](recordings/perception_smoke.mp4) | 1.5 s | [JSON](recordings/perception_smoke.json) | [Latest image](recordings/live_smoke.png) |
| [Driving smoke](recordings/yolo26_drive_smoke.mp4) | 1.9 s | [JSON](recordings/yolo26_drive_smoke.json) | [Latest image](recordings/live_drive_smoke.png) |
| [Asynchronous smoke](recordings/yolo26_async_smoke.mp4) | 5.6 s | [JSON](recordings/yolo26_async_smoke.json) | [Latest image](recordings/yolo26_async_smoke.png) |
| [Legacy autopilot-named run](recordings/yolo26_autopilot.mp4) | 3.2 s | [JSON](recordings/yolo26_autopilot.json) | [Latest image](recordings/live_yolo26.png) |
| [Legacy autopilot-named demo](recordings/yolo26_autopilot_demo.mp4) | 7.7 s | [JSON](recordings/yolo26_autopilot_demo.json) | [Latest image](recordings/yolo26_autopilot_latest.png) |

Additional paired examples: [raw front camera](recordings/front_camera_raw.png)
and [detector overlay](recordings/front_camera_yolo26.png).

Legacy summary `video`/`latest_frame` paths and local endpoint addresses are
historical provenance. Resolve files through the links above, not those absolute
paths. Actor IDs and filenames alone do not establish control ownership,
cleanup, detector accuracy or the model's contribution to driving.

## Labelled dataset

[`data/ds-carla0916-town10-pilot-v001/`](data/ds-carla0916-town10-pilot-v001/)
contains three stationary front-RGB frames, privileged instance-segmentation
teacher images, exact-frame metadata, 27 COCO annotations, derived YOLO labels,
an ontology and the original manifests/checksum index.

| File or directory | Purpose |
| --- | --- |
| [Dataset descriptor](data/ds-carla0916-town10-pilot-v001/dataset.json) | Sample identities, references, hashes and capture metadata |
| [Tracker manifest](data/ds-carla0916-town10-pilot-v001/manifest.json) | Historical producer and registered-artifact identities |
| [COCO annotations](data/ds-carla0916-town10-pilot-v001/annotations/instances.coco.json) | Canonical annotations |
| [RGB frames](data/ds-carla0916-town10-pilot-v001/images/unassigned/) | Three retained camera frames |
| [Teacher images](data/ds-carla0916-town10-pilot-v001/teacher/instance/unassigned/) | Privileged labels, not runtime model inputs |
| [Metadata](data/ds-carla0916-town10-pilot-v001/metadata/) | Sensor, frame and annotation provenance |
| [Dataset checksum index](data/ds-carla0916-town10-pilot-v001/checksums.sha256) | Original payload integrity index |

The samples belong to the `unassigned` partition. The retained `data.yaml`
names train/validation/test directories that this small capture does not contain;
do not use it unchanged as a training configuration or treat the sample as a
held-out benchmark. Annotation quality and current consumer compatibility need
separate review. Byte preservation alone does not establish either.

## Historical reports

All eight report snapshots are retained with their original JSON, tables,
Markdown, plots, manifests and checksum indexes:

- [2026-07-26 v001](historical-reports/rpt-framework-validation-20260726-v001/report.md)
- [2026-07-26 v002](historical-reports/rpt-framework-validation-20260726-v002/report.md)
- [2026-07-26 v003](historical-reports/rpt-framework-validation-20260726-v003/report.md)
- [2026-07-26 v004](historical-reports/rpt-framework-validation-20260726-v004/report.md)
- [2026-07-26 v005](historical-reports/rpt-framework-validation-20260726-v005/report.md)
- [2026-07-27 v006](historical-reports/rpt-framework-validation-20260727-v006/report.md)
- [2026-07-27 v007](historical-reports/rpt-framework-validation-20260727-v007/report.md)
- [2026-07-27 v008](historical-reports/rpt-framework-validation-20260727-v008/report.md)

These reports describe their original development snapshot. Their absolute
workspace paths and references to old runs, bundles and native kits are preserved;
the complete referenced source trees are not all available in this archive.
Report files can be inspected, but portable reproduction or deep source
verification is not claimed.

## Model class tables

- [64-class DeiT ontology](model-labels/ontology_final_64.csv)
- [68-class DeiT ontology](model-labels/ontology_final_68.csv)

The files retain the local model's label ordering. They do not contain model
weights or prove classifier accuracy. See the [M9/DeiT guide](../docs/m9_detector.md)
for matching checkpoint requirements.

## Integrity and scope

To inspect the archive's retained byte hashes on macOS:

```bash
cd samples
shasum -a 256 -c checksums.sha256
```

This checks file integrity only. It neither starts CARLA nor substitutes for
semantic dataset validation or a live acceptance run. The archive is outside
the Operator's canonical workspace roots and is not automatically listed as
new runtime results. Keep original runtime folders unchanged.

No root project license or separate sample license is currently declared. This
publication assigns no new license and contains no model weights, CARLA server
binary, local credentials, caches or generated native source packages.
