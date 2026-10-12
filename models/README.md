# Bundled model assets

These assets are published at the maintainer's request so a new clone has the
existing detector, sign-classifier, road/lane and predicted-depth weights.
Ten weight files total **732,845,619 bytes (about 699 MiB)**. Weight payloads use
GitHub Git LFS; class tables, configurations and notices are ordinary Git files.
No weight was retrained, converted or deserialized for this publication.

Install [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage),
run `git lfs install` before cloning, and run `git lfs pull` from the repository
root. An LFS-enabled clone downloads the weights; a pointer-only checkout is
not ready to load models. Before this delivery is merged, clone with
`--branch codex/repository-audit-samples` or check out that branch first.

## Models and runtime paths

| Model | Path under `models/` | Weight size | Required profile |
| --- | --- | ---: | --- |
| Custom hierarchical M9 | `hierarchical_rtdetr_m9_precal_m6_query_film_img800.pt` | 127.7 MiB | `m9` |
| DeiT-64 Stage B signs | `deit64/deit64_stageB_blocks10_11_best.pt` and `deit64/ontology_final_64.csv` | 83.6 MiB | `deit64` with `m9` |
| DeiT-68 Stage C signs | `deit68/deit68_carla_stageC_epoch15.pt` and `deit68/ontology_final_68.csv` | 112.5 MiB | `deit64` with `m9` |
| RT-DETR-L | `rtdetr-l.pt` | 63.4 MiB | `vision` |
| YOLO26n | `yolo26n.pt` | 5.3 MiB | `vision` |
| YOLOP road/lane | `yolop/yolop-640-640.onnx` | 34.2 MiB | `segmentation` |
| YOLOPv2 road/lane | `yolopv2/yolopv2.pt` | 149.1 MiB | `segmentation` |
| SegFormer B0 | The two pinned snapshots described below | 14.2 + 14.3 MiB | `segmentation` |
| Depth Anything V2 metric outdoor Small | The pinned snapshot described below | 94.6 MiB | `voxel` |

The same model classes and adapters already exist in `carla_vision/`; shipping a
checkpoint does not create a new driving policy or prove its accuracy. Use the
[M9/DeiT guide](../docs/m9_detector.md), [road-model guide](../docs/road_models.md)
and [RGB-depth voxel guide](../docs/camera_voxel_world_fa.md) for supported inputs
and selections. New environments still install their Python dependencies from
`uv.lock`; a virtual environment is not part of this asset set.

## Pinned Hugging Face snapshots

The existing cache paths are retained so the current adapters can find files
without a new loader or a dependency on the original user's home directory:

- `huggingface/models--nvidia--segformer-b0-finetuned-cityscapes-1024-1024/snapshots/a8df754056355c08d3d8c9c4f5b5988e4bd2f676/`
  contains the safetensors weights, configuration and image processor. This is
  the revision pinned by the current SegFormer adapter.
- `huggingface/models--nvidia--segformer-b0-finetuned-cityscapes-1024-1024/snapshots/21b3847fae21ddee674abd31129307b6a1235bd9/`
  retains the original PyTorch `.bin` snapshot and matching configuration for
  completeness. The current adapter uses the safetensors revision above.
- `huggingface/models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf/snapshots/fd2c22027eaf20374204f14099b8341e1925ad39/`
  contains metric-depth safetensors weights, configuration and image processor.

Snapshot entries are regular files, so Windows does not need symlink privileges.
Duplicated `blobs/` payloads, `.no_exist` markers, locks and macOS metadata are
not needed by these snapshots and are excluded. Their local originals remain
available in the maintainer's workspace. The snapshot file bytes are unchanged.

To use retained Hugging Face models without remote metadata requests, set this
before launching the Operator on macOS/Linux:

```bash
export HF_HUB_OFFLINE=1
```

In PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

Run from the repository root with its default workspace. If selecting a local
SegFormer path explicitly, use the safetensors snapshot directory above.
Offline mode applies to models already present; it does not install dependencies
or make the remote CARLA simulator available.

## Integrity and provenance

[`assets.json`](assets.json) records every retained asset's path, SHA-256,
byte size, storage method, model group and available source identity. Hashes
identify the exact local weights supplied; filenames are not treated as
independent proof of a training recipe or an upstream checkpoint match.
The two YOLOP-family hashes match the adapters' existing pinned identities.

On macOS, inspect the retained-byte inventory after LFS download:

```bash
cd models
shasum -a 256 -c checksums.sha256
```

The [checksum index](checksums.sha256) covers full weight bytes, configurations,
class tables, notices, this guide and the asset inventory. A missing LFS download
fails these byte hashes. This check does not load a model, run inference, or
validate live-CARLA behavior. Those remain deferred to the next validation session.

## Upstream notices

- YOLOP and YOLOPv2 retain their upstream MIT notices in
  [`notices/YOLOP-LICENSE`](notices/YOLOP-LICENSE) and
  [`notices/YOLOPv2-LICENSE`](notices/YOLOPv2-LICENSE).
- SegFormer retains the [NVIDIA SegFormer license](notices/SegFormer-LICENSE)
  and [pinned model card](notices/SegFormer-MODEL-CARD.md). Its upstream license
  limits use to non-commercial research or evaluation; bundling does not replace it.
- Depth Anything V2 Small retains its upstream
  [Apache-2.0 license](notices/DepthAnythingV2-LICENSE) and
  [pinned model card](notices/DepthAnythingV2-MODEL-CARD.md).
- The [Ultralytics AGPL-3.0 text](notices/Ultralytics-LICENSE) and source links in
  the inventory accompany the Ultralytics-based detector integration. This
  delivery does not assign a different license to upstream components.
- Custom M9/DeiT checkpoints and their class tables are the maintainer-supplied
  files, published unchanged at the maintainer's request; no new project or
  checkpoint license is assigned here.

[`notices/sources.json`](notices/sources.json) records the fetched notice URLs,
date and hashes. These notices are not a claim that the repository's remaining
release-license work is complete.

## Maintaining the bundle

`.gitignore` has an explicit allow-list for these assets, and `.gitattributes`
routes model weight extensions through LFS. New training checkpoints and other
downloads remain ignored. Add a new asset deliberately with its metadata,
provenance, notices and updated checksums; never place a large raw weight blob
directly into ordinary Git history.
