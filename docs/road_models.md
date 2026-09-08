# Road and lane models

Garage menu → Vision → Road & lane: enable the overlay and choose YOLOP,
YOLOPv2, or SegFormer. Start a new session to use the selected model.
The existing object detector remains independent. No Windows Worker update
is needed for these Operator-side adapters.

YOLOPv2 uses the official CAIC-AD V0.0.1 TorchScript release. Leave the
checkpoint blank to download approximately 150 MiB once into the local model
cache, or select a workspace copy of that exact release. SHA-256 is verified
before loading; arbitrary TorchScript files are not accepted by this adapter.
Install the segmentation extra (`uv sync --extra segmentation`). CPU and CUDA
are supported; MPS is not advertised for this adapter. First download/loading
can take time. Compare latency on your machine before preferring it over YOLOP.

Both YOLOP adapters resize continuous head outputs before selecting classes,
instead of enlarging an already discrete mask. No morphological gap filling,
invented lane curves or temporal averaging is applied. Original RGB stays
unchanged; the overlay is model output, not CARLA ground truth. This improves
boundary reconstruction but cannot repair genuinely missed markings.

YOLOPv2 preprocessing is RGB /255 with stride-32 letterboxing, not the
ImageNet normalization used by our original YOLOP ONNX adapter. Padding is
removed using the actual image dimensions rather than the demo's fixed crop.
The two-class road and single-channel lane heads are restored independently;
lane markings take display priority. SegFormer Cityscapes does not predict
lane markings.

Upstream: https://github.com/CAIC-AD/YOLOPv2 (MIT). We use its released model
and documented inference convention without importing upstream Python code.
BDD100K performance is not a guarantee of CARLA accuracy or runtime speed.
