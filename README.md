# SatellaDet

SatellaDet is a lightweight, anchor-free object detection framework implemented in PyTorch. It is designed for small-object detection, CPU-friendly ONNX deployment, and straightforward training from standard YOLO-format annotations.

This repository contains **framework code only**. It does not contain training images, datasets, private checkpoints, or pretrained model weights.

## Current release

**SatellaDet-N v0.2.1**

- P2 / P3 / P4 detection scales (strides 4, 8, 16)
- anchor-free LTRB box regression
- shared decoupled classification and box towers
- CIoU regression loss
- focal BCE classification loss
- EMA training weights
- per-class precision, recall, F1, mAP50 and mAP50-95
- count-oriented validation metrics
- SatellaScore composite checkpoint selection
- configurable fixed input sizes such as 640, 960 and 1280
- YOLO-format dataset support
- ONNX export with a simple `[1, 4+nc, N]` deployment head

## Installation

From a clone of this repository:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Once published to PyPI, the intended installation command is:

```bash
pip install satelladet
```

> The `satelladet` project name should be re-checked on PyPI immediately before publication; package-name availability can change.

## Python API

```python
import torch
from satelladet import build_model

model = build_model("n", num_classes=3).eval()
x = torch.zeros(1, 3, 640, 640)

with torch.no_grad():
    outputs = model(x)

for level in outputs:
    print(level["box"].shape, level["cls"].shape)
```

## Dataset format

SatellaDet reads YOLO-format bounding-box labels:

```text
class_id x_center y_center width height
```

Coordinates are normalized to 0–1.

Example `data.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for the expected directory structure.

## Check a dataset

```bash
satelladet-check --data /path/to/data.yaml --imgsz 640
```

## Train

```bash
satelladet-train \
  --data /path/to/data.yaml \
  --output model.onnx \
  --epochs 200 \
  --patience 50 \
  --imgsz 640 \
  --batch 4
```

Higher fixed resolutions are also supported:

```bash
satelladet-train \
  --data /path/to/data.yaml \
  --output model_1280.onnx \
  --epochs 200 \
  --imgsz 1280 \
  --batch 2
```

Input size must be at least 128 and divisible by 32. The model architecture and parameter count do not change when input resolution changes; the number of prediction locations does.

## Fine-tune and resume

Fine-tune from a checkpoint:

```bash
satelladet-train \
  --data /path/to/data.yaml \
  --output finetuned.onnx \
  --weights /path/to/best.pt
```

Changing resolution while fine-tuning is supported. For example, 640 weights can initialize a 1280 run.

A full `--resume` is intended to continue the same experiment and therefore requires the same image size as the saved checkpoint.

## Export a checkpoint

```bash
satelladet-export \
  --weights /path/to/best.pt \
  --output model.onnx
```

The exporter reads class count and image size from SatellaDet checkpoints when possible.

## Verify an ONNX export

```bash
satelladet-verify --model model.onnx --imgsz 640
```

The deployment wrapper emits a single tensor in the form:

```text
[1, 4 + num_classes, N]
```

with channels:

```text
[cx, cy, width, height, class_probability...]
```

Boxes are in input-image pixel coordinates and class probabilities are sigmoid-activated inside the exported graph. NMS is intentionally left outside the model.

## SatellaScore

SatellaScore is the default checkpoint-selection metric. It combines class-weighted F1, mAP50-95, mAP50, and a count-accuracy component. The weighting is configurable from the training CLI.

This is intended to make model selection less dependent on a single aggregate metric. Applications that do not care about count accuracy can set the count component to zero and rebalance the remaining weights.

## Privacy and datasets

The public framework does not ship datasets, images, checkpoints, experiment logs, or trained weights. Runtime configuration and checkpoint metadata also avoid storing absolute local paths by default.

Before publishing any separately trained model, independently review the training data rights, model metadata, and any logs bundled with it.

See [PRIVACY.md](PRIVACY.md).

## Status

SatellaDet v0.2.1 is an alpha framework. The API and training strategy may evolve as additional model sizes, assignment strategies, augmentations, and deployment targets are added.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
