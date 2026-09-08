# Dataset Format

SatellaDet v0.2.1 consumes YOLO-style detection datasets.

## Directory example

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

For each image, the corresponding label file uses the same stem with a `.txt` extension.

## Label rows

Each row contains five values:

```text
class_id x_center y_center width height
```

Coordinates are normalized to 0–1.

Example:

```text
0 0.514 0.487 0.032 0.028
1 0.223 0.742 0.081 0.065
```

Images without objects may use an empty label file. Missing label files are also interpreted as zero objects by the loader, but the dataset checker reports them separately.

## data.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

The `path` may be absolute or relative to `data.yaml`.
