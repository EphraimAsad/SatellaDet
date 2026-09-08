from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetConfig:
    yaml_path: Path
    root: Path
    train: list[Path]
    val: list[Path]
    names: list[str]
    num_classes: int


def _resolve_entry(root: Path, entry, yaml_dir: Path) -> list[Path]:
    if isinstance(entry, (list, tuple)):
        out: list[Path] = []
        for item in entry:
            out.extend(_resolve_entry(root, item, yaml_dir))
        return out
    p = Path(str(entry))
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if p.is_dir():
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS)
    if p.is_file() and p.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            q = Path(raw)
            if not q.is_absolute():
                q = (p.parent / q).resolve()
            images.append(q)
        return images
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    raise FileNotFoundError(f"Could not resolve dataset entry: {entry!r} -> {p}")


def load_dataset_yaml(path: str | Path) -> DatasetConfig:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    yaml_dir = yaml_path.parent
    root_value = cfg.get("path", ".")
    root = Path(str(root_value))
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    if "train" not in cfg or "val" not in cfg:
        raise ValueError("data.yaml must contain both 'train' and 'val' entries.")
    names_raw = cfg.get("names")
    if isinstance(names_raw, dict):
        names = [str(names_raw[k]) for k in sorted(names_raw, key=lambda x: int(x))]
    elif isinstance(names_raw, list):
        names = [str(x) for x in names_raw]
    else:
        nc = int(cfg.get("nc", 1))
        names = [f"class_{i}" for i in range(nc)]
    nc = int(cfg.get("nc", len(names)))
    if nc != len(names):
        if names_raw is None:
            names = [f"class_{i}" for i in range(nc)]
        else:
            raise ValueError(f"data.yaml says nc={nc} but names contains {len(names)} entries.")
    train_images = _resolve_entry(root, cfg["train"], yaml_dir)
    val_images = _resolve_entry(root, cfg["val"], yaml_dir)
    if not train_images:
        raise ValueError("No training images found.")
    if not val_images:
        raise ValueError("No validation images found.")
    return DatasetConfig(yaml_path=yaml_path, root=root, train=train_images, val=val_images, names=names, num_classes=nc)


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.name.replace(image_path.suffix, ".txt")


def read_yolo_labels(path: Path, num_classes: int) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows: list[list[float]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed label {path}:{line_no}: expected 5 values, got {len(parts)}")
        cls = int(float(parts[0]))
        if not (0 <= cls < num_classes):
            raise ValueError(f"Class {cls} out of range in {path}:{line_no} (nc={num_classes})")
        cx, cy, w, h = map(float, parts[1:5])
        if w <= 0 or h <= 0:
            continue
        rows.append([float(cls), cx, cy, w, h])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    t = torch.tensor(rows, dtype=torch.float32)
    t[:, 1:] = t[:, 1:].clamp(0.0, 1.0)
    return t


class YoloDetectionDataset(Dataset):
    def __init__(self, image_paths: Iterable[Path], num_classes: int, image_size: int = 640, augment: bool = False):
        self.images = list(image_paths)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        label_path = image_to_label_path(image_path)
        labels = read_yolo_labels(label_path, self.num_classes)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if len(labels):
                labels[:, 2] = 1.0 - labels[:, 2]
        if self.augment:
            if random.random() < 0.7:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.7:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if len(labels):
            classes = labels[:, 0].long()
            cx = labels[:, 1] * self.image_size
            cy = labels[:, 2] * self.image_size
            w = labels[:, 3] * self.image_size
            h = labels[:, 4] * self.image_size
            x1 = (cx - w * 0.5).clamp(0, self.image_size)
            y1 = (cy - h * 0.5).clamp(0, self.image_size)
            x2 = (cx + w * 0.5).clamp(0, self.image_size)
            y2 = (cy + h * 0.5).clamp(0, self.image_size)
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            valid = (x2 > x1) & (y2 > y1)
            boxes = boxes[valid]
            classes = classes[valid]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)
        target = {"boxes": boxes, "classes": classes, "image_path": str(image_path)}
        return tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
