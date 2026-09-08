from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from .assigner import SatellaAssigner
from .data import load_dataset_yaml, image_to_label_path, read_yolo_labels


def scan(images, nc, image_size, assigner):
    class_counts = Counter()
    level_counts = Counter()
    sizes = []
    boxes_total = 0
    empty = 0
    missing = 0

    for image_path in images:
        label_path = image_to_label_path(image_path)
        if not label_path.exists():
            missing += 1
        labels = read_yolo_labels(label_path, nc)
        if len(labels) == 0:
            empty += 1
            continue
        for row in labels:
            cls, _, _, wn, hn = row.tolist()
            w = wn * image_size
            h = hn * image_size
            size = math.sqrt(max(w * h, 1e-9))
            sizes.append(size)
            class_counts[int(cls)] += 1
            import torch
            level = assigner.choose_level(torch.tensor([0.0, 0.0, w, h]))
            level_counts[level] += 1
            boxes_total += 1

    return {"images": len(images), "boxes": boxes_total, "empty": empty, "missing": missing, "class_counts": class_counts, "level_counts": level_counts, "sizes": np.asarray(sizes, dtype=float)}


def print_split(name, stats, names, image_size):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Images: {stats['images']:,}")
    print(f"Boxes: {stats['boxes']:,}")
    print(f"Empty-label images: {stats['empty']:,}")
    print(f"Missing label files: {stats['missing']:,}")
    if stats["boxes"]:
        print("Classes:")
        for i, class_name in enumerate(names):
            print(f"  {i}: {class_name}: {stats['class_counts'][i]:,}")
        s = stats["sizes"]
        print(f"Equivalent box size @{image_size} (sqrt area): median {np.median(s):.1f}px, P10 {np.percentile(s,10):.1f}px, P90 {np.percentile(s,90):.1f}px")
        total = max(stats["boxes"], 1)
        print("Expected v0.2 assignment by preferred size:")
        for level, label in enumerate(("P2 stride 4", "P3 stride 8", "P4 stride 16")):
            n = stats["level_counts"][level]
            print(f"  {label}: {n:,} ({100*n/total:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if args.imgsz < 128 or args.imgsz % 32 != 0:
        raise ValueError("--imgsz must be >= 128 and divisible by 32")
    cfg = load_dataset_yaml(args.data)
    assigner = SatellaAssigner(image_size=args.imgsz)
    print(f"Dataset: {cfg.yaml_path.name}")
    print(f"Classes ({cfg.num_classes}): {cfg.names}")
    train = scan(cfg.train, cfg.num_classes, args.imgsz, assigner)
    val = scan(cfg.val, cfg.num_classes, args.imgsz, assigner)
    print_split("TRAIN", train, cfg.names, args.imgsz)
    print_split("VALIDATION", val, cfg.names, args.imgsz)
    print("\nPASS: data.yaml and YOLO label files are readable by SatellaDet.")


if __name__ == "__main__":
    main()
