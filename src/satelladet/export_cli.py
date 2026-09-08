from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .model import SatellaDetN
from .exporter import export_onnx_from_model


def _checkpoint_state(ckpt):
    if isinstance(ckpt, dict):
        if ckpt.get("ema") is not None:
            return ckpt["ema"]
        if ckpt.get("model") is not None:
            return ckpt["model"]
    return ckpt


def main():
    p = argparse.ArgumentParser(description="Export a SatellaDet checkpoint to ONNX.")
    p.add_argument("--weights", required=True, help="Path to a SatellaDet .pt checkpoint")
    p.add_argument("--output", required=True, help="Destination .onnx file")
    p.add_argument("--imgsz", type=int, default=None, help="Override input size; otherwise read from checkpoint")
    p.add_argument("--classes", type=int, default=None, help="Override class count; otherwise infer from checkpoint names")
    args = p.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    saved_names = ckpt.get("names") if isinstance(ckpt, dict) else None

    image_size = int(args.imgsz or saved_args.get("imgsz", 640))
    if image_size < 128 or image_size % 32 != 0:
        raise ValueError("Image size must be >=128 and divisible by 32")

    if args.classes is not None:
        num_classes = int(args.classes)
    elif saved_names:
        num_classes = len(saved_names)
    else:
        raise ValueError("Could not infer class count; pass --classes")

    model = SatellaDetN(num_classes=num_classes)
    model.load_state_dict(_checkpoint_state(ckpt), strict=True)
    output = export_onnx_from_model(model, args.output, image_size=image_size)
    n = sum((image_size // s) ** 2 for s in model.strides)
    print(f"Exported: {Path(output).resolve()}")
    print(f"Input: [1, 3, {image_size}, {image_size}]")
    print(f"Output: [1, {4 + num_classes}, {n}]")


if __name__ == "__main__":
    main()
