from __future__ import annotations

import argparse
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description="Verify a SatellaDet ONNX tensor contract on CPU.")
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    args = p.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    x = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    y = session.run([output_meta.name], {input_meta.name: x})[0]

    if y.ndim != 3 or y.shape[0] != 1:
        raise RuntimeError(f"Expected rank-3 output with batch 1, got {y.shape}")
    feature_len = min(y.shape[1], y.shape[2])
    detections = max(y.shape[1], y.shape[2])
    if feature_len < 5:
        raise RuntimeError(f"Expected 4+ classes features, got {y.shape}")
    num_classes = feature_len - 4
    expected_n = sum((args.imgsz // s) ** 2 for s in (4, 8, 16))
    if detections != expected_n:
        raise RuntimeError(f"Expected {expected_n} prediction locations, got {detections}")
    if not np.isfinite(y).all():
        raise RuntimeError("ONNX output contains NaN/Inf")

    # Normalize orientation to [1,F,N] for score inspection.
    ff = y if y.shape[1] == feature_len else np.transpose(y, (0, 2, 1))
    scores = ff[:, 4:, :]
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise RuntimeError("Class scores are not probabilities in [0,1]")

    print("PASS: SatellaDet ONNX contract verified")
    print(f"Input: {x.shape} float32")
    print(f"Output: {y.shape}")
    print(f"Classes: {num_classes}")
    print(f"Prediction locations: {detections}")


if __name__ == "__main__":
    main()
