# SatellaDet Architecture

SatellaDet-N is a lightweight anchor-free detector built from independently implemented PyTorch modules.

## Backbone

The backbone uses convolutional downsampling and `SatellaBlock`, a split-transform-fuse residual block:

1. 1×1 projection
2. channel split
3. one retained branch
4. transformed branch with two 3×3 convolutions
5. concatenation
6. 1×1 fusion
7. residual addition

The backbone produces feature maps at strides 4, 8, and 16.

## Neck

A bidirectional top-down / bottom-up fusion neck combines P2, P3, and P4 features. P2 is deliberately retained for small-object detection.

## Head

Each feature level is projected to a shared head width. Classification and box prediction use separate shared towers.

Box regression predicts positive left/top/right/bottom distances from each grid location. The deployment wrapper decodes those distances to center-based `cx,cy,w,h` boxes in input pixels.

Classification produces logits during training and sigmoid probabilities in the ONNX deployment wrapper.

## Prediction locations

For a square input of size `S`, the dense prediction count is:

```text
(S/4)^2 + (S/8)^2 + (S/16)^2
```

Examples:

- 640 → 33,600 locations
- 1280 → 134,400 locations

## Current scope

SatellaDet-N is the only size in v0.2.1. The public `build_model()` factory is intended to allow future sizes without changing user-facing construction code.
