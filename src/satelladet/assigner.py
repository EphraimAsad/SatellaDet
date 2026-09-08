from __future__ import annotations

import math
import torch


class SatellaAssigner:
    """
    Simple anchor-free v0.1 assigner.

    - Chooses P2/P3/P4 by closest preferred object size.
    - Assigns each GT to a grid cell near its centre.
    - Searches a 3x3 centre neighbourhood to avoid collisions between dense objects.
    """

    def __init__(
        self,
        strides=(4, 8, 16),
        preferred_sizes=(16.0, 40.0, 96.0),
        image_size: int = 640,
        reference_image_size: int = 640,
    ):
        self.strides = tuple(int(x) for x in strides)
        self.image_size = int(image_size)
        self.reference_image_size = int(reference_image_size)
        if self.image_size <= 0 or self.reference_image_size <= 0:
            raise ValueError("image_size and reference_image_size must be > 0")
        scale = self.image_size / float(self.reference_image_size)
        self.preferred_sizes = tuple(float(x) * scale for x in preferred_sizes)
        if len(self.strides) != len(self.preferred_sizes):
            raise ValueError("strides and preferred_sizes must have the same length")

    def choose_level(self, box: torch.Tensor) -> int:
        w = max(float(box[2] - box[0]), 1e-6)
        h = max(float(box[3] - box[1]), 1e-6)
        size = math.sqrt(w * h)
        scores = [abs(math.log(size / ref)) for ref in self.preferred_sizes]
        return min(range(len(scores)), key=scores.__getitem__)

    @staticmethod
    def _candidate_cells(cx: float, cy: float, stride: int, h: int, w: int, box: torch.Tensor):
        gx = int(cx / stride)
        gy = int(cy / stride)
        x1, y1, x2, y2 = map(float, box.tolist())
        candidates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x = gx + dx
                y = gy + dy
                if not (0 <= x < w and 0 <= y < h):
                    continue
                px = (x + 0.5) * stride
                py = (y + 0.5) * stride
                inside = (x1 <= px <= x2) and (y1 <= py <= y2)
                dist2 = (px - cx) ** 2 + (py - cy) ** 2
                candidates.append((0 if inside else 1, dist2, y, x))
        candidates.sort()
        return [(y, x) for _, _, y, x in candidates]

    def build_targets(self, outputs, targets, num_classes: int, device: torch.device):
        batch_size = outputs[0]["cls"].shape[0]
        level_targets = []
        for out in outputs:
            _, _, h, w = out["cls"].shape
            level_targets.append({
                "cls": torch.zeros((batch_size, num_classes, h, w), device=device),
                "box": torch.zeros((batch_size, 4, h, w), device=device),
                "pos": torch.zeros((batch_size, h, w), dtype=torch.bool, device=device),
            })

        assigned = 0
        collisions = 0
        for b, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            classes = target["classes"].to(device)
            if len(boxes):
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                order = torch.argsort(area, descending=True)
                boxes = boxes[order]
                classes = classes[order]

            for box, cls in zip(boxes, classes):
                level = self.choose_level(box)
                stride = self.strides[level]
                lt = level_targets[level]
                h, w = lt["pos"].shape[-2:]
                cx = float((box[0] + box[2]) * 0.5)
                cy = float((box[1] + box[3]) * 0.5)

                chosen = None
                for gy, gx in self._candidate_cells(cx, cy, stride, h, w, box.detach().cpu()):
                    if not bool(lt["pos"][b, gy, gx]):
                        chosen = (gy, gx)
                        break

                if chosen is None:
                    collisions += 1
                    continue

                gy, gx = chosen
                lt["pos"][b, gy, gx] = True
                lt["box"][b, :, gy, gx] = box
                lt["cls"][b, int(cls), gy, gx] = 1.0
                assigned += 1

        return level_targets, {"assigned": assigned, "collisions": collisions}
