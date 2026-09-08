from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import SatellaAssigner


def decode_ltrb_map(box_raw: torch.Tensor, stride: int):
    """Decode one raw [B,4,H,W] LTRB map to xyxy pixel boxes [B,H,W,4]."""
    b, _, h, w = box_raw.shape
    dtype, device = box_raw.dtype, box_raw.device
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    gx = (xx + 0.5) * stride
    gy = (yy + 0.5) * stride

    d = F.softplus(box_raw) * stride
    left, top, right, bottom = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    x1 = gx.unsqueeze(0) - left
    y1 = gy.unsqueeze(0) - top
    x2 = gx.unsqueeze(0) + right
    y2 = gy.unsqueeze(0) + bottom
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target.unbind(dim=-1)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps
    v = (4.0 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def sigmoid_focal_bce(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * modulating * bce


class SatellaDetectionLoss(nn.Module):
    def __init__(self, num_classes: int, strides=(4, 8, 16), box_weight: float = 5.0, cls_weight: float = 1.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0, image_size: int = 640):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(strides)
        self.box_weight = float(box_weight)
        self.cls_weight = float(cls_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.image_size = int(image_size)
        self.assigner = SatellaAssigner(strides=self.strides, image_size=self.image_size)

    def forward(self, outputs, targets):
        device = outputs[0]["cls"].device
        assigned, assign_stats = self.assigner.build_targets(outputs, targets, self.num_classes, device)
        num_pos = sum(int(x["pos"].sum().item()) for x in assigned)
        normalizer = max(num_pos, 1)
        cls_sum = outputs[0]["cls"].new_zeros(())
        box_sum = outputs[0]["cls"].new_zeros(())
        for out, target_level, stride in zip(outputs, assigned, self.strides):
            cls_loss = sigmoid_focal_bce(out["cls"], target_level["cls"], alpha=self.focal_alpha, gamma=self.focal_gamma)
            cls_sum = cls_sum + cls_loss.sum()
            pos = target_level["pos"]
            if pos.any():
                pred_map = decode_ltrb_map(out["box"], stride)
                pred_boxes = pred_map[pos]
                gt_boxes = target_level["box"].permute(0, 2, 3, 1)[pos]
                ciou = bbox_ciou(pred_boxes, gt_boxes)
                box_sum = box_sum + (1.0 - ciou).sum()
        cls_loss = cls_sum / normalizer
        box_loss = box_sum / normalizer
        total = self.box_weight * box_loss + self.cls_weight * cls_loss
        stats = {"loss": float(total.detach()), "box_loss": float(box_loss.detach()), "cls_loss": float(cls_loss.detach()), "num_pos": num_pos, **assign_stats}
        return total, stats
