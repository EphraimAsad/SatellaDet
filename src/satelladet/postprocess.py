from __future__ import annotations

import torch
import torch.nn.functional as F


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7):
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def decode_outputs(outputs, strides=(4, 8, 16), image_size: int = 640):
    all_boxes, all_scores = [], []
    for out, stride in zip(outputs, strides):
        box_raw, cls_raw = out["box"], out["cls"]
        b, _, h, w = box_raw.shape
        dtype, device = box_raw.dtype, box_raw.device
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=dtype, device=device),
            torch.arange(w, dtype=dtype, device=device),
            indexing="ij",
        )
        gx = ((xx + 0.5) * stride).unsqueeze(0)
        gy = ((yy + 0.5) * stride).unsqueeze(0)
        d = F.softplus(box_raw) * stride
        x1 = (gx - d[:, 0]).clamp(0, image_size)
        y1 = (gy - d[:, 1]).clamp(0, image_size)
        x2 = (gx + d[:, 2]).clamp(0, image_size)
        y2 = (gy + d[:, 3]).clamp(0, image_size)
        boxes = torch.stack((x1, y1, x2, y2), dim=-1).reshape(b, -1, 4)
        scores = torch.sigmoid(cls_raw).permute(0, 2, 3, 1).reshape(b, -1, cls_raw.shape[1])
        all_boxes.append(boxes)
        all_scores.append(scores)
    return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)


def nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    if keep:
        return torch.stack(keep)
    return torch.empty((0,), dtype=torch.long, device=boxes.device)


def postprocess_batch(
    outputs,
    strides=(4, 8, 16),
    image_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    pre_nms_topk: int = 5000,
    max_det: int = 3000,
):
    boxes_batch, scores_batch = decode_outputs(outputs, strides, image_size)
    results = []
    for boxes, class_scores in zip(boxes_batch, scores_batch):
        best_scores, class_ids = class_scores.max(dim=1)
        mask = best_scores >= conf_threshold
        boxes = boxes[mask]
        best_scores = best_scores[mask]
        class_ids = class_ids[mask]

        if boxes.numel() == 0:
            results.append({
                "boxes": boxes.reshape(0, 4),
                "scores": best_scores,
                "classes": class_ids,
            })
            continue

        if len(best_scores) > pre_nms_topk:
            top = torch.topk(best_scores, pre_nms_topk).indices
            boxes, best_scores, class_ids = boxes[top], best_scores[top], class_ids[top]

        keep_parts = []
        for cls in class_ids.unique():
            idx = torch.nonzero(class_ids == cls, as_tuple=False).squeeze(1)
            kept_local = nms_single_class(boxes[idx], best_scores[idx], iou_threshold)
            keep_parts.append(idx[kept_local])

        keep = torch.cat(keep_parts) if keep_parts else torch.empty((0,), dtype=torch.long, device=boxes.device)
        keep = keep[best_scores[keep].argsort(descending=True)]
        if max_det > 0:
            keep = keep[:max_det]
        results.append({
            "boxes": boxes[keep],
            "scores": best_scores[keep],
            "classes": class_ids[keep],
        })
    return results
