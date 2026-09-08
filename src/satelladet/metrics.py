from __future__ import annotations

import numpy as np
import torch

from .postprocess import box_iou_one_to_many


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def compute_detection_metrics(records, num_classes: int, iou_thresholds=None, class_weight_power: float = 0.5, explicit_class_weights=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)
    aps = np.zeros((num_classes, len(iou_thresholds)), dtype=np.float64)
    precision50 = []
    recall50 = []
    valid_classes = []
    per_class = []
    for cls in range(num_classes):
        gt_total = sum(int((r["gt_classes"] == cls).sum()) for r in records)
        pred_total = sum(int((r["pred_classes"] == cls).sum()) for r in records)
        if gt_total == 0:
            per_class.append({"class_id": cls, "gt": 0, "pred": pred_total, "precision": None, "recall": None, "map50": None, "map50_95": None, "f1": None, "score_weight": 0.0})
            continue
        valid_classes.append(cls)
        preds = []
        for image_idx, r in enumerate(records):
            mask = r["pred_classes"] == cls
            for box, score in zip(r["pred_boxes"][mask], r["pred_scores"][mask]):
                preds.append((float(score), image_idx, box))
        preds.sort(key=lambda x: x[0], reverse=True)
        class_precision50 = 0.0
        class_recall50 = 0.0
        for ti, threshold in enumerate(iou_thresholds):
            matched = {i: torch.zeros(len(r["gt_boxes"]), dtype=torch.bool) for i, r in enumerate(records)}
            tp, fp = [], []
            for _, image_idx, pred_box in preds:
                r = records[image_idx]
                gt_mask = r["gt_classes"] == cls
                gt_indices = torch.nonzero(gt_mask, as_tuple=False).squeeze(1)
                if gt_indices.numel() == 0:
                    tp.append(0.0); fp.append(1.0); continue
                candidate_boxes = r["gt_boxes"][gt_indices]
                ious = box_iou_one_to_many(pred_box, candidate_boxes)
                best_iou, local = ious.max(dim=0)
                global_gt = int(gt_indices[int(local)])
                if float(best_iou) >= float(threshold) and not bool(matched[image_idx][global_gt]):
                    matched[image_idx][global_gt] = True
                    tp.append(1.0); fp.append(0.0)
                else:
                    tp.append(0.0); fp.append(1.0)
            if len(tp) == 0:
                aps[cls, ti] = 0.0
                if ti == 0:
                    class_precision50 = 0.0; class_recall50 = 0.0
                continue
            tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
            recall = tp_c / max(gt_total, 1)
            precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
            aps[cls, ti] = _ap_from_pr(recall, precision)
            if ti == 0:
                class_precision50 = float(precision[-1]); class_recall50 = float(recall[-1])
        precision50.append(class_precision50); recall50.append(class_recall50)
        per_class.append({"class_id": cls, "gt": gt_total, "pred": pred_total, "precision": class_precision50, "recall": class_recall50, "map50": float(aps[cls, 0]), "map50_95": float(aps[cls].mean()), "f1": 2.0 * class_precision50 * class_recall50 / max(class_precision50 + class_recall50, 1e-12), "score_weight": 0.0})
    count_errors = []; signed_errors = []
    for r in records:
        pred_n = len(r["pred_boxes"]); gt_n = len(r["gt_boxes"])
        signed = pred_n - gt_n
        signed_errors.append(signed); count_errors.append(abs(signed))
    count_errors_np = np.asarray(count_errors, dtype=float) if count_errors else np.zeros(1)
    signed_np = np.asarray(signed_errors, dtype=float) if signed_errors else np.zeros(1)
    valid_aps = aps[valid_classes] if valid_classes else np.zeros((0, len(iou_thresholds)), dtype=np.float64)
    if not 0.0 <= class_weight_power <= 1.0:
        raise ValueError("class_weight_power must be between 0.0 and 1.0")
    raw_weights = []; evaluable_items = []
    for item in per_class:
        if int(item["gt"]) <= 0: continue
        cls_id = int(item["class_id"]); evaluable_items.append(item)
        raw = float(explicit_class_weights.get(cls_id, 0.0)) if explicit_class_weights is not None else float(item["gt"]) ** float(class_weight_power)
        if not np.isfinite(raw) or raw < 0: raise ValueError(f"Invalid class weight for class {cls_id}: {raw}")
        raw_weights.append(raw)
    weight_total = sum(raw_weights)
    if evaluable_items and weight_total <= 0: raise ValueError("Evaluable validation classes have zero total SatellaScore class weight")
    if weight_total > 0:
        for item, raw in zip(evaluable_items, raw_weights): item["score_weight"] = raw / weight_total
    def weighted(field: str) -> float:
        return float(sum(float(item["score_weight"]) * float(item[field]) for item in evaluable_items)) if evaluable_items else 0.0
    total_gt = sum(len(r["gt_boxes"]) for r in records)
    mean_gt_count = float(total_gt / max(len(records), 1))
    count_mae = float(count_errors_np.mean())
    count_score = max(0.0, 1.0 - count_mae / max(mean_gt_count, 1.0))
    return {"precision": float(np.mean(precision50)) if precision50 else 0.0, "recall": float(np.mean(recall50)) if recall50 else 0.0, "map50": float(valid_aps[:, 0].mean()) if valid_aps.size else 0.0, "map50_95": float(valid_aps.mean()) if valid_aps.size else 0.0, "weighted_precision": weighted("precision"), "weighted_recall": weighted("recall"), "weighted_f1": weighted("f1"), "weighted_map50": weighted("map50"), "weighted_map50_95": weighted("map50_95"), "count_mae": count_mae, "count_median_ae": float(np.median(count_errors_np)), "count_bias": float(signed_np.mean()), "mean_gt_count": mean_gt_count, "count_score": count_score, "within_1": float(np.mean(count_errors_np <= 1)), "within_2": float(np.mean(count_errors_np <= 2)), "within_5": float(np.mean(count_errors_np <= 5)), "per_class": per_class}
