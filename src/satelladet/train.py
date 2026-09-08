from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import YoloDetectionDataset, detection_collate, load_dataset_yaml
from .ema import ModelEMA
from .exporter import export_onnx_from_model
from .losses import SatellaDetectionLoss
from .metrics import compute_detection_metrics
from .model import SatellaDetN, count_parameters
from .postprocess import postprocess_batch


def parse_args():
    p = argparse.ArgumentParser(description="Train SatellaDet-N on YOLO-format labels.")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--weights", default=None, help="Checkpoint used to initialise fine-tuning")
    p.add_argument("--resume", default=None, help="Checkpoint used to resume a training run")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr-factor", type=float, default=0.01)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--score-f1-weight", type=float, default=0.35)
    p.add_argument("--score-map50-95-weight", type=float, default=0.30)
    p.add_argument("--score-map50-weight", type=float, default=0.20)
    p.add_argument("--score-count-weight", type=float, default=0.15)
    return p.parse_args()


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _device(spec: str):
    return torch.device("cuda" if spec == "auto" and torch.cuda.is_available() else "cpu" if spec == "auto" else spec)


def _lr(epoch, epochs, base, warmup, floor):
    if warmup and epoch < warmup:
        return base * (epoch + 1) / warmup
    p = (epoch - warmup) / max(epochs - warmup - 1, 1)
    c = 0.5 * (1 + math.cos(math.pi * min(max(p, 0.0), 1.0)))
    return base * (floor + (1 - floor) * c)


def _safe_args(args):
    out = vars(args).copy()
    for key in ("data", "output", "weights", "resume", "run_dir"):
        if out.get(key): out[key] = Path(str(out[key])).name
    return out


def _state(ckpt):
    if not isinstance(ckpt, dict): return ckpt
    if ckpt.get("ema") is not None and ckpt.get("ema_updates") is not None: return ckpt["ema"]
    return ckpt.get("model", ckpt)


def _satella_score(metrics, args):
    weights = {
        "f1": args.score_f1_weight,
        "map50_95": args.score_map50_95_weight,
        "map50": args.score_map50_weight,
        "count": args.score_count_weight,
    }
    if any(v < 0 for v in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("SatellaScore component weights must be non-negative and sum to 1.0")
    parts = {
        "weighted_f1": float(metrics["weighted_f1"]),
        "weighted_map50_95": float(metrics["weighted_map50_95"]),
        "weighted_map50": float(metrics["weighted_map50"]),
        "count_score": float(metrics["count_score"]),
    }
    score = (
        weights["f1"] * parts["weighted_f1"]
        + weights["map50_95"] * parts["weighted_map50_95"]
        + weights["map50"] * parts["weighted_map50"]
        + weights["count"] * parts["count_score"]
    )
    return float(score), weights, parts


@torch.no_grad()
def _validate(model, loader, device, nc, args):
    model.eval(); records = []
    for images, targets in loader:
        detections = postprocess_batch(
            model(images.to(device, non_blocking=True)),
            image_size=args.imgsz,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        for det, target in zip(detections, targets):
            records.append({
                "pred_boxes": det["boxes"].detach().cpu(),
                "pred_scores": det["scores"].detach().cpu(),
                "pred_classes": det["classes"].detach().cpu(),
                "gt_boxes": target["boxes"].detach().cpu(),
                "gt_classes": target["classes"].detach().cpu(),
            })
    return compute_detection_metrics(records, nc, class_weight_power=args.class_weight_power)


def _checkpoint(path, model, ema, optimizer, epoch, best_score, args, names, metrics):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
        "metrics": metrics,
        "args": _safe_args(args),
        "names": names,
    }, path)


def _write_tracker(path, tracker):
    clean = {k: {"value": float(v["value"]), "epoch": int(v["epoch"]), "file": v["file"]} for k, v in tracker.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _print_classes(per_class, names):
    print("Class                 GT   Pred      P      R     F1   mAP50  mAP50-95", flush=True)
    for item in per_class:
        name = names[int(item["class_id"])]
        if item["gt"] == 0:
            print(f"{name:<20} {item['gt']:>4} {item['pred']:>6}      —      —      —       —         —", flush=True)
        else:
            print(
                f"{name:<20} {item['gt']:>4} {item['pred']:>6} "
                f"{item['precision']:>6.3f} {item['recall']:>6.3f} {item['f1']:>6.3f} "
                f"{item['map50']:>7.3f} {item['map50_95']:>9.3f}", flush=True,
            )


def main():
    args = parse_args()
    if args.imgsz < 128 or args.imgsz % 32:
        raise ValueError("--imgsz must be >=128 and divisible by 32")
    if args.weights and args.resume:
        raise ValueError("Use only one of --weights or --resume")
    _seed(args.seed)
    device = _device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    data = load_dataset_yaml(args.data)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / Path(args.output).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output).expanduser().resolve()

    train_ds = YoloDetectionDataset(data.train, data.num_classes, args.imgsz, augment=True)
    val_ds = YoloDetectionDataset(data.val, data.num_classes, args.imgsz, augment=False)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, collate_fn=detection_collate)

    model = SatellaDetN(data.num_classes).to(device)
    criterion = SatellaDetectionLoss(data.num_classes, image_size=args.imgsz)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model)
    start_epoch, best_score, stale = 0, -float("inf"), 0

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(_state(ckpt), strict=True); ema = ModelEMA(model)
    elif args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_imgsz = ckpt.get("args", {}).get("imgsz")
        if saved_imgsz is not None and int(saved_imgsz) != args.imgsz:
            raise ValueError("Cannot change --imgsz when using --resume; use --weights instead")
        model.load_state_dict(ckpt["model"]); ema.ema.load_state_dict(ckpt.get("ema", ckpt["model"]))
        ema.updates = int(ckpt.get("ema_updates", 0)); optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1; best_score = float(ckpt.get("best_score", -float("inf")))

    tracker = {
        "score": {"value": best_score, "epoch": 0, "file": "best.pt"},
        "map50": {"value": -float("inf"), "epoch": 0, "file": "best_map50.pt"},
        "map50_95": {"value": -float("inf"), "epoch": 0, "file": "best_map50_95.pt"},
        "count_mae": {"value": float("inf"), "epoch": 0, "file": "best_count.pt"},
    }
    metrics_csv = run_dir / "metrics.csv"
    if start_epoch == 0:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch","lr","loss","box_loss","cls_loss","precision","recall","map50","map50_95","count_mae","satella_score","seconds"])

    print(f"SatellaDet-N v0.2.1 | device={device} | imgsz={args.imgsz} | classes={data.num_classes} | params={count_parameters(model):,}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time(); model.train(); sums = {"loss":0.0,"box_loss":0.0,"cls_loss":0.0}; batches = 0
        lr = _lr(epoch, args.epochs, args.lr, args.warmup_epochs, args.min_lr_factor)
        for group in optimizer.param_groups: group["lr"] = lr
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, stats = criterion(model(images), targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update(); ema.update(model)
            for key in sums: sums[key] += stats[key]
            batches += 1
        for key in sums: sums[key] /= max(batches, 1)

        metrics = _validate(ema.ema.to(device), val_loader, device, data.num_classes, args)
        score, score_weights, score_parts = _satella_score(metrics, args); metrics["satella_score"] = score
        improved = score > best_score + 1e-6
        if improved: best_score, stale = score, 0
        else: stale += 1

        _checkpoint(run_dir / "last.pt", model, ema, optimizer, epoch, best_score, args, data.names, metrics)
        if improved:
            _checkpoint(run_dir / "best.pt", model, ema, optimizer, epoch, best_score, args, data.names, metrics)
            tracker["score"] = {"value": score, "epoch": epoch + 1, "file": "best.pt"}
        auxiliary = [
            ("map50", metrics["weighted_map50"], True, "best_map50.pt"),
            ("map50_95", metrics["weighted_map50_95"], True, "best_map50_95.pt"),
            ("count_mae", metrics["count_mae"], False, "best_count.pt"),
        ]
        for key, value, higher, filename in auxiliary:
            old = tracker[key]["value"]
            better = value > old + 1e-6 if higher else value < old - 1e-6
            if better:
                tracker[key] = {"value": float(value), "epoch": epoch + 1, "file": filename}
                _checkpoint(run_dir / filename, model, ema, optimizer, epoch, best_score, args, data.names, metrics)
        _write_tracker(run_dir / "checkpoint_tracker.json", tracker)

        seconds = time.time() - t0
        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch+1,lr,sums["loss"],sums["box_loss"],sums["cls_loss"],metrics["precision"],metrics["recall"],metrics["map50"],metrics["map50_95"],metrics["count_mae"],score,seconds])
        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss {sums['loss']:.4f} | P {metrics['precision']:.3f} R {metrics['recall']:.3f} | mAP50 {metrics['map50']:.3f} mAP50-95 {metrics['map50_95']:.3f} | count MAE {metrics['count_mae']:.2f} | Score {score:.4f} | {seconds:.1f}s", flush=True)
        if improved:
            _print_classes(metrics["per_class"], data.names)
            print(f"SatellaScore parts={score_parts} weights={score_weights}", flush=True)
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping after {stale} epochs without SatellaScore improvement.", flush=True); break

    best_path = run_dir / "best.pt"
    if not best_path.exists(): best_path = run_dir / "last.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    export_model = SatellaDetN(data.num_classes); export_model.load_state_dict(_state(best), strict=True)
    export_onnx_from_model(export_model, output, image_size=args.imgsz)
    print(f"Best checkpoint: {best_path.resolve()}", flush=True)
    print(f"ONNX: {output}", flush=True)


if __name__ == "__main__":
    main()
