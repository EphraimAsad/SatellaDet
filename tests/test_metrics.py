import torch
from satelladet.metrics import compute_detection_metrics


def test_absent_classes_do_not_dilute_map():
    records = [{
        "pred_boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "pred_scores": torch.tensor([0.99]),
        "pred_classes": torch.tensor([0]),
        "gt_boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "gt_classes": torch.tensor([0]),
    }]
    m = compute_detection_metrics(records, 3)
    assert m["map50"] > 0.99
    assert m["per_class"][1]["precision"] is None
    assert m["per_class"][2]["precision"] is None
