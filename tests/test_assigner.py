import torch
from satelladet.assigner import SatellaAssigner


def test_assignment_scales_with_resolution():
    a640 = SatellaAssigner(image_size=640)
    a1280 = SatellaAssigner(image_size=1280)
    for size in (12, 16, 28, 40, 96):
        b640 = torch.tensor([0.0, 0.0, float(size), float(size)])
        b1280 = torch.tensor([0.0, 0.0, float(size * 2), float(size * 2)])
        assert a640.choose_level(b640) == a1280.choose_level(b1280)
