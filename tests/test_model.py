import torch
from satelladet import build_model, SatellaDetONNX


def test_deployment_shape_128():
    model = build_model("n", num_classes=3).eval()
    deploy = SatellaDetONNX(model, image_size=128).eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        y = deploy(x)
    expected_n = (128 // 4) ** 2 + (128 // 8) ** 2 + (128 // 16) ** 2
    assert y.shape == (1, 7, expected_n)
    assert torch.isfinite(y).all()
    assert y[:, 4:, :].min() >= 0
    assert y[:, 4:, :].max() <= 1
