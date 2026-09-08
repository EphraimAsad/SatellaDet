import torch
from satelladet import build_model, count_parameters

model = build_model("n", num_classes=2).eval()
print(f"Trainable parameters: {count_parameters(model):,}")

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    outputs = model(x)

for level, result in zip(("P2", "P3", "P4"), outputs):
    print(level, result["box"].shape, result["cls"].shape)
