from __future__ import annotations

from pathlib import Path
import torch

from .model import SatellaDetN, SatellaDetONNX


def export_onnx_from_model(model: SatellaDetN, output: str | Path, image_size: int = 640):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.eval().cpu()
    deploy = SatellaDetONNX(cpu_model, image_size=image_size).eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        deploy,
        (dummy,),
        output,
        input_names=["images"],
        output_names=["output0"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return output
