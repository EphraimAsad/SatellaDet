"""SatellaDet: a lightweight anchor-free object detection framework."""

from .model import SatellaDetN, SatellaDetONNX, build_model, count_parameters

__version__ = "0.2.1"

__all__ = [
    "SatellaDetN",
    "SatellaDetONNX",
    "build_model",
    "count_parameters",
    "__version__",
]
