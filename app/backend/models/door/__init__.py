"""Door stream segmentation, fault classification, and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import DoorPrediction

if TYPE_CHECKING:
    from .predict import DoorPredictor

__all__ = ["DoorPrediction", "DoorPredictor", "predict_stream"]


def __getattr__(name: str) -> Any:
    """Lazily expose inference helpers without preloading CLI modules."""

    if name in {"DoorPredictor", "predict_stream"}:
        from . import predict

        return getattr(predict, name)
    raise AttributeError(name)
