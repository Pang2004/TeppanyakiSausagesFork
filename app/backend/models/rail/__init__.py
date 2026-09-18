"""Rail corrugation feature extraction, training, and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import RailPrediction

if TYPE_CHECKING:
    from .predict import RailPredictor

__all__ = ["RailPrediction", "RailPredictor", "predict_file"]


def __getattr__(name: str) -> Any:
    """Lazily expose inference helpers without preloading CLI modules."""

    if name in {"RailPredictor", "predict_file"}:
        from . import predict

        return getattr(predict, name)
    raise AttributeError(name)
