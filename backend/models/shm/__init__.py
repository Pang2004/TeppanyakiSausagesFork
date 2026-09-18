"""Structural Health Monitoring fatigue-damage prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import SHMPrediction

if TYPE_CHECKING:
    from .predict import SHMPredictor

__all__ = ["SHMPrediction", "SHMPredictor", "predict_file"]


def __getattr__(name: str) -> Any:
    """Lazily expose inference helpers without preloading CLI modules."""

    if name in {"SHMPredictor", "predict_file"}:
        from . import predict

        return getattr(predict, name)
    raise AttributeError(name)
