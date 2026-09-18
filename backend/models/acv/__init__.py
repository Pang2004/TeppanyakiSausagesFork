"""ACV refrigerant-leak localisation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import ACVPrediction

if TYPE_CHECKING:
    from .predict import ACVPredictor

__all__ = ["ACVPrediction", "ACVPredictor", "predict_file"]


def __getattr__(name: str) -> Any:
    """Lazily expose inference helpers without preloading CLI modules."""

    if name in {"ACVPredictor", "predict_file"}:
        from . import predict

        return getattr(predict, name)
    raise AttributeError(name)
