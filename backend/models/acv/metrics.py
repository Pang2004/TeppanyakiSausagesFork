"""Metrics for the ACV car-ranking task."""

from __future__ import annotations

from collections.abc import Sequence


def rank_decay_score(ranked_cars: Sequence[str], faulty_car: str) -> float:
    """Return the official linear rank-decay score for one case."""

    if faulty_car not in ranked_cars or not ranked_cars:
        return 0.0
    rank = ranked_cars.index(faulty_car) + 1
    return (len(ranked_cars) - (rank - 1)) / len(ranked_cars)
