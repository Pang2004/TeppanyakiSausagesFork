"""Official IoU-weighted Door scoring implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ScoredDoorSegment:
    start_time: datetime
    end_time: datetime
    label: str


def segment_iou(left: ScoredDoorSegment, right: ScoredDoorSegment) -> float:
    """Calculate temporal intersection-over-union for two segments."""

    intersection = max(
        0.0,
        (
            min(left.end_time, right.end_time) - max(left.start_time, right.start_time)
        ).total_seconds(),
    )
    left_duration = (left.end_time - left.start_time).total_seconds()
    right_duration = (right.end_time - right.start_time).total_seconds()
    union = left_duration + right_duration - intersection
    return intersection / union if union > 0 else 0.0


def official_door_score(
    truth: list[ScoredDoorSegment], predictions: list[ScoredDoorSegment]
) -> float:
    """Apply greedy same-label matching and return the official soft F1 score."""

    candidates: list[tuple[float, int, int]] = []
    for truth_index, expected in enumerate(truth):
        for prediction_index, predicted in enumerate(predictions):
            if expected.label != predicted.label:
                continue
            overlap = segment_iou(expected, predicted)
            if overlap > 0:
                candidates.append((overlap, truth_index, prediction_index))
    candidates.sort(reverse=True)

    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    overlap_sum = 0.0
    for overlap, truth_index, prediction_index in candidates:
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        overlap_sum += overlap

    recall = overlap_sum / len(truth) if truth else 0.0
    precision = overlap_sum / len(predictions) if predictions else 0.0
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0
