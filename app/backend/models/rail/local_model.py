"""Portable local-spectrum classifier with training-only side augmentation."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mirror_mapping(names):
    """Swap side summaries and negate antisymmetric contrasts."""
    lookup = {name: i for i, name in enumerate(names)}
    indices, signs = [], []
    for name in names:
        target, sign = name, 1
        for left, right in (
            ("side_i_", "side_ii_"),
            ("vibration_i_", "vibration_ii_"),
            ("shock_i_", "shock_ii_"),
        ):
            if name.startswith(left):
                target = right + name[len(left) :]
                break
            if name.startswith(right):
                target = left + name[len(right) :]
                break
        if name.startswith("side_comparison_") or "_contrast_" in name:
            sign = -1
        indices.append(lookup[target])
        signs.append(sign)
    return np.array(indices), np.array(signs)


class LocalRailClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, names, columns, c=0.1, mirror=False, boost=1.0):
        self.names = names
        self.columns = columns
        self.c = c
        self.mirror = mirror
        self.boost = boost

    def fit(self, x, y):
        x = x[:, self.columns]
        if self.mirror:
            indices, signs = mirror_mapping(self.names)
            x = np.concatenate((x, x[:, indices] * signs))
            swapped = np.array(
                [
                    {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}[v]
                    for v in y
                ]
            )
            y = np.concatenate((y, swapped))
        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=5000, random_state=42
            ),
        ).fit(x, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(x[:, self.columns])

    def predict(self, x):
        scores = self.predict_proba(x)
        scores[:, list(self.classes_).index("Side I")] *= self.boost
        return self.classes_[scores.argmax(axis=1)]
