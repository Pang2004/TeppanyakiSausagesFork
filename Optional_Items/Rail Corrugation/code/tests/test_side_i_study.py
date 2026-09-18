"""Checks for research decision tuning and hierarchical score composition."""

import numpy as np
from rail_dev.side_i_study import StudyClassifier, boosted_prediction, choose


def test_side_i_selection_rejects_large_macro_loss():
    scores = [
        {"candidate": "original_C1.0", "boost": 1.0, "macro": 0.72, "side_i": 0.44},
        {"candidate": "mirror_C1.0", "boost": 2.0, "macro": 0.71, "side_i": 0.58},
        {"candidate": "mirror_C1.0", "boost": 4.0, "macro": 0.60, "side_i": 0.70},
    ]
    assert choose(scores)["select_side_i_guarded"] == scores[1]
    probabilities = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    np.testing.assert_array_equal(
        boosted_prediction(probabilities, 2), ["Side I", "Side II"]
    )
    np.testing.assert_array_equal(probabilities, [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])


def test_hierarchy_scores_follow_class_order_and_sum_to_one():
    x = np.array([[0, 0], [0.1, 0], [4, 0], [5, 0], [0, 4], [0, 5]])
    y = np.array(["Normal", "Normal", "Side I", "Side I", "Side II", "Side II"])
    model = StudyClassifier(names=("a", "b"), c=10, kind="hierarchical").fit(x, y)
    probabilities = model.predict_proba(x)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict(x), y)
