"""概念モデル（Hoeffding tree）: 平滑化した確率、損失、写し、空の新しいモデル、特徴の変換を確かめる（WSLのRiver環境）。"""

import math

import numpy as np
import pytest

pytest.importorskip("river")

from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (  # noqa: E402
    HoeffdingTreeConceptModel,
    convert_features_to_dictionary,
)

SMOOTHING = 0.01


def train(model, count=400, flip=False):
    for i in range(count):
        x = {0: float(i % 7), 1: float(i % 5)}
        label = int(x[0] >= 3.0)
        model.learn_one(features=x, observed_class_id=1 - label if flip else label)


def test_empty_model_predicts_the_uniform_distribution():
    model = HoeffdingTreeConceptModel.create_empty(class_count=4, probability_smoothing_weight=SMOOTHING)
    probabilities = model.predict_class_probabilities(features={0: 1.0})
    assert probabilities == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert model.node_count == 0


def test_probabilities_are_smoothed_sum_to_one_and_keep_the_argmax():
    model = HoeffdingTreeConceptModel.create_empty(class_count=3, probability_smoothing_weight=SMOOTHING)
    train(model)
    raw = model.tree.predict_proba_one({0: 6.0, 1: 0.0})
    probabilities = model.predict_class_probabilities(features={0: 6.0, 1: 0.0})
    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.0)
    # 木が返さないクラス（2）は0から平滑化される。
    assert 2 not in raw or raw[2] == 0
    assert probabilities[2] == pytest.approx(SMOOTHING / 3)
    for class_id in range(2):
        expected = (1 - SMOOTHING) * raw.get(class_id, 0.0) + SMOOTHING / 3
        assert probabilities[class_id] == pytest.approx(expected)
    assert int(np.argmax(probabilities)) == max(raw, key=raw.get)


def test_losses_follow_the_smoothed_probability_of_the_observed_class():
    model = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=SMOOTHING)
    train(model)
    x = {0: 6.0, 1: 1.0}
    p = model.predict_class_probabilities(features=x)
    for observed in (0, 1):
        assert model.monitoring_loss(features=x, observed_class_id=observed) == pytest.approx(1 - p[observed])
        assert model.description_length_loss(features=x, observed_class_id=observed) == pytest.approx(
            -math.log(p[observed])
        )
        assert math.isfinite(model.description_length_loss(features=x, observed_class_id=observed))


def test_copy_learns_independently_of_the_original():
    original = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=SMOOTHING)
    train(original)
    before = original.predict_class_probabilities(features={0: 6.0, 1: 1.0})
    copied = original.copy()
    train(copied, count=2000, flip=True)
    assert original.predict_class_probabilities(features={0: 6.0, 1: 1.0}) == before
    assert copied.predict_class_probabilities(features={0: 6.0, 1: 1.0}) != before
    assert copied.class_count == original.class_count


def test_new_empty_model_has_the_same_class_count_and_no_knowledge():
    original = HoeffdingTreeConceptModel.create_empty(class_count=5, probability_smoothing_weight=SMOOTHING)
    train(original)
    fresh = original.create_empty_like()
    assert fresh.class_count == 5
    assert fresh.node_count == 0
    assert fresh.predict_class_probabilities(features={0: 6.0, 1: 1.0}) == pytest.approx([0.2] * 5)


def test_default_tree_hyperparameters_are_rivers_defaults():
    from river import tree

    model = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=SMOOTHING)
    default = tree.HoeffdingTreeClassifier()
    assert model.tree._get_params() == default._get_params()


@pytest.mark.parametrize("class_id", [-1, 2, True, 1.0])
def test_invalid_class_ids_are_rejected_before_learning(class_id):
    model = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=SMOOTHING)
    with pytest.raises((ValueError, TypeError)):
        model.learn_one(features={0: 1.0}, observed_class_id=class_id)
    assert model.node_count == 0


def test_feature_conversion_matches_the_river_baseline_tool():
    from tools.research_single_client_stream_baselines.river_baselines import _as_feature_dictionary

    row = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    assert convert_features_to_dictionary(row=row) == _as_feature_dictionary(row)
    assert all(type(v) is float for v in convert_features_to_dictionary(row=row).values())
