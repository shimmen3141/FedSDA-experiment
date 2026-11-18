"""警報の時点のMDLによる選択: 3つの選択、保有モデルを変えないこと、逆順の前置きの和、新しいモデルの作り直しを確かめる（WSL）。"""

import math

import pytest

pytest.importorskip("river")

from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (  # noqa: E402
    REINITIALIZED_LEARNER_NAME,
    WARM_STARTED_LEARNER_NAME,
    compute_switching_code_length,
)
from single_stream_hard_assignment_soft_prediction.alarm_interval_assignment import (  # noqa: E402
    build_selected_new_concept_model,
    select_alarm_interval_assignment,
)
from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (  # noqa: E402
    HoeffdingTreeConceptModel,
)

SWITCHING_PROBABILITY = 1.0 / 120
PROBE = tuple({0: float(a), 1: float(b)} for a in range(7) for b in range(5))


def concept_label(concept, x):
    if concept == "A":
        return int(x[0] >= 3.0)
    if concept == "B":
        return int(x[0] < 3.0)
    return int(x[1] >= 2.0)


def samples(concept, count, offset=0):
    out = []
    for i in range(offset, offset + count):
        x = {0: float(i % 7), 1: float((i * 3) % 5)}
        out.append((x, concept_label(concept, x)))
    return out


def trained(concept, count=1500):
    model = HoeffdingTreeConceptModel.create_empty(class_count=2, probability_smoothing_weight=0.01)
    for x, y in samples(concept, count):
        model.learn_one(features=x, observed_class_id=y)
    return model


def snapshot(model):
    return (model.node_count, [model.predict_class_probabilities(features=x) for x in PROBE])


def select(window, held, current_model_id, with_statistics=None):
    return select_alarm_interval_assignment(
        pending_samples=tuple(window),
        held_models=tuple(held),
        current_model_id=current_model_id,
        model_ids_with_loss_statistics=frozenset(
            with_statistics if with_statistics is not None else [m for m, _ in held]
        ),
        switching_probability=SWITCHING_PROBABILITY,
        minimum_change_interval_sample_count=5,
    )


def test_matching_current_model_keeps_it_with_k_equal_n():
    held = [(0, trained("A")), (1, trained("B"))]
    window = samples("A", 31, offset=3)
    selection = select(window, held, 0)
    assert selection.selected_hypothesis == "current_model"
    assert selection.change_interval_sample_count == 31


def test_matching_held_model_is_reused_after_the_change_point():
    held = [(0, trained("A")), (1, trained("B"))]
    window = samples("A", 10, offset=1) + samples("B", 21, offset=11)
    selection = select(window, held, 0)
    assert selection.selected_hypothesis == "held_model"
    assert selection.destination_model_id == 1
    assert 15 <= selection.change_interval_sample_count <= 25


def test_unseen_concept_creates_a_new_model():
    held = [(0, trained("A"))]
    window = samples("A", 6, offset=2) + samples("C", 25, offset=40)
    selection = select(window, held, 0)
    assert selection.selected_hypothesis == "new_model"
    assert selection.new_model_learner_name in (REINITIALIZED_LEARNER_NAME, WARM_STARTED_LEARNER_NAME)


def test_held_models_without_loss_statistics_are_not_destinations():
    held = [(0, trained("A")), (1, trained("B"))]
    window = samples("A", 10, offset=1) + samples("B", 21, offset=11)
    selection = select(window, held, 0, with_statistics=[0])
    assert selection.selected_hypothesis != "held_model"


def test_held_models_are_not_changed():
    held = [(0, trained("A")), (1, trained("B"))]
    before = [snapshot(model) for _, model in held]
    select(samples("A", 6, offset=2) + samples("C", 25, offset=40), held, 0)
    assert [snapshot(model) for _, model in held] == before


def reverse_prefixes(learner, window):
    prefixes, total = [], 0.0
    for x, y in reversed(window):
        total += learner.description_length_loss(features=x, observed_class_id=y)
        prefixes.append(total)
        learner.learn_one(features=x, observed_class_id=y)
    return prefixes


def test_new_model_code_length_uses_reverse_prefix_sums_of_reset_and_keep_learners():
    current = trained("A")
    held = [(0, current)]
    window = samples("A", 6, offset=2) + samples("C", 25, offset=40)
    selection = select(window, held, 0)
    assert selection.selected_hypothesis == "new_model"
    n, k = len(window), selection.change_interval_sample_count
    reset = reverse_prefixes(current.create_empty_like(), window)
    keep = reverse_prefixes(current.copy(), window)
    before = math.fsum(current.description_length_loss(features=x, observed_class_id=y) for x, y in window[: n - k])
    switching = compute_switching_code_length(switching_probability=SWITCHING_PROBABILITY, destination_count=1)
    expected = (before + switching) + (min(reset[k - 1], keep[k - 1]) + math.log(2.0))
    assert selection.description_length == pytest.approx(expected, rel=1e-12)
    expected_name = REINITIALIZED_LEARNER_NAME if reset[k - 1] <= keep[k - 1] else WARM_STARTED_LEARNER_NAME
    assert selection.new_model_learner_name == expected_name


@pytest.mark.parametrize("learner_name", [REINITIALIZED_LEARNER_NAME, WARM_STARTED_LEARNER_NAME])
def test_built_new_model_equals_the_measured_learner_after_k_samples(learner_name):
    current = trained("A")
    window = samples("A", 6, offset=2) + samples("C", 25, offset=40)
    k = 20
    built, recorded_losses = build_selected_new_concept_model(
        learner_name=learner_name, current_model=current, pending_samples=tuple(window), change_interval_sample_count=k
    )
    learner = current.create_empty_like() if learner_name == REINITIALIZED_LEARNER_NAME else current.copy()
    expected_losses = []
    for x, y in list(reversed(window))[:k]:
        expected_losses.append(learner.monitoring_loss(features=x, observed_class_id=y))
        learner.learn_one(features=x, observed_class_id=y)
    assert snapshot(built) == snapshot(learner)
    assert recorded_losses == pytest.approx(expected_losses)
    # 現在のモデルは変わらない。
    assert snapshot(current) == snapshot(trained("A"))


def test_short_window_has_no_new_model_hypothesis():
    held = [(0, trained("A"))]
    selection = select(samples("C", 4), held, 0)
    assert selection.selected_hypothesis == "current_model"
    assert selection.change_interval_sample_count == 4
