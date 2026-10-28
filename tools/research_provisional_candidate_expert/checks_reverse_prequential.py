"""逆順の流れの再利用と、候補・位置の費用・計算量を検査する。"""

import random

import pytest
import torch

from federated_learning_experiments.learning.training import prequential_code_length as training
from tests.refactoring.test_alarm_interval_minimum_description_length_selection import (
    OPTIMIZER_SETTINGS,
    _classifier,
    _samples,
)
from tools.research_provisional_candidate_expert.reverse_prequential import (
    candidate_lengths,
    reverse_prequential_prefixes,
    select_reverse_assignment,
)


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("warm", [False, True])
def test_reverse_prefixes_match_independent_runs(classes, warm):
    model = _classifier(class_count=classes)
    x, y = _samples(count=31, class_count=classes)
    arguments = dict(
        template_classifier=model,
        input_features=x.flip(0),
        observed_class_labels=y.flip(0),
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
    )
    function = (
        training.compute_warm_started_learner_prequential_code_length
        if warm
        else training.compute_reinitialized_learner_prequential_code_length
    )
    if not warm:
        arguments["initialization_seed"] = 42
    before = torch.get_rng_state().clone(), random.getstate()
    prefixes = reverse_prequential_prefixes(compute=function, **arguments)
    for k in range(5, 32, 5):
        assert prefixes[k] == function(
            **{**arguments, "input_features": x.flip(0)[:k], "observed_class_labels": y.flip(0)[:k]}
        )
    assert torch.equal(before[0], torch.get_rng_state())
    assert before[1] == random.getstate()


def test_candidates_and_position_cost_and_ties():
    assert candidate_lengths(31, 5, False) == tuple(range(5, 31, 5))
    assert candidate_lengths(31, 5, True) == tuple(range(5, 32))
    arguments = dict(
        current_model_log_losses=(10.0,) * 6,
        held_model_log_losses_by_model_id=((2, (1.0,) * 6),),
        new_model_code_lengths_by_change_interval_sample_count={5: 20.0, 6: 20.0},
        new_model_learner_names_by_change_interval_sample_count={
            5: "reinitialized",
            6: "reinitialized",
        },
        switching_code_length=2.0,
    )
    # 名前は製品の宣言を使う。
    from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
        REINITIALIZED_LEARNER_NAME,
    )

    arguments["new_model_learner_names_by_change_interval_sample_count"] = dict.fromkeys(
        (5, 6), REINITIALIZED_LEARNER_NAME
    )
    import math

    for held, new, expected in [
        (1.0, 20.0, "held_model"),
        (9.0, 1.0, "new_model"),
        (10.0, 100.0, "current_model"),
    ]:
        arguments["held_model_log_losses_by_model_id"] = ((2, (held,) * 6),)
        arguments["new_model_code_lengths_by_change_interval_sample_count"] = dict.fromkeys(
            (5, 6), new
        )
        plain = select_reverse_assignment(**arguments, position_cost=False)
        cost = select_reverse_assignment(**arguments, position_cost=True)
        assert plain.selected_hypothesis == cost.selected_hypothesis == expected
        assert cost.description_length - plain.description_length == pytest.approx(
            0 if expected == "current_model" else math.log(2)
        )


def test_grid5_comparison_matches_product_including_ties():
    from federated_learning_experiments.methods.fedsda.training_data_assignment import (
        minimum_description_length_assignment_selection as product,
    )

    generator = random.Random(12)
    for case in range(50):
        # 整数の損失を使い、同値の仮説や同値のkも比較する。
        arguments = dict(
            current_model_log_losses=tuple(float(generator.randrange(4)) for _ in range(16)),
            held_model_log_losses_by_model_id=tuple(
                (model_id, tuple(float(generator.randrange(4)) for _ in range(16)))
                for model_id in (9, 2)
            ),
            new_model_code_lengths_by_change_interval_sample_count={
                k: float(generator.randrange(20)) for k in (5, 10, 15)
            },
            new_model_learner_names_by_change_interval_sample_count=dict.fromkeys(
                (5, 10, 15), product.REINITIALIZED_LEARNER_NAME
            ),
            switching_code_length=float(case % 3),
        )
        assert select_reverse_assignment(position_cost=False, **arguments) == (
            product.select_assignment_by_minimum_description_length(
                change_interval_grid_sample_count=5, **arguments
            )
        )


@pytest.mark.parametrize(
    "variant",
    [
        "product-default",
        "default-reverse-prequential-grid5",
        "default-reverse-prequential-grid1",
        "default-reverse-prequential-grid1-position-cost",
    ],
)
def test_update_counts_and_default_transparency(variant):
    from federated_learning_experiments.runtime import observed_sample_processing as processing
    from tests.refactoring.test_alarm_interval_minimum_description_length_selection import (
        CurrentTrainingModelAssignment,
        ModelAndClassLossStatisticsStore,
        _assert_random_states_equal,
        _observations,
        _random_states,
        _registry,
    )
    from tools.research_provisional_candidate_expert.reverse_prequential import (
        reverse_prequential_policy,
    )

    observations = _observations(count=31, class_count=4)
    arguments = dict(
        client_id=1,
        alarm_sample_index=observations[-1].sample_index,
        pending_sample_observations=observations,
        held_model_training_state_registry=_registry(class_count=4, model_ids=(4, 9)),
        loss_statistics_store=ModelAndClassLossStatisticsStore(),
        current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=9),
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        switching_probability=1 / 120,
        change_interval_grid_sample_count=5,
        new_model_hypothesis_learners="reinitialized_and_warm_started_learners",
    )
    before = _random_states()
    expected = processing.select_alarm_interval_assignment(**arguments)
    diagnostics = {}
    original = processing.select_alarm_interval_assignment
    with reverse_prequential_policy(variant=variant, diagnostics=diagnostics):
        actual = processing.select_alarm_interval_assignment(**arguments)
    assert processing.select_alarm_interval_assignment is original
    _assert_random_states_equal(before, _random_states())
    count = 105 if variant == "product-default" else 31
    assert diagnostics["learner_update_count"] == 2 * count
    assert list(diagnostics["learner_update_counts_by_name"].values()) == [count, count]
    assert diagnostics["selections"][0]["selected_k"] == actual.change_interval_sample_count
    assert diagnostics["selections"][0]["candidate_lengths"] == candidate_lengths(
        31, 5, "grid1" in variant
    )
    if variant == "product-default":
        assert actual == expected
