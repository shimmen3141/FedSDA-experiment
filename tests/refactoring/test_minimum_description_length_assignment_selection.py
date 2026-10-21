"""警報の後の変化区間の長さと帰属先を、最小記述長の原理で選ぶ純粋な計算を検証する。

期待値は、仮説の符号長をtestの中で直接書いて求める（研究ツールの関数は使わない）。
"""

import math

import pytest

from federated_learning_experiments.methods.fedsda.training_data_assignment import (
    minimum_description_length_assignment_selection as selection_module,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
    compute_switching_code_length,
    select_assignment_by_minimum_description_length,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES,
    DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
    MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME,
    validate_alarm_interval_assignment_policy,
)


def test_policy_names_and_validation():
    assert ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES == (
        DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
        MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME,
    )
    assert DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME == (
        "detector_change_span_and_mean_loss_increase_tolerance"
    )
    assert MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME == "minimum_description_length"
    for policy in ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES:
        validate_alarm_interval_assignment_policy(alarm_interval_assignment_policy=policy)
    with pytest.raises(ValueError):
        validate_alarm_interval_assignment_policy(alarm_interval_assignment_policy="other")

    class DerivedStr(str):
        pass

    with pytest.raises(TypeError):
        validate_alarm_interval_assignment_policy(
            alarm_interval_assignment_policy=DerivedStr(MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME)
        )


def test_switching_code_length_is_negative_log_probability_plus_log_destination_count():
    assert compute_switching_code_length(switching_probability=1 / 120, destination_count=3) == (
        -math.log(1 / 120) + math.log(3)
    )
    assert compute_switching_code_length(switching_probability=0.5, destination_count=1) == (
        -math.log(0.5) + math.log(1)
    )
    for probability in (0.0, 1.0, -0.1, 1.5, math.nan):
        with pytest.raises(ValueError):
            compute_switching_code_length(switching_probability=probability, destination_count=1)
    with pytest.raises(TypeError):
        compute_switching_code_length(switching_probability=True, destination_count=1)
    for count in (0, -1):
        with pytest.raises(ValueError):
            compute_switching_code_length(switching_probability=0.5, destination_count=count)
    with pytest.raises(TypeError):
        compute_switching_code_length(switching_probability=0.5, destination_count=True)


def _select(current, held, new, switching=4.0, grid=5):
    return select_assignment_by_minimum_description_length(
        current_model_log_losses=tuple(current),
        held_model_log_losses_by_model_id=tuple(
            (model_id, tuple(losses)) for model_id, losses in held
        ),
        new_model_code_lengths_by_change_interval_sample_count=dict(new),
        switching_code_length=switching,
        change_interval_grid_sample_count=grid,
    )


def test_current_model_is_selected_when_switching_does_not_shorten_the_description():
    current = [0.1] * 10
    held = [(7, [1.0] * 10)]
    new = {5: 9.0, 10: 9.0}
    assert _select(current, held, new) == MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="current_model",
        change_interval_sample_count=10,
        destination_model_id=None,
        description_length=sum(current),
    )


def test_held_model_is_selected_with_the_change_interval_where_it_explains_the_samples():
    # 前の5件は現在のモデルがよく当たり、後の5件は保有モデル7がよく当たる。
    current = [0.1] * 5 + [3.0] * 5
    held = [(7, [3.0] * 5 + [0.05] * 5), (9, [2.0] * 10)]
    new = {5: 8.0, 10: 16.0}
    switching = 4.0
    expected = (sum(current[:5]) + switching) + sum(held[0][1][5:])
    assert _select(current, held, new, switching) == MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="held_model",
        change_interval_sample_count=5,
        destination_model_id=7,
        description_length=expected,
    )


def test_new_model_is_selected_when_its_prequential_code_length_is_shortest():
    current = [0.1] * 5 + [3.0] * 5
    held = [(7, [3.0] * 10)]
    new = {5: 2.0, 10: 20.0}
    switching = 4.0
    expected = (sum(current[:5]) + switching) + 2.0
    assert _select(current, held, new, switching) == MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="new_model",
        change_interval_sample_count=5,
        destination_model_id=None,
        description_length=expected,
    )


def test_ties_prefer_current_then_shorter_interval_then_held_order_then_new_model():
    # 全部の仮説が同じ長さなら、現在のモデルのまま。
    current = [1.0] * 10
    held = [(3, [1.0] * 10), (8, [1.0] * 10)]
    zero_switch = 0.0
    assert (
        _select(current, held, {5: 5.0, 10: 10.0}, zero_switch).selected_hypothesis
        == "current_model"
    )
    # 切り替わる仮説が同じ長さなら、短い区間、保有の順、新しいモデルの順。
    current = [2.0] * 10
    held = [(3, [1.0] * 10), (8, [1.0] * 10)]
    selected = _select(current, held, {5: 5.0, 10: 10.0}, 0.0)
    # 区間5: 10+5=15（保有3・保有8・新しいモデルが同じ）。区間10: 0+10=10 が最短。
    assert (selected.change_interval_sample_count, selected.destination_model_id) == (10, 3)
    current = [2.0] * 10
    held = [(3, [1.0] * 10), (8, [1.0] * 10)]
    selected = _select(current, held, {5: 4.0, 10: 9.0}, 0.0)
    # 区間10: 保有3=10、保有8=10、新しいモデル=9（最短）。区間5: 10+5=15、新しいモデル10+4=14。
    assert (selected.selected_hypothesis, selected.change_interval_sample_count) == (
        "new_model",
        10,
    )
    # 保有3と保有8が同じ長さなら、保有の順（3）。
    current = [2.0] * 5
    held = [(3, [1.0] * 5), (8, [1.0] * 5)]
    selected = _select(current, held, {5: 9.0}, 0.0)
    assert (selected.selected_hypothesis, selected.destination_model_id) == ("held_model", 3)
    # 同じ長さの区間なら、短い方（先に計算した方）。
    current = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    held = [(4, [0.0] * 10)]
    selected = _select(current, held, {5: 99.0, 10: 99.0}, 0.0)
    assert selected.change_interval_sample_count == 10
    current = [0.0] * 5 + [1.0] * 5
    held = [(4, [0.0] * 10)]
    selected = _select(current, held, {5: 99.0, 10: 99.0}, 0.0)
    # 区間5: 0+0=0、区間10: 0。同じ長さで、短い区間（5）。
    assert selected.change_interval_sample_count == 5


def test_sequences_shorter_than_the_grid_keep_the_current_model_with_the_whole_sequence():
    current = [5.0, 5.0, 5.0]
    held = [(2, [0.0, 0.0, 0.0])]
    assert _select(current, held, {}, 0.0, grid=5) == MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="current_model",
        change_interval_sample_count=3,
        destination_model_id=None,
        description_length=15.0,
    )


def test_description_length_uses_the_same_summation_order():
    current = [0.1, 0.2, 0.3, 0.4, 0.5, 9.0, 9.0, 9.0, 9.0, 9.0]
    held = [(1, [9.0] * 5 + [0.11, 0.12, 0.13, 0.14, 0.15])]
    switching = -math.log(1 / 120) + math.log(2)
    selected = _select(current, held, {5: 50.0, 10: 50.0}, switching)
    assert selected.description_length == (sum(current[:5]) + switching) + sum(held[0][1][5:])


@pytest.mark.parametrize(
    "arguments",
    (
        dict(current=[0.1] * 5, held=[(1, [0.1] * 4)], new={5: 1.0}),
        dict(current=[0.1] * 5, held=[(1, [0.1] * 5), (1, [0.1] * 5)], new={5: 1.0}),
        dict(current=[0.1] * 10, held=[], new={5: 1.0}),
        dict(current=[-0.1] * 5, held=[], new={5: 1.0}),
        dict(current=[math.nan] * 5, held=[], new={5: 1.0}),
        dict(current=[0.1] * 5, held=[], new={5: -1.0}),
        dict(current=[0.1] * 5, held=[], new={5: math.inf}),
    ),
)
def test_invalid_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        _select(**arguments)


def test_invalid_switching_and_grid_are_rejected():
    for switching in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            _select([0.1] * 5, [], {5: 1.0}, switching)
    for grid in (0, -5):
        with pytest.raises(ValueError):
            _select([0.1] * 5, [], {5: 1.0}, 1.0, grid)
    with pytest.raises(TypeError):
        _select([0.1] * 5, [], {5: 1.0}, 1.0, True)
    with pytest.raises(ValueError):
        _select([], [], {}, 1.0)


def test_selection_record_rejects_inconsistent_fields():
    with pytest.raises(ValueError):
        MinimumDescriptionLengthAssignmentSelection(
            selected_hypothesis="held_model",
            change_interval_sample_count=5,
            destination_model_id=None,
            description_length=1.0,
        )
    with pytest.raises(ValueError):
        MinimumDescriptionLengthAssignmentSelection(
            selected_hypothesis="new_model",
            change_interval_sample_count=5,
            destination_model_id=3,
            description_length=1.0,
        )
    with pytest.raises(ValueError):
        MinimumDescriptionLengthAssignmentSelection(
            selected_hypothesis="other",
            change_interval_sample_count=5,
            destination_model_id=None,
            description_length=1.0,
        )
    with pytest.raises(ValueError):
        MinimumDescriptionLengthAssignmentSelection(
            selected_hypothesis="current_model",
            change_interval_sample_count=0,
            destination_model_id=None,
            description_length=1.0,
        )
    assert selection_module.SELECTED_HYPOTHESES == ("current_model", "held_model", "new_model")
