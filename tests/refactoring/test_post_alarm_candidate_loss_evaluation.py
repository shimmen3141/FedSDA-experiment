"""警報後の候補loss評価を旧最終判定と直接照合する。"""

from copy import deepcopy

import pytest
import torch

from federated_drift_experiment.provisional_model import select_forward_fitting_reference
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    select_available_reference_within_historical_loss_tolerance,
)


@pytest.mark.parametrize("reference_losses_by_model_id,reference_historical_mean_losses_by_model_id,available_reference_model_ids,current_training_model_id,maximum_reference_mean_loss_increase", [
    ({9: (.18, .18), 1: (.12, .12)}, {9: .1, 1: .1}, (9, 1), 9, .1),
    ({9: (.18, .18), 1: (.12, .12)}, {9: .1, 1: .1}, (9, 1), 0, .1),
    ({9: (.125, .125), -3: (.125, .125)}, {9: .1, -3: .1}, (9, -3), 0, .1),
    ({9: (.25, .25), 1: (.31, .29)}, {9: .1, 1: .1}, (9, 1), 0, .1),
    ({9: (.125, .125), 1: (.8, .8)}, {9: .1, 1: .1}, (1,), 9, .1),
    ({9: (.125, .125), 1: (.25, .25)}, {}, (9, 1), 0, .1),
    ({9: (.125, .125), 1: (.125, .125)}, {1: .1}, (9, 1), 9, .1),
    ({9: (.125, .125)}, {9: 0}, (9,), 9, .125),
    ({9: (.125, .125)}, {9: 0}, (9,), 9, .124999999),
    ({9: (0, 0)}, {9: 0}, (9,), 9, 0),
    ({9: (1, 1, 1)}, {9: 1}, (9,), 9, 0),
    ({9: (.2, .3, .4)}, {9: .5}, (), 9, .1),
])
def test_post_alarm_candidate_reference_selection_matches_legacy(reference_losses_by_model_id, reference_historical_mean_losses_by_model_id, available_reference_model_ids, current_training_model_id, maximum_reference_mean_loss_increase):
    """現行優先・平均同率・履歴欠落・消えた参照・等号を比較する。"""
    inputs_before_call = deepcopy((reference_losses_by_model_id, reference_historical_mean_losses_by_model_id, available_reference_model_ids))
    legacy_reference_model_id = select_forward_fitting_reference(
        {model_id: [float(reference_loss) for reference_loss in losses] for model_id, losses in reference_losses_by_model_id.items() if model_id in available_reference_model_ids},
        reference_historical_mean_losses_by_model_id,
        maximum_reference_mean_loss_increase,
        preferred_model_id=current_training_model_id,
    )
    assert select_available_reference_within_historical_loss_tolerance(
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=current_training_model_id,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
    ) == legacy_reference_model_id
    assert (reference_losses_by_model_id, reference_historical_mean_losses_by_model_id, available_reference_model_ids) == inputs_before_call


@pytest.mark.parametrize("field_name,invalid_value", [
    ("reference_losses_by_model_id", []),
    ("reference_losses_by_model_id", {}),
    ("reference_losses_by_model_id", {True: (.1, .2)}),
    ("reference_losses_by_model_id", {1.0: (.1, .2)}),
    ("reference_losses_by_model_id", {9: [.1, .2]}),
    ("reference_losses_by_model_id", {9: (.1,)}),
    ("reference_losses_by_model_id", {9: (.1, .2), 1: (.1, .2, .3)}),
    ("reference_losses_by_model_id", {9: (True, .2)}),
    ("reference_losses_by_model_id", {9: (float("nan"), .2)}),
    ("reference_losses_by_model_id", {9: (float("inf"), .2)}),
    ("reference_losses_by_model_id", {9: (-.1, .2)}),
    ("reference_losses_by_model_id", {9: (1.1, .2)}),
    ("reference_losses_by_model_id", {9: ("0.1", .2)}),
    ("reference_losses_by_model_id", {9: (torch.tensor(.1), .2)}),
    ("reference_historical_mean_losses_by_model_id", []),
    ("reference_historical_mean_losses_by_model_id", {True: .1}),
    ("reference_historical_mean_losses_by_model_id", {9: float("nan")}),
    ("reference_historical_mean_losses_by_model_id", {9: 1.1}),
    ("available_reference_model_ids", [9]),
    ("available_reference_model_ids", (9, 9)),
    ("available_reference_model_ids", (True,)),
    ("current_training_model_id", True),
    ("current_training_model_id", 1.0),
    ("maximum_reference_mean_loss_increase", True),
    ("maximum_reference_mean_loss_increase", float("nan")),
    ("maximum_reference_mean_loss_increase", float("inf")),
    ("maximum_reference_mean_loss_increase", -.1),
    ("maximum_reference_mean_loss_increase", 10**500),
])
def test_post_alarm_candidate_invalid_reference_inputs_are_rejected_without_mutation(field_name, invalid_value):
    """契約違反を強制変換せず拒否し、再使用する入力を保持する。"""
    reference_selection_arguments = dict(
        reference_losses_by_model_id={9: (.1, .2), 1: (.3, .4)},
        reference_historical_mean_losses_by_model_id={9: .1, 1: .2},
        available_reference_model_ids=(9, 1),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=.1,
    )
    inputs_before_call = deepcopy(reference_selection_arguments)
    with pytest.raises((TypeError, ValueError)) as exception_info:
        select_available_reference_within_historical_loss_tolerance(**(reference_selection_arguments | {field_name: invalid_value}))
    assert str(exception_info.value)
    assert reference_selection_arguments == inputs_before_call
