"""警報後の候補loss評価を旧最終判定と直接照合する。"""

import inspect
import random
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_drift_experiment.provisional_model import (
    ForwardValidationSession,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
    select_available_reference_within_historical_loss_tolerance,
)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("reference_losses_by_model_id", []),
        ("reference_losses_by_model_id", {}),
        ("reference_losses_by_model_id", {True: (0.1, 0.2)}),
        ("reference_losses_by_model_id", {1.0: (0.1, 0.2)}),
        ("reference_losses_by_model_id", {9: [0.1, 0.2]}),
        ("reference_losses_by_model_id", {9: (0.1,)}),
        ("reference_losses_by_model_id", {9: (0.1, 0.2), 1: (0.1, 0.2, 0.3)}),
        ("reference_losses_by_model_id", {9: (True, 0.2)}),
        ("reference_losses_by_model_id", {9: (float("nan"), 0.2)}),
        ("reference_losses_by_model_id", {9: (float("inf"), 0.2)}),
        ("reference_losses_by_model_id", {9: (-0.1, 0.2)}),
        ("reference_losses_by_model_id", {9: (1.1, 0.2)}),
        ("reference_losses_by_model_id", {9: ("0.1", 0.2)}),
        ("reference_losses_by_model_id", {9: (torch.tensor(0.1), 0.2)}),
        ("reference_historical_mean_losses_by_model_id", []),
        ("reference_historical_mean_losses_by_model_id", {True: 0.1}),
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
        ("maximum_reference_mean_loss_increase", -0.1),
        ("maximum_reference_mean_loss_increase", 10**500),
    ],
)
def test_post_alarm_candidate_invalid_reference_inputs_are_rejected_without_mutation(
    field_name, invalid_value
):
    """契約違反を強制変換せず拒否し、再使用する入力を保持する。"""
    reference_selection_arguments = dict(
        reference_losses_by_model_id={9: (0.1, 0.2), 1: (0.3, 0.4)},
        reference_historical_mean_losses_by_model_id={9: 0.1, 1: 0.2},
        available_reference_model_ids=(9, 1),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.1,
    )
    inputs_before_call = deepcopy(reference_selection_arguments)
    with pytest.raises((TypeError, ValueError)) as exception_info:
        select_available_reference_within_historical_loss_tolerance(
            **(reference_selection_arguments | {field_name: invalid_value})
        )
    assert str(exception_info.value)
    assert reference_selection_arguments == inputs_before_call


class LegacyDecisionCaptured(Exception):
    """旧decisionが記録された直後にモデル操作の実行を止める。"""


class LegacyDecisionCapture:
    """旧finalizeの判定だけを取得するテスト用収集先。"""

    def append(self, legacy_decision):
        self.decision = legacy_decision
        raise LegacyDecisionCaptured


def make_acceptance_settings(*, sample_count):
    """既存の最終方針を明示し、対象検証件数を設定する。"""
    return CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=sample_count,
    )


def capture_legacy_candidate_decision(*, evaluation_arguments, monkeypatch):
    """旧採否・理由・診断を全実行し、登録等の副作用の直前で中断する。"""
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_persistent")
    monkeypatch.setattr(
        config,
        "NEW_MODEL_EARLY_STOPPING_MIN_DELTA",
        evaluation_arguments["minimum_candidate_mean_loss_improvement"],
    )
    legacy_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        legacy_session = ForwardValidationSession(
            proposal_position=10,
            estimated_change_point=8,
            episode_id=None,
            old_model_id=-100,
            detector="e-SR",
            candidate=object(),
            training_x=torch.empty((0, 1)),
            training_y=torch.empty((0, 1)),
            held_data=[],
            reference_models={
                model_id: object()
                for model_id in evaluation_arguments["reference_losses_by_model_id"]
            },
            target_count=evaluation_arguments[
                "candidate_model_training_and_acceptance_settings"
            ].candidate_post_alarm_validation_sample_count,
            reference_historical_means=dict(
                evaluation_arguments["reference_historical_mean_losses_by_model_id"]
            ),
        )
        for sample_index, candidate_loss in enumerate(evaluation_arguments["candidate_losses"]):
            legacy_session.append_losses(
                candidate_loss,
                {
                    model_id: losses[sample_index]
                    for model_id, losses in evaluation_arguments[
                        "reference_losses_by_model_id"
                    ].items()
                },
            )
        captured_decision = LegacyDecisionCapture()
        legacy_client = SimpleNamespace(
            _forward_validation=legacy_session,
            models={
                model_id: object()
                for model_id in evaluation_arguments["available_reference_model_ids"]
            },
            current_model_id=evaluation_arguments["current_training_model_id"],
            distance_threshold=evaluation_arguments["maximum_reference_mean_loss_increase"],
            provisional_model_decisions=captured_decision,
        )
        with pytest.raises(LegacyDecisionCaptured):
            FedSDAClient._finalize_forward_validation(
                legacy_client, 10 + len(evaluation_arguments["candidate_losses"])
            )
        return captured_decision.decision
    finally:
        torch.set_default_dtype(legacy_default_dtype)


def assert_evaluation_matches_legacy_decision(*, result, evaluation_arguments, monkeypatch):
    """新理由との対応は本体aliasではなくテストで明示する。"""
    legacy_decision = capture_legacy_candidate_decision(
        evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch
    )
    legacy_reason_names = {
        "current_reference_within_historical_loss_tolerance": "current_reference_refit",
        "alternative_reference_within_historical_loss_tolerance": "alternative_reference_refit",
        "first_segment_margin_failed": "first_interval",
        "second_segment_margin_failed": "second_interval",
        "both_segment_margins_failed": "first_and_second",
        "both_segment_margins_passed": "accepted",
    }
    assert result.candidate_accepted == legacy_decision.accepted
    assert legacy_reason_names[result.decision_reason] == legacy_decision.reason
    assert result.comparison_reference_model_id == legacy_decision.reference_model_id
    assert result.validation_sample_count == legacy_decision.validation_count
    assert result.candidate_full_interval_mean_loss == legacy_decision.candidate_mean_loss
    assert result.reference_full_interval_mean_loss == legacy_decision.reference_mean_loss
    assert result.candidate_second_segment_mean_loss == legacy_decision.candidate_recent_loss
    assert result.reference_second_segment_mean_loss == legacy_decision.reference_recent_loss
    assert result.reference_historical_mean_loss == evaluation_arguments[
        "reference_historical_mean_losses_by_model_id"
    ].get(legacy_decision.reference_model_id)
    assert result.reusable_reference_model_id == (
        legacy_decision.reference_model_id if legacy_decision.reason.endswith("_refit") else None
    )


@pytest.mark.parametrize(
    "operation",
    [
        select_available_reference_within_historical_loss_tolerance,
        evaluate_candidate_using_post_alarm_losses,
    ],
)
def test_post_alarm_candidate_public_functions_require_explicit_keyword_arguments(operation):
    """類似loss列・ID・閾値の取り違えを避ける明示keyword契約。"""
    assert "*, " in str(inspect.signature(operation))
    with pytest.raises(TypeError):
        operation(object(), object())


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("candidate_model_training_and_acceptance_settings", None),
        (
            "candidate_model_training_and_acceptance_settings",
            make_acceptance_settings(sample_count=3),
        ),
        ("candidate_losses", [0.1, 0.2]),
        ("candidate_losses", (0.1,)),
        ("candidate_losses", (0.1, 0.2, 0.3)),
        ("candidate_losses", (True, 0.2)),
        ("candidate_losses", (float("nan"), 0.2)),
        ("candidate_losses", (float("inf"), 0.2)),
        ("candidate_losses", (-0.1, 0.2)),
        ("candidate_losses", (1.1, 0.2)),
        ("candidate_losses", ("0.1", 0.2)),
        ("minimum_candidate_mean_loss_improvement", True),
        ("minimum_candidate_mean_loss_improvement", -0.1),
        ("minimum_candidate_mean_loss_improvement", float("nan")),
        ("minimum_candidate_mean_loss_improvement", float("inf")),
        ("minimum_candidate_mean_loss_improvement", 10**500),
    ],
)
def test_post_alarm_candidate_invalid_evaluation_inputs_are_rejected_without_mutation(
    field_name, invalid_value
):
    """既存件数条件と候補入力を検査し、再評価可能な入力を保持する。"""
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(0.1, 0.2),
        reference_losses_by_model_id={9: (0.8, 0.8)},
        reference_historical_mean_losses_by_model_id={},
        available_reference_model_ids=(9,),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.1,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    inputs_before_call = deepcopy(evaluation_arguments)
    with pytest.raises((TypeError, ValueError)) as exception_info:
        evaluate_candidate_using_post_alarm_losses(
            **(evaluation_arguments | {field_name: invalid_value})
        )
    assert str(exception_info.value)
    assert evaluation_arguments == inputs_before_call


def test_post_alarm_candidate_results_inputs_and_random_states_are_independent():
    """共有dtype/device/乱数と結果copyを変更せず、反復時も同じ診断を返す。"""
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(0.2, 0.2),
        reference_losses_by_model_id={9: (0.8, 0.8)},
        reference_historical_mean_losses_by_model_id={},
        available_reference_model_ids=(9,),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.1,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    inputs_before_call = deepcopy(evaluation_arguments)
    evaluation_before_call = evaluate_candidate_using_post_alarm_losses(**evaluation_arguments)
    with pytest.raises(FrozenInstanceError):
        evaluation_before_call.candidate_accepted = False
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.random.get_rng_state().clone()
    legacy_default_dtype = torch.get_default_dtype()
    legacy_default_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            assert (
                evaluate_candidate_using_post_alarm_losses(**evaluation_arguments)
                == evaluation_before_call
            )
            assert (
                select_available_reference_within_historical_loss_tolerance(
                    reference_losses_by_model_id={9: (0.2, 0.2)},
                    reference_historical_mean_losses_by_model_id={9: 0.2},
                    available_reference_model_ids=(9,),
                    current_training_model_id=9,
                    maximum_reference_mean_loss_increase=0.1,
                )
                == 9
            )
            assert torch.get_default_dtype() == torch.float64
            assert str(torch.get_default_device()) == "meta"
    finally:
        torch.set_default_dtype(legacy_default_dtype)
    assert (
        evaluate_candidate_using_post_alarm_losses(**evaluation_arguments) == evaluation_before_call
    )
    assert evaluation_arguments == inputs_before_call
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.random.get_rng_state(), global_torch_random_state)
    assert torch.get_default_dtype() == legacy_default_dtype
    assert torch.get_default_device() == legacy_default_device
