"""警報後の候補loss評価を旧最終判定と直接照合する。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_drift_experiment.provisional_model import ForwardValidationSession, select_forward_fitting_reference
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import CandidateModelTrainingAndAcceptanceSettings
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    select_available_reference_within_historical_loss_tolerance,
    evaluate_candidate_using_post_alarm_losses,
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
    monkeypatch.setattr(config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", evaluation_arguments["minimum_candidate_mean_loss_improvement"])
    legacy_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        legacy_session = ForwardValidationSession(
            proposal_position=10, estimated_change_point=8, episode_id=None, old_model_id=-100, detector="e-SR",
            candidate=object(), training_x=torch.empty((0, 1)), training_y=torch.empty((0, 1)), held_data=[],
            reference_models={model_id: object() for model_id in evaluation_arguments["reference_losses_by_model_id"]},
            target_count=evaluation_arguments["candidate_model_training_and_acceptance_settings"].candidate_post_alarm_validation_sample_count,
            reference_historical_means=dict(evaluation_arguments["reference_historical_mean_losses_by_model_id"]),
        )
        for sample_index, candidate_loss in enumerate(evaluation_arguments["candidate_losses"]):
            legacy_session.append_losses(candidate_loss, {model_id: losses[sample_index] for model_id, losses in evaluation_arguments["reference_losses_by_model_id"].items()})
        captured_decision = LegacyDecisionCapture()
        legacy_client = SimpleNamespace(
            _forward_validation=legacy_session,
            models={model_id: object() for model_id in evaluation_arguments["available_reference_model_ids"]},
            current_model_id=evaluation_arguments["current_training_model_id"],
            distance_threshold=evaluation_arguments["maximum_reference_mean_loss_increase"],
            provisional_model_decisions=captured_decision,
        )
        with pytest.raises(LegacyDecisionCaptured):
            FedSDAClient._finalize_forward_validation(legacy_client, 10 + len(evaluation_arguments["candidate_losses"]))
        return captured_decision.decision
    finally:
        torch.set_default_dtype(legacy_default_dtype)


def assert_evaluation_matches_legacy_decision(*, result, evaluation_arguments, monkeypatch):
    """新理由との対応は本体aliasではなくテストで明示する。"""
    legacy_decision = capture_legacy_candidate_decision(evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch)
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
    assert result.reference_historical_mean_loss == evaluation_arguments["reference_historical_mean_losses_by_model_id"].get(legacy_decision.reference_model_id)
    assert result.reusable_reference_model_id == (legacy_decision.reference_model_id if legacy_decision.reason.endswith("_refit") else None)


@pytest.mark.parametrize("sample_count", [2, 3, 10, 11])
@pytest.mark.parametrize("decision_reason", ["improve", "first_fail", "second_fail", "both_fail", "current", "alternative", "removed", "tie", "unrounded_sum"])
def test_post_alarm_candidate_evaluation_matches_legacy_finalization(monkeypatch, sample_count, decision_reason):
    """旧finalizeの全診断を奇数・偶数と参照選択の各分岐で照合する。"""
    candidate_losses = (.2,) * sample_count
    reference_losses_by_model_id = {9: (.8,) * sample_count, -3: (.9,) * sample_count}
    reference_historical_mean_losses_by_model_id = {}
    available_reference_model_ids = (9, -3)
    if decision_reason == "first_fail":
        candidate_losses = (.9,) * (sample_count // 2) + (.2,) * (sample_count - sample_count // 2)
    elif decision_reason == "second_fail":
        candidate_losses = (.2,) * (sample_count // 2) + (.9,) * (sample_count - sample_count // 2)
    elif decision_reason == "both_fail":
        candidate_losses = (.9,) * sample_count
    elif decision_reason in ("current", "alternative"):
        reference_losses_by_model_id[-3] = (.4,) * sample_count
        reference_historical_mean_losses_by_model_id = {9: .8, -3: .4} if decision_reason == "current" else {-3: .4}
    elif decision_reason == "removed":
        available_reference_model_ids = (-3,)
        reference_historical_mean_losses_by_model_id = {9: .8}
    elif decision_reason == "tie":
        reference_losses_by_model_id[-3] = (.8,) * sample_count
    elif decision_reason == "unrounded_sum":
        reference_losses_by_model_id = {9: (.800000001,) * sample_count, -3: (.8,) * sample_count}
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=candidate_losses,
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=.1,
        minimum_candidate_mean_loss_improvement=.01,
    )
    inputs_before_call = deepcopy(evaluation_arguments)
    result = evaluate_candidate_using_post_alarm_losses(**evaluation_arguments)
    assert_evaluation_matches_legacy_decision(result=result, evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch)
    assert evaluation_arguments == inputs_before_call


@pytest.mark.parametrize("candidate_loss,reference_loss,threshold", [
    (.2, .8, .60000001),
    (.30000004, .80000007, .5000000149011612),
    (.125, .25, .125),
    (0, 1, 0),
])
def test_post_alarm_candidate_rounding_boundary_preserves_decision_reason(monkeypatch, candidate_loss, reference_loss, threshold):
    """厳密等号と採否/理由の両方向の丸め差を保持する。"""
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(candidate_loss, candidate_loss),
        reference_losses_by_model_id={9: (reference_loss, reference_loss)},
        reference_historical_mean_losses_by_model_id={},
        available_reference_model_ids=(9,),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0,
        minimum_candidate_mean_loss_improvement=threshold,
    )
    result = evaluate_candidate_using_post_alarm_losses(**evaluation_arguments)
    assert_evaluation_matches_legacy_decision(result=result, evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch)
    if threshold == .60000001:
        assert not result.candidate_accepted and result.decision_reason == "both_segment_margins_passed"
    elif threshold == .5000000149011612:
        assert result.candidate_accepted and result.decision_reason == "both_segment_margins_failed"
    elif threshold == .125:
        assert not result.candidate_accepted and result.decision_reason == "both_segment_margins_failed"


@pytest.mark.parametrize("field_name,invalid_value", [
    ("candidate_model_training_and_acceptance_settings", None),
    ("candidate_model_training_and_acceptance_settings", make_acceptance_settings(sample_count=3)),
    ("candidate_losses", [.1, .2]),
    ("candidate_losses", (.1,)),
    ("candidate_losses", (.1, .2, .3)),
    ("candidate_losses", (True, .2)),
    ("candidate_losses", (float("nan"), .2)),
    ("candidate_losses", (float("inf"), .2)),
    ("candidate_losses", (-.1, .2)),
    ("candidate_losses", (1.1, .2)),
    ("candidate_losses", ("0.1", .2)),
    ("minimum_candidate_mean_loss_improvement", True),
    ("minimum_candidate_mean_loss_improvement", -.1),
    ("minimum_candidate_mean_loss_improvement", float("nan")),
    ("minimum_candidate_mean_loss_improvement", float("inf")),
    ("minimum_candidate_mean_loss_improvement", 10**500),
])
def test_post_alarm_candidate_invalid_evaluation_inputs_are_rejected_without_mutation(field_name, invalid_value):
    """既存件数条件と候補入力を検査し、再評価可能な入力を保持する。"""
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(.1, .2), reference_losses_by_model_id={9: (.8, .8)},
        reference_historical_mean_losses_by_model_id={}, available_reference_model_ids=(9,),
        current_training_model_id=9, maximum_reference_mean_loss_increase=.1,
        minimum_candidate_mean_loss_improvement=.01,
    )
    inputs_before_call = deepcopy(evaluation_arguments)
    with pytest.raises((TypeError, ValueError)) as exception_info:
        evaluate_candidate_using_post_alarm_losses(**(evaluation_arguments | {field_name: invalid_value}))
    assert str(exception_info.value)
    assert evaluation_arguments == inputs_before_call
