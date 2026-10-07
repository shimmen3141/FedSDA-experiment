"""候補検証の判定記録と通常進行を実旧の処理へ対照する。"""

import math
from dataclasses import FrozenInstanceError

import pytest
from test_post_alarm_candidate_loss_evaluation import (
    assert_evaluation_matches_legacy_decision,
    capture_legacy_candidate_decision,
    make_acceptance_settings,
)

from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)


@pytest.mark.parametrize("sample_count", (2, 4, 5))
@pytest.mark.parametrize(
    "legacy_resolution_case", ("create", "reuse", "maintain", "create_rejected")
)
def test_validation_decision_record_matches_legacy(
    sample_count, legacy_resolution_case, monkeypatch
):
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(
            sample_count=sample_count
        ),
        candidate_losses=(0.01 if legacy_resolution_case == "create" else 0.9,) * sample_count,
        reference_losses_by_model_id={4: (0.2,) * sample_count, 9: (0.3,) * sample_count},
        reference_historical_mean_losses_by_model_id={
            "create": {},
            "reuse": {4: 0.2},
            "maintain": {9: 0.3},
            "create_rejected": {},
        }[legacy_resolution_case],
        available_reference_model_ids=(4, 9),
        current_training_model_id=9,
        maximum_reference_mean_loss_increase=0.0,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    post_alarm_candidate_loss_evaluation = evaluate_candidate_using_post_alarm_losses(
        **evaluation_arguments
    )
    legacy_decision = capture_legacy_candidate_decision(
        evaluation_arguments=evaluation_arguments, monkeypatch=monkeypatch
    )
    decision_record = PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=legacy_decision.position,
        resolution_sample_index=legacy_decision.resolution_position,
        detector_name=legacy_decision.detector,
        candidate_training_interval_sample_count=legacy_decision.training_count,
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
    )
    assert_evaluation_matches_legacy_decision(
        result=decision_record.post_alarm_candidate_loss_evaluation,
        evaluation_arguments=evaluation_arguments,
        monkeypatch=monkeypatch,
    )
    assert decision_record.proposal_sample_index == legacy_decision.position
    assert decision_record.resolution_sample_index == legacy_decision.resolution_position
    assert decision_record.detector_name == legacy_decision.detector
    assert (
        decision_record.candidate_training_interval_sample_count == legacy_decision.interval_count
    )
    assert (
        decision_record.candidate_training_interval_sample_count == legacy_decision.training_count
    )
    assert legacy_decision.validation_source == "forward"
    assert (
        decision_record.validation_completion_delay_sample_count == legacy_decision.resolution_delay
    )
    assert decision_record.full_validation_mean_loss_advantage == legacy_decision.full_margin
    assert decision_record.second_segment_mean_loss_advantage == legacy_decision.recent_margin
    if math.isnan(legacy_decision.reference_excess):
        assert decision_record.reference_mean_loss_difference_from_history is None
    else:
        assert (
            decision_record.reference_mean_loss_difference_from_history
            == legacy_decision.reference_excess
        )
    assert (
        decision_record.post_alarm_candidate_loss_evaluation is post_alarm_candidate_loss_evaluation
    )


def test_validation_decision_record_is_immutable():
    post_alarm_candidate_loss_evaluation = evaluate_candidate_using_post_alarm_losses(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(sample_count=2),
        candidate_losses=(0.1, 0.2),
        reference_losses_by_model_id={4: (0.3, 0.4)},
        reference_historical_mean_losses_by_model_id={4: 0.5},
        available_reference_model_ids=(4,),
        current_training_model_id=4,
        maximum_reference_mean_loss_increase=0.0,
        minimum_candidate_mean_loss_improvement=0.01,
    )
    decision_record = PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=10,
        resolution_sample_index=12,
        detector_name="class_esr",
        candidate_training_interval_sample_count=11,
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
    )
    # 履歴との差は負にもなる。increaseという非負に見える名前にしない。
    assert decision_record.reference_mean_loss_difference_from_history < 0
    for field_name in vars(decision_record):
        with pytest.raises(FrozenInstanceError):
            setattr(decision_record, field_name, None)
    with pytest.raises(FrozenInstanceError):
        post_alarm_candidate_loss_evaluation.candidate_accepted = True
    with pytest.raises(TypeError):
        PostAlarmCandidateValidationDecisionRecord(
            10, 12, "class_esr", 11, post_alarm_candidate_loss_evaluation
        )
