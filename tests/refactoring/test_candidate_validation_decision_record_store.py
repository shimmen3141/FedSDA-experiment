"""候補検証の判定記録のowner: 足した順の保持、読取りの不変、型の拒否。"""

from dataclasses import dataclass

import pytest

from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_validation_decision_record_store import (
    CandidateValidationDecisionRecordStore,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    PostAlarmCandidateLossEvaluation,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)


def make_completed_decision_record(*, proposal_sample_index):
    return PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=proposal_sample_index,
        resolution_sample_index=proposal_sample_index + 4,
        detector_name="test detector",
        candidate_training_interval_sample_count=6,
        post_alarm_candidate_loss_evaluation=PostAlarmCandidateLossEvaluation(
            comparison_reference_model_id=0,
            reusable_reference_model_id=None,
            candidate_accepted=True,
            decision_reason="both_segment_margins_passed",
            validation_sample_count=4,
            candidate_full_interval_mean_loss=0.125,
            reference_full_interval_mean_loss=0.5,
            candidate_second_segment_mean_loss=0.25,
            reference_second_segment_mean_loss=0.75,
            reference_historical_mean_loss=None,
        ),
    )


def make_incomplete_decision_record(*, proposal_sample_index):
    return IncompletePostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=proposal_sample_index,
        finalization_sample_index=proposal_sample_index + 2,
        detector_name="test detector",
        candidate_training_interval_sample_count=6,
        validation_sample_count=2,
    )


def test_store_keeps_both_decision_record_types_in_append_order():
    """確定の記録と、未完了用の記録を、足した順に、同じオブジェクトのまま保持する。"""
    decision_record_store = CandidateValidationDecisionRecordStore()
    assert decision_record_store.snapshot_candidate_validation_decision_records() == ()
    decision_records = (
        make_completed_decision_record(proposal_sample_index=30),
        make_completed_decision_record(proposal_sample_index=10),
        make_incomplete_decision_record(proposal_sample_index=50),
    )
    for decision_record in decision_records:
        assert (
            decision_record_store.append_candidate_validation_decision_record(
                decision_record=decision_record
            )
            is None
        )
    snapshot = decision_record_store.snapshot_candidate_validation_decision_records()
    assert type(snapshot) is tuple
    assert len(snapshot) == 3
    # 位置で並べ直さない。足した順のまま。
    assert all(
        kept_record is decision_record
        for kept_record, decision_record in zip(snapshot, decision_records, strict=True)
    )


def test_snapshot_does_not_change_after_later_appends_and_stores_are_independent():
    """読取りは、後からの追加で変わらない。2つのownerは、記録を共有しない。"""
    decision_record_store = CandidateValidationDecisionRecordStore()
    other_decision_record_store = CandidateValidationDecisionRecordStore()
    first_record = make_completed_decision_record(proposal_sample_index=3)
    decision_record_store.append_candidate_validation_decision_record(decision_record=first_record)
    snapshot = decision_record_store.snapshot_candidate_validation_decision_records()
    decision_record_store.append_candidate_validation_decision_record(
        decision_record=make_incomplete_decision_record(proposal_sample_index=9)
    )
    assert snapshot == (first_record,)
    assert len(decision_record_store.snapshot_candidate_validation_decision_records()) == 2
    assert other_decision_record_store.snapshot_candidate_validation_decision_records() == ()


@dataclass(frozen=True, kw_only=True)
class DerivedIncompleteDecisionRecord(IncompletePostAlarmCandidateValidationDecisionRecord):
    """派生型（exactな型ではない）。"""


@pytest.mark.parametrize(
    "invalid_decision_record",
    [
        None,
        {"proposal_sample_index": 3},
        (make_incomplete_decision_record(proposal_sample_index=3),),
        AdaptationRecord(
            adaptation_sample_index=3,
            detector_name="test detector",
            adaptation_outcome="post_alarm_validation_candidate_rejected",
            previous_training_model_id=0,
            current_training_model_id=0,
            estimated_change_point_sample_index=None,
            detection_episode_id=None,
        ),
        DerivedIncompleteDecisionRecord(
            proposal_sample_index=3,
            finalization_sample_index=5,
            detector_name="test detector",
            candidate_training_interval_sample_count=6,
            validation_sample_count=2,
        ),
    ],
    ids=["none", "dict", "tuple", "adaptation_record", "derived_type"],
)
def test_store_rejects_values_other_than_the_two_decision_record_types(invalid_decision_record):
    """2つの判定記録の型のどちらでもない値は、保持を変えずに拒否する。"""
    decision_record_store = CandidateValidationDecisionRecordStore()
    kept_record = make_completed_decision_record(proposal_sample_index=1)
    decision_record_store.append_candidate_validation_decision_record(decision_record=kept_record)
    with pytest.raises(TypeError):
        decision_record_store.append_candidate_validation_decision_record(
            decision_record=invalid_decision_record
        )
    assert decision_record_store.snapshot_candidate_validation_decision_records() == (kept_record,)


def test_append_requires_keyword_argument():
    decision_record_store = CandidateValidationDecisionRecordStore()
    with pytest.raises(TypeError):
        decision_record_store.append_candidate_validation_decision_record(
            make_completed_decision_record(proposal_sample_index=1)
        )
    assert decision_record_store.snapshot_candidate_validation_decision_records() == ()
