"""未完了候補の不足判定記録を、実旧の終端確定記録へ対照する。"""

import math
from dataclasses import FrozenInstanceError, fields, replace
from inspect import Parameter, signature

import pytest
import torch
from test_post_alarm_candidate_validation_progress import build_validation_progress_oracle

from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize(
    "observed_validation_sample_count,processed_sample_count", ((0, 0), (1, 53), (3, 60))
)
def test_incomplete_validation_decision_record_matches_legacy(
    class_count, observed_validation_sample_count, processed_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        progress_arguments, _, _, legacy_client, _ = build_validation_progress_oracle(
            class_count=class_count,
            legacy_resolution_case="create",
            pending_sample_count=0,
            monkeypatch=monkeypatch,
        )
        validation_session = progress_arguments["validation_session"]
        legacy_session = legacy_client._forward_validation
        legacy_session.candidate_losses = legacy_session.candidate_losses[
            :observed_validation_sample_count
        ]
        legacy_session.reference_losses = {
            model_id: losses[:observed_validation_sample_count]
            for model_id, losses in legacy_session.reference_losses.items()
        }
        collection = PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=progress_arguments[
                "candidate_model_training_and_acceptance_settings"
            ],
            proposal_sample_index=validation_session.proposal_sample_index,
            reference_model_ids=tuple(legacy_session.reference_models),
        )
        for sample_index in range(observed_validation_sample_count):
            collection.observe_losses_after_label_observation(
                sample_index=validation_session.proposal_sample_index + 1 + sample_index,
                candidate_loss=legacy_session.candidate_losses[sample_index],
                reference_losses_by_model_id={
                    model_id: losses[sample_index]
                    for model_id, losses in legacy_session.reference_losses.items()
                },
            )
        validation_session = replace(
            validation_session, post_alarm_candidate_loss_collection=collection
        )
        collection_state_snapshot = collection.get_state_snapshot()
        assert not collection_state_snapshot.ready_for_acceptance_evaluation
        assert collection_state_snapshot.validation_sample_count == observed_validation_sample_count
        legacy_client.processed_samples = processed_sample_count
        legacy_client.finalize_incomplete_forward_validation()
        legacy_decision = legacy_client.provisional_model_decisions[-1]
        decision_record = IncompletePostAlarmCandidateValidationDecisionRecord(
            proposal_sample_index=validation_session.proposal_sample_index,
            finalization_sample_index=max(
                validation_session.proposal_sample_index, processed_sample_count - 1
            ),
            detector_name=validation_session.detector_name,
            candidate_training_interval_sample_count=len(
                validation_session.training_input_features
            ),
            validation_sample_count=collection_state_snapshot.validation_sample_count,
        )
        assert decision_record.proposal_sample_index == legacy_decision.position
        assert decision_record.finalization_sample_index == legacy_decision.resolution_position
        assert decision_record.detector_name == legacy_decision.detector
        assert (
            decision_record.candidate_training_interval_sample_count
            == legacy_decision.interval_count
            == legacy_decision.training_count
        )
        assert decision_record.validation_sample_count == legacy_decision.validation_count
        assert (
            decision_record.validation_sample_count
            != decision_record.candidate_training_interval_sample_count
        )
        assert decision_record.finalization_delay_sample_count == legacy_decision.resolution_delay
        # 不足recordの型そのものが棄却・比較未成立を表す。旧理由やNaNを新fieldへ持ち込まない。
        assert type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord
        assert legacy_decision.accepted is False
        assert legacy_decision.reason == "insufficient_forward_data"
        assert legacy_decision.reference_model_id is None
        assert legacy_decision.validation_source == "forward"
        for field_name in (
            "candidate_mean_loss",
            "reference_mean_loss",
            "candidate_recent_loss",
            "reference_recent_loss",
            "full_margin",
            "recent_margin",
            "reference_excess",
            "reference_historical_mean",
        ):
            assert math.isnan(getattr(legacy_decision, field_name))
            assert not hasattr(decision_record, field_name)
        assert not hasattr(decision_record, "reference_model_id")
        assert collection.get_state_snapshot() == collection_state_snapshot


def test_incomplete_validation_decision_record_is_immutable():
    decision_record = IncompletePostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=53,
        finalization_sample_index=59,
        detector_name="class_esr",
        candidate_training_interval_sample_count=11,
        validation_sample_count=3,
    )
    assert tuple(field.name for field in fields(decision_record)) == (
        "proposal_sample_index",
        "finalization_sample_index",
        "detector_name",
        "candidate_training_interval_sample_count",
        "validation_sample_count",
    )
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
        for parameter in signature(
            IncompletePostAlarmCandidateValidationDecisionRecord
        ).parameters.values()
    )
    assert decision_record.finalization_delay_sample_count == 6
    for field_name in vars(decision_record):
        with pytest.raises(FrozenInstanceError):
            setattr(decision_record, field_name, None)
    with pytest.raises(FrozenInstanceError):
        decision_record.finalization_delay_sample_count = 0
    with pytest.raises(TypeError):
        IncompletePostAlarmCandidateValidationDecisionRecord(53, 59, "class_esr", 11, 3)
    for field_name in vars(decision_record):
        with pytest.raises(TypeError):
            IncompletePostAlarmCandidateValidationDecisionRecord(
                **{
                    name: value
                    for name, value in vars(decision_record).items()
                    if name != field_name
                }
            )
