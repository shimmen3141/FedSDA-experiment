"""警報の変化区間の解決を、結果recordと実旧の警報処理の両方で検証する。"""

import math
import random
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import assert_held_model_states_match_legacy
from test_adopted_candidate_local_adoption import assert_training_samples_match_legacy
from test_alarm_interval_model_reuse_assessment import LEGACY_INITIALIZATION_NAMES
from test_candidate_epoch_training import assert_candidate_epoch_training_matches_legacy
from test_joint_model_parameter_update import assert_nested_state_equal, run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_post_alarm_candidate_validation_sample_observation import (
    assert_collected_losses_match_legacy,
)
from test_post_alarm_candidate_validation_session_start import build_session_start_oracle
from test_post_alarm_reference_model_fixation import (
    assert_fixed_references_match_legacy,
    set_overall_loss_statistics_in_both_implementations,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.runtime import (
    alarm_change_interval_resolution as resolution_module,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
    AlarmChangeIntervalResolution,
    resolve_alarm_change_interval,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


def test_alarm_change_interval_resolution_is_immutable():
    assert ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES == (
        "alarm_interval_held_model_reused",
        "alarm_interval_current_model_maintained",
        "alarm_interval_candidate_validation_started",
    )
    assert type(ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES) is tuple
    assert tuple(field.name for field in fields(AlarmChangeIntervalResolution)) == (
        "resolution_outcome",
        "alarm_interval_reuse_assessment",
        "assigned_model_id",
        "training_model_assignment_change",
        "started_validation_session",
    )
    assert all(field.kw_only for field in fields(AlarmChangeIntervalResolution))
    alarm_interval_reuse_assessment = AlarmIntervalModelReuseAssessment(
        baseline_supported_interval_mean_losses_by_model_id=((4, 0.5), (-103, 0.25)),
        reusable_mean_losses_by_model_id=((-103, 0.25),),
    )
    training_model_assignment_change = TrainingModelAssignmentChange(
        previous_model_id=4, current_model_id=-103
    )
    started_validation_session = object.__new__(PostAlarmCandidateValidationSession)
    # 3つの結果種別それぞれのfieldの組を、そのまま保持する。
    for (
        resolution_outcome,
        assigned_model_id,
        field_value,
        expected_started_validation_session,
    ) in (
        ("alarm_interval_held_model_reused", -103, training_model_assignment_change, None),
        ("alarm_interval_current_model_maintained", 4, None, None),
        ("alarm_interval_candidate_validation_started", None, None, started_validation_session),
    ):
        alarm_change_interval_resolution = AlarmChangeIntervalResolution(
            resolution_outcome=resolution_outcome,
            alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
            assigned_model_id=assigned_model_id,
            training_model_assignment_change=field_value,
            started_validation_session=expected_started_validation_session,
        )
        assert alarm_change_interval_resolution.resolution_outcome == resolution_outcome
        assert (
            alarm_change_interval_resolution.alarm_interval_reuse_assessment
            is alarm_interval_reuse_assessment
        )
        assert alarm_change_interval_resolution.assigned_model_id == assigned_model_id
        assert alarm_change_interval_resolution.training_model_assignment_change is field_value
        assert (
            alarm_change_interval_resolution.started_validation_session
            is expected_started_validation_session
        )
        for field_name in (
            "resolution_outcome",
            "alarm_interval_reuse_assessment",
            "assigned_model_id",
            "training_model_assignment_change",
            "started_validation_session",
        ):
            with pytest.raises(FrozenInstanceError):
                setattr(alarm_change_interval_resolution, field_name, None)
    # 正式値以外の結果種別（候補検証の確定処理の値や旧のaction名を含む）は拒否する。
    for invalid_case in (
        "held_reference_model_reused",
        "current_model_maintained",
        "candidate_validation_started",
        "reuse",
        "",
        None,
        1,
    ):
        with pytest.raises(ValueError):
            AlarmChangeIntervalResolution(
                resolution_outcome=invalid_case,
                alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
                assigned_model_id=4,
                training_model_assignment_change=None,
                started_validation_session=None,
            )
    # 全fieldがkeyword必須。
    with pytest.raises(TypeError):
        AlarmChangeIntervalResolution(
            "alarm_interval_current_model_maintained",
            alarm_interval_reuse_assessment,
            4,
            None,
            None,
        )
    with pytest.raises(TypeError):
        AlarmChangeIntervalResolution(
            resolution_outcome="alarm_interval_current_model_maintained",
            alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
            assigned_model_id=4,
        )
    assert not hasattr(resolution_module, "__all__")


# 警報区間の結果種別と、実旧がイベントへ記録するactionの対応（test内だけ）。
LEGACY_ACTION_BY_ALARM_INTERVAL_RESOLUTION_OUTCOME = {
    "alarm_interval_held_model_reused": "reuse",
    "alarm_interval_current_model_maintained": "maintain",
    "alarm_interval_candidate_validation_started": "create_pending",
}

# 現行は保有順の最後のモデル。履歴基準が区間平均と同じなら適合、1/4なら不適合、平均0なら基準なし。
ALARM_INTERVAL_RESOLUTION_CASES = {
    "other_model_reused": dict(
        held_model_ids=(4, 9),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            4: (3, legacy_interval_mean_losses_by_model_id[4]),
            9: (3, legacy_interval_mean_losses_by_model_id[9] / 4),
        },
        expected_resolution_outcome="alarm_interval_held_model_reused",
        expected_assigned_model_id=4,
    ),
    "negative_id_model_reused": dict(
        held_model_ids=(4, -3, 9),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            4: (3, legacy_interval_mean_losses_by_model_id[4] / 4),
            -3: (2, (legacy_interval_mean_losses_by_model_id[-3] + 1) / 2),
            9: (3, legacy_interval_mean_losses_by_model_id[9] / 4),
        },
        expected_resolution_outcome="alarm_interval_held_model_reused",
        expected_assigned_model_id=-3,
    ),
    # 後ろ2件を同じ値のモデルにする。現行（最後）も適合するが、同率なので保有順で先の-3を選ぶ。
    "tie_selects_earlier_model_although_current_fits": dict(
        held_model_ids=(9, -3, 4),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            9: (3, legacy_interval_mean_losses_by_model_id[9] / 4),
            -3: (3, legacy_interval_mean_losses_by_model_id[-3]),
            4: (3, legacy_interval_mean_losses_by_model_id[4]),
        },
        expected_resolution_outcome="alarm_interval_held_model_reused",
        expected_assigned_model_id=-3,
    ),
    "current_model_maintained": dict(
        held_model_ids=(4, 9),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            4: (3, legacy_interval_mean_losses_by_model_id[4] / 4),
            9: (3, legacy_interval_mean_losses_by_model_id[9]),
        },
        expected_resolution_outcome="alarm_interval_current_model_maintained",
        expected_assigned_model_id=9,
    ),
    "only_current_model_has_usable_history": dict(
        held_model_ids=(9, -3, 4),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            9: (3, 0.0),
            -3: (1, legacy_interval_mean_losses_by_model_id[-3]),
            4: (2, legacy_interval_mean_losses_by_model_id[4]),
        },
        expected_resolution_outcome="alarm_interval_current_model_maintained",
        expected_assigned_model_id=4,
    ),
    "single_model_maintained": dict(
        held_model_ids=(4,),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            4: (3, legacy_interval_mean_losses_by_model_id[4]),
        },
        expected_resolution_outcome="alarm_interval_current_model_maintained",
        expected_assigned_model_id=4,
    ),
    "no_model_fits": dict(
        held_model_ids=(9, -3, 4),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            model_id: (3, mean_loss / 4)
            for model_id, mean_loss in legacy_interval_mean_losses_by_model_id.items()
        },
        expected_resolution_outcome="alarm_interval_candidate_validation_started",
        expected_assigned_model_id=None,
    ),
    "no_model_has_usable_history": dict(
        held_model_ids=(9, -3, 4),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            9: (3, 0.0),
            -3: (1, legacy_interval_mean_losses_by_model_id[-3]),
            4: (0, 0.0),
        },
        expected_resolution_outcome="alarm_interval_candidate_validation_started",
        expected_assigned_model_id=None,
    ),
    "lowest_mean_model_lacks_history": dict(
        held_model_ids=(9, -3, 4),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            model_id: (3, 0.0)
            if mean_loss == min(legacy_interval_mean_losses_by_model_id.values())
            else (3, mean_loss / 4)
            for model_id, mean_loss in legacy_interval_mean_losses_by_model_id.items()
        },
        expected_resolution_outcome="alarm_interval_candidate_validation_started",
        expected_assigned_model_id=None,
    ),
    "single_model_does_not_fit": dict(
        held_model_ids=(4,),
        statistics_by_model_id=lambda legacy_interval_mean_losses_by_model_id: {
            4: (3, legacy_interval_mean_losses_by_model_id[4] / 4),
        },
        expected_resolution_outcome="alarm_interval_candidate_validation_started",
        expected_assigned_model_id=None,
    ),
}


def build_alarm_change_interval_resolution_oracle(
    *,
    class_count,
    alarm_interval_resolution_case,
    monkeypatch,
    optimizer_variant="standard",
    candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model",
):
    """同じ実NN・統計・標本・計数・区間を持つ新owner群と実旧clientを準備する。"""
    held_model_ids = ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
        "held_model_ids"
    ]
    session_start_arguments, shared_optimizer_owners, legacy_client = build_session_start_oracle(
        class_count=class_count,
        optimizer_variant=optimizer_variant,
        monkeypatch=monkeypatch,
        held_model_ids=held_model_ids,
    )
    registry = session_start_arguments["held_model_training_state_registry"]
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    assert legacy_client.current_model_id == held_model_ids[-1]
    if alarm_interval_resolution_case == "tie_selects_earlier_model_although_current_fits":
        held_model_training_states[2].classifier.load_state_dict(
            held_model_training_states[1].classifier.state_dict()
        )
        legacy_client.models[held_model_ids[2]].load_state_dict(
            legacy_client.models[held_model_ids[1]].state_dict()
        )
    # 上流oracleが実旧clientへ置いた既存の標本と計数を、同じobject・同じ順で新ownerへ写す。
    training_sample_store = ModelTrainingSampleStore()
    counts_store = ModelTrainingAndAssignmentCountsStore()
    for model_id, legacy_training_samples in legacy_client.train_data_store.items():
        training_sample_store.append_model_training_samples(
            model_id=model_id,
            training_samples=tuple(
                ObservedTrainingSample(
                    input_features=legacy_training_sample[0],
                    observed_class_labels=legacy_training_sample[1],
                )
                for legacy_training_sample in legacy_training_samples
            ),
        )
    for model_id, sample_count in legacy_client.model_training_examples.items():
        counts_store.record_completed_model_training(
            model_id=model_id,
            trained_sample_count=sample_count,
            parameter_update_step_count=legacy_client.model_optimizer_steps[model_id],
        )
    for model_id, concept_counts in legacy_client.model_concept_counts.items():
        for observed_concept_id, sample_count in concept_counts.items():
            for _ in range(sample_count):
                counts_store.record_assigned_sample_concept(
                    model_id=model_id, observed_concept_id=observed_concept_id
                )
    assert_training_samples_match_legacy(
        training_sample_store=training_sample_store, legacy_client=legacy_client
    )
    assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
    input_features = session_start_arguments["input_features"]
    observed_class_labels = session_start_arguments["observed_class_labels"]
    sample_count = len(input_features)
    with torch.no_grad():
        legacy_interval_mean_losses_by_model_id = {
            model_id: float(
                torch.mean(
                    legacy_model.per_sample_error(input_features, observed_class_labels)
                ).item()
            )
            for model_id, legacy_model in legacy_client.models.items()
        }
    assert all(0 < mean_loss < 1 for mean_loss in legacy_interval_mean_losses_by_model_id.values())
    set_overall_loss_statistics_in_both_implementations(
        loss_statistics_store=session_start_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
        statistics_by_model_id=ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
            "statistics_by_model_id"
        ](legacy_interval_mean_losses_by_model_id),
    )
    monkeypatch.setattr(
        config,
        "NEW_MODEL_INITIALIZATION",
        LEGACY_INITIALIZATION_NAMES[candidate_parameter_initialization_source],
    )
    resolution_arguments = dict(
        change_interval_training_samples=tuple(
            ObservedTrainingSample(
                input_features=input_features[sample_index : sample_index + 1],
                observed_class_labels=observed_class_labels[sample_index : sample_index + 1],
            )
            for sample_index in range(sample_count)
        ),
        # 概念IDなしの標本を含める。
        change_interval_sample_concept_ids=tuple(
            None if sample_index % 4 == 2 else sample_index % 2
            for sample_index in range(sample_count)
        ),
        # 不適合の差（平均の3/4）より小さく、0より大きい許容増加量。
        maximum_alarm_interval_mean_loss_increase=min(
            legacy_interval_mean_losses_by_model_id.values()
        )
        / 8,
        held_model_training_state_registry=registry,
        loss_statistics_store=session_start_arguments["loss_statistics_store"],
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=counts_store,
        current_training_model_assignment=session_start_arguments[
            "current_training_model_assignment"
        ],
        candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source=candidate_parameter_initialization_source
        ),
        **{
            field_name: session_start_arguments[field_name]
            for field_name in (
                "architecture_reference_classifier",
                "parameter_optimizer_settings",
                "candidate_epoch_training_settings",
                "candidate_model_training_and_acceptance_settings",
                "proposal_sample_index",
                "estimated_change_point_sample_index",
                "detection_episode_id",
                "detector_name",
            )
        },
    )
    return (
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_interval_mean_losses_by_model_id,
    )


def resolve_alarm_change_interval_in_legacy_client(
    *, legacy_client, resolution_arguments, monkeypatch
):
    """実旧_resolve_driftを、吸収・帰属切替・初期値選択・session開始を差し替えずに実行する。"""
    change_interval_training_samples = resolution_arguments["change_interval_training_samples"]
    sample_count = len(change_interval_training_samples)
    legacy_adaptation_events = []

    def record_legacy_adaptation_event(**event_fields):
        legacy_adaptation_events.append(event_fields)

    monkeypatch.setattr(config, "MIN_DRIFT_DATA", 1)
    legacy_client.verbose = False
    legacy_client._forward_validation = None
    legacy_client.reuse_selection_counts = Counter()
    legacy_client._estimated_new_concept_span = lambda sample_index: sample_count
    legacy_client._reset_drift_detectors = lambda: None
    legacy_client._record_adaptation_event = record_legacy_adaptation_event
    legacy_client.distance_threshold = resolution_arguments[
        "maximum_alarm_interval_mean_loss_increase"
    ]
    legacy_client.buffer = deque(
        (training_sample.input_features, training_sample.observed_class_labels, observed_concept_id)
        for training_sample, observed_concept_id in zip(
            change_interval_training_samples,
            resolution_arguments["change_interval_sample_concept_ids"],
        )
    )
    with patch.object(
        legacy_client, "_update_new_model_epochs", wraps=legacy_client._update_new_model_epochs
    ) as legacy_epoch_training_calls:
        legacy_drift_type = legacy_client._resolve_drift(
            resolution_arguments["proposal_sample_index"],
            resolution_arguments["estimated_change_point_sample_index"],
            resolution_arguments["detection_episode_id"],
        )
    return legacy_drift_type, legacy_adaptation_events, legacy_epoch_training_calls


def snapshot_alarm_change_interval_resolution_state(
    *, resolution_arguments, shared_optimizer_owners
):
    registry = resolution_arguments["held_model_training_state_registry"]
    held_model_training_states = (
        registry.snapshot_ordered_held_model_training_states()
        if type(registry) is HeldModelTrainingStateRegistry
        else ()
    )
    loss_statistics_store = resolution_arguments["loss_statistics_store"]
    training_sample_store = resolution_arguments["training_sample_store"]
    counts_store = resolution_arguments["model_training_and_assignment_counts_store"]
    current_training_model_assignment = resolution_arguments["current_training_model_assignment"]
    return dict(
        resolution_arguments=dict(resolution_arguments),
        held_model_training_states=held_model_training_states,
        parameter_snapshot=snapshot_parameter_values_and_gradients(
            tuple(
                parameter
                for state in held_model_training_states
                for parameter in state.classifier.parameters()
            )
        ),
        training_modes=tuple(state.classifier.training for state in held_model_training_states),
        optimizer_state_snapshots=tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in tuple(
                state.concept_specific_parameter_optimizer_state
                for state in held_model_training_states
            )
            + tuple(shared_optimizer_owners)
        ),
        loss_statistics_snapshot=(
            loss_statistics_store.get_state_snapshot()
            if type(loss_statistics_store) is ModelAndClassLossStatisticsStore
            else None
        ),
        # 標本はTensorの同一性で比べる（値の比較は真偽値にならない）。
        training_sample_snapshot=(
            tuple(
                (
                    collection.model_id,
                    tuple(
                        (
                            id(training_sample.input_features),
                            id(training_sample.observed_class_labels),
                        )
                        for training_sample in collection.training_samples
                    ),
                )
                for collection in training_sample_store.snapshot_ordered_model_training_samples()
            )
            if type(training_sample_store) is ModelTrainingSampleStore
            else None
        ),
        counts_snapshot=(
            counts_store.snapshot_model_training_and_assignment_counts()
            if type(counts_store) is ModelTrainingAndAssignmentCountsStore
            else None
        ),
        current_training_model_id=(
            current_training_model_assignment.current_training_model_id
            if type(current_training_model_assignment) is CurrentTrainingModelAssignment
            else None
        ),
        random_states=(torch.get_rng_state().clone(), random.getstate(), np.random.get_state()),
    )


def assert_alarm_change_interval_resolution_state_unchanged(
    state_snapshot, *, expected_rng_state=None
):
    """expected_rng_stateを渡した場合だけ、torch乱数をその値と照合する（候補検証開始の成功時）。"""
    resolution_arguments = state_snapshot["resolution_arguments"]
    current_state_snapshot = snapshot_alarm_change_interval_resolution_state(
        resolution_arguments=resolution_arguments, shared_optimizer_owners=()
    )
    assert len(current_state_snapshot["held_model_training_states"]) == len(
        state_snapshot["held_model_training_states"]
    )
    for state, previous_state in zip(
        current_state_snapshot["held_model_training_states"],
        state_snapshot["held_model_training_states"],
    ):
        assert state is previous_state
        assert state.classifier is previous_state.classifier
    assert_parameter_values_and_gradients_unchanged(state_snapshot["parameter_snapshot"])
    assert current_state_snapshot["training_modes"] == state_snapshot["training_modes"]
    for owner, previous_optimizer, previous_optimizer_state in state_snapshot[
        "optimizer_state_snapshots"
    ]:
        assert owner.parameter_optimizer is previous_optimizer
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
    for field_name in (
        "loss_statistics_snapshot",
        "training_sample_snapshot",
        "counts_snapshot",
        "current_training_model_id",
    ):
        assert current_state_snapshot[field_name] == state_snapshot[field_name], field_name
    torch_random_state, python_random_state, numpy_random_state = state_snapshot["random_states"]
    assert torch.equal(
        torch.get_rng_state(),
        torch_random_state if expected_rng_state is None else expected_rng_state,
    )
    assert random.getstate() == python_random_state
    numpy_state = np.random.get_state()
    assert numpy_state[0] == numpy_random_state[0]
    assert np.array_equal(numpy_state[1], numpy_random_state[1])
    assert numpy_state[2:] == numpy_random_state[2:]


def assert_alarm_change_interval_resolution_matches_legacy(
    *,
    alarm_change_interval_resolution,
    resolution_arguments,
    shared_optimizer_owners,
    legacy_client,
    legacy_drift_type,
    legacy_adaptation_events,
    legacy_epoch_training_calls,
    initial_training_model_id,
):
    """結果recordと、標本・計数・統計・現行ID・保有モデル・開始sessionを実旧の警報処理後と照合する。"""
    assert type(alarm_change_interval_resolution) is AlarmChangeIntervalResolution
    resolution_outcome = alarm_change_interval_resolution.resolution_outcome
    assert len(legacy_adaptation_events) == 1
    assert (
        legacy_adaptation_events[0]["action"]
        == LEGACY_ACTION_BY_ALARM_INTERVAL_RESOLUTION_OUTCOME[resolution_outcome]
    )
    assert legacy_drift_type == (
        1 if resolution_outcome == "alarm_interval_held_model_reused" else 0
    )
    assert legacy_adaptation_events[0]["old_model_id"] == initial_training_model_id
    current_training_model_id = resolution_arguments[
        "current_training_model_assignment"
    ].current_training_model_id
    assert current_training_model_id == legacy_client.current_model_id
    assert legacy_adaptation_events[0]["new_model_id"] == current_training_model_id
    training_model_assignment_change = (
        alarm_change_interval_resolution.training_model_assignment_change
    )
    if resolution_outcome == "alarm_interval_held_model_reused":
        assert type(training_model_assignment_change) is TrainingModelAssignmentChange
        assert (
            training_model_assignment_change.previous_model_id,
            training_model_assignment_change.current_model_id,
        ) == legacy_client.local_model_changes[-1]
        assert training_model_assignment_change.previous_model_id == initial_training_model_id
        assert alarm_change_interval_resolution.assigned_model_id == current_training_model_id
        assert current_training_model_id != initial_training_model_id
        assert legacy_client.local_switch_positions == [
            resolution_arguments["proposal_sample_index"]
        ]
        assert dict(legacy_client.reuse_selection_counts) == {"alternative_fit": 1}
    else:
        assert training_model_assignment_change is None
        assert legacy_client.local_model_changes == []
        assert legacy_client.local_switch_positions == []
        assert current_training_model_id == initial_training_model_id
    if resolution_outcome == "alarm_interval_current_model_maintained":
        assert alarm_change_interval_resolution.assigned_model_id == initial_training_model_id
        assert dict(legacy_client.reuse_selection_counts) == {"current_fit": 1}
    # 標本（同じTensor object）、割当概念計数、損失統計。候補検証開始では両実装とも変わっていない。
    assert_training_samples_match_legacy(
        training_sample_store=resolution_arguments["training_sample_store"],
        legacy_client=legacy_client,
    )
    assert_model_counts_match_legacy(
        counts_store=resolution_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert_store_statistics_match_legacy(
        loss_statistics_store=resolution_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
    )
    registry = resolution_arguments["held_model_training_state_registry"]
    change_interval_input_features = torch.cat(
        [
            training_sample.input_features
            for training_sample in resolution_arguments["change_interval_training_samples"]
        ]
    )
    assert_held_model_states_match_legacy(
        registry=registry,
        shared_optimizer_owners=shared_optimizer_owners[:-1],
        legacy_client=legacy_client,
        input_features=change_interval_input_features,
    )
    started_validation_session = alarm_change_interval_resolution.started_validation_session
    legacy_session = legacy_client._forward_validation
    if resolution_outcome != "alarm_interval_candidate_validation_started":
        assert started_validation_session is None
        assert legacy_session is None
        assert len(legacy_epoch_training_calls.call_args_list) == 0
        return
    assert alarm_change_interval_resolution.assigned_model_id is None
    assert dict(legacy_client.reuse_selection_counts) == {}
    assert type(started_validation_session) is PostAlarmCandidateValidationSession
    assert legacy_session is not None
    assert_candidate_epoch_training_matches_legacy(
        candidate_training_state=started_validation_session.candidate_training_state,
        legacy_candidate=legacy_session.candidate,
    )
    assert_fixed_references_match_legacy(
        fixed_reference_models=started_validation_session.fixed_reference_models,
        legacy_reference_models=legacy_session.reference_models,
        registry=registry,
        input_features=change_interval_input_features,
    )
    assert (
        started_validation_session.fixed_reference_models.reference_historical_mean_losses_by_model_id
        == legacy_session.reference_historical_means
    )
    assert started_validation_session.proposal_sample_index == legacy_session.proposal_position
    assert (
        started_validation_session.estimated_change_point_sample_index
        == legacy_session.estimated_change_point
    )
    assert started_validation_session.detection_episode_id == legacy_session.episode_id
    assert started_validation_session.detector_name == legacy_session.detector
    assert started_validation_session.initial_training_model_id == legacy_session.old_model_id
    # 区間は標本順の連結で、旧のbx/byと同じ値。
    assert torch.equal(
        started_validation_session.training_input_features, legacy_session.training_x
    )
    assert torch.equal(
        started_validation_session.training_observed_class_labels, legacy_session.training_y
    )
    assert (
        started_validation_session.training_input_features.shape == legacy_session.training_x.shape
    )
    # 保留標本は渡した標本列そのもので、旧の保留標本と同じTensor objectを同じ順に持つ。
    assert (
        started_validation_session.pending_assignment_training_samples
        is resolution_arguments["change_interval_training_samples"]
    )
    assert len(legacy_session.held_data) == len(
        started_validation_session.pending_assignment_training_samples
    )
    for training_sample, legacy_training_sample, observed_concept_id in zip(
        started_validation_session.pending_assignment_training_samples,
        legacy_session.held_data,
        resolution_arguments["change_interval_sample_concept_ids"],
    ):
        assert training_sample.input_features is legacy_training_sample[0]
        assert training_sample.observed_class_labels is legacy_training_sample[1]
        assert observed_concept_id == legacy_training_sample[2]
    candidate_epoch_training_result = started_validation_session.candidate_epoch_training_result
    assert candidate_epoch_training_result.completed_epoch_count == sum(
        legacy_epoch_training_call.args[2]
        for legacy_epoch_training_call in legacy_epoch_training_calls.call_args_list
    )
    assert (
        candidate_epoch_training_result.candidate_trained_sample_count
        == legacy_session.candidate_training_examples
    )
    assert (
        candidate_epoch_training_result.candidate_parameter_update_step_count
        == legacy_session.candidate_optimizer_steps
    )
    collection_state = (
        started_validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
    )
    assert collection_state.proposal_sample_index == legacy_session.proposal_position
    assert collection_state.candidate_losses == tuple(legacy_session.candidate_losses) == ()
    assert collection_state.reference_losses_by_model_id == tuple(
        (model_id, tuple(losses)) for model_id, losses in legacy_session.reference_losses.items()
    )
    assert collection_state.required_validation_sample_count == legacy_session.target_count


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "sgd"])
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    [
        alarm_interval_resolution_case
        for alarm_interval_resolution_case in ALARM_INTERVAL_RESOLUTION_CASES
        if ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
            "expected_resolution_outcome"
        ]
        != "alarm_interval_candidate_validation_started"
    ],
)
def test_alarm_change_interval_resolution_matches_legacy(
    class_count, optimizer_variant, alarm_interval_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(841)
        (
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        initial_training_model_id = legacy_client.current_model_id
        (
            legacy_drift_type,
            legacy_adaptation_events,
            legacy_epoch_training_calls,
        ) = resolve_alarm_change_interval_in_legacy_client(
            legacy_client=legacy_client,
            resolution_arguments=resolution_arguments,
            monkeypatch=monkeypatch,
        )
        state_snapshot = snapshot_alarm_change_interval_resolution_state(
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        alarm_change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        assert (
            alarm_change_interval_resolution.resolution_outcome
            == ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
                "expected_resolution_outcome"
            ]
        )
        assert (
            alarm_change_interval_resolution.assigned_model_id
            == ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
                "expected_assigned_model_id"
            ]
        )
        assert (
            alarm_change_interval_resolution.alarm_interval_reuse_assessment.selected_reuse_model_id
            == alarm_change_interval_resolution.assigned_model_id
        )
        assert_alarm_change_interval_resolution_matches_legacy(
            alarm_change_interval_resolution=alarm_change_interval_resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            legacy_adaptation_events=legacy_adaptation_events,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            initial_training_model_id=initial_training_model_id,
        )
        # 再利用・維持は、標本・計数・統計・現行ID以外（モデル、optimizer、3乱数）を変えない。
        assert_parameter_values_and_gradients_unchanged(state_snapshot["parameter_snapshot"])
        for owner, previous_optimizer, previous_optimizer_state in state_snapshot[
            "optimizer_state_snapshots"
        ]:
            assert owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert torch.equal(torch.get_rng_state(), state_snapshot["random_states"][0])
        assert random.getstate() == state_snapshot["random_states"][1]
        assert np.array_equal(np.random.get_state()[1], state_snapshot["random_states"][2][1])
        # 吸収先以外のモデルの標本と統計は変わらない。区間の標本は吸収先の末尾へ標本順に入る。
        sample_count = len(resolution_arguments["change_interval_training_samples"])
        for (model_id, training_sample_snapshot), collection in zip(
            state_snapshot["training_sample_snapshot"],
            resolution_arguments["training_sample_store"].snapshot_ordered_model_training_samples(),
        ):
            assert collection.model_id == model_id
            if model_id != alarm_change_interval_resolution.assigned_model_id:
                assert len(collection.training_samples) == len(training_sample_snapshot)
                continue
            assert len(collection.training_samples) == len(training_sample_snapshot) + sample_count
            for training_sample, expected_training_sample in zip(
                collection.training_samples[-sample_count:],
                resolution_arguments["change_interval_training_samples"],
            ):
                assert training_sample is expected_training_sample
        for (model_id, loss_statistics), (_, previous_loss_statistics) in zip(
            resolution_arguments["loss_statistics_store"].get_state_snapshot(),
            state_snapshot["loss_statistics_snapshot"],
        ):
            if model_id != alarm_change_interval_resolution.assigned_model_id:
                assert loss_statistics == previous_loss_statistics
            else:
                assert (
                    loss_statistics.overall_loss_moments.observed_loss_count
                    == previous_loss_statistics.overall_loss_moments.observed_loss_count
                    + sample_count
                )
        # 概念IDなしの標本は割当概念計数へ加えない。
        assigned_concept_counts = resolution_arguments[
            "model_training_and_assignment_counts_store"
        ].get_model_assigned_sample_concept_counts(
            model_id=alarm_change_interval_resolution.assigned_model_id
        )
        previous_concept_counts = state_snapshot[
            "counts_snapshot"
        ].assigned_sample_counts_by_model_and_concept_id.get(
            alarm_change_interval_resolution.assigned_model_id, {}
        )
        assert sum(assigned_concept_counts.values()) - sum(previous_concept_counts.values()) == sum(
            observed_concept_id is not None
            for observed_concept_id in resolution_arguments["change_interval_sample_concept_ids"]
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("candidate_parameter_initialization_source", LEGACY_INITIALIZATION_NAMES)
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    [
        alarm_interval_resolution_case
        for alarm_interval_resolution_case in ALARM_INTERVAL_RESOLUTION_CASES
        if ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
            "expected_resolution_outcome"
        ]
        == "alarm_interval_candidate_validation_started"
    ],
)
def test_alarm_change_interval_resolution_starts_candidate_validation_like_legacy(
    class_count,
    candidate_parameter_initialization_source,
    alarm_interval_resolution_case,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(842)
        (
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            candidate_parameter_initialization_source=candidate_parameter_initialization_source,
            monkeypatch=monkeypatch,
        )
        initial_training_model_id = legacy_client.current_model_id
        initial_rng_state = torch.get_rng_state().clone()
        (
            legacy_drift_type,
            legacy_adaptation_events,
            legacy_epoch_training_calls,
        ) = resolve_alarm_change_interval_in_legacy_client(
            legacy_client=legacy_client,
            resolution_arguments=resolution_arguments,
            monkeypatch=monkeypatch,
        )
        expected_rng_state = torch.get_rng_state().clone()
        assert not torch.equal(expected_rng_state, initial_rng_state)
        torch.set_rng_state(initial_rng_state)
        state_snapshot = snapshot_alarm_change_interval_resolution_state(
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        alarm_change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        assert (
            alarm_change_interval_resolution.resolution_outcome
            == "alarm_interval_candidate_validation_started"
        )
        assert (
            alarm_change_interval_resolution.alarm_interval_reuse_assessment.selected_reuse_model_id
            is None
        )
        # 保有モデル・統計・標本・計数・現行IDは不変で、torch乱数は実旧の処理後と一致する。
        assert_alarm_change_interval_resolution_state_unchanged(
            state_snapshot, expected_rng_state=expected_rng_state
        )
        assert_alarm_change_interval_resolution_matches_legacy(
            alarm_change_interval_resolution=alarm_change_interval_resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            legacy_adaptation_events=legacy_adaptation_events,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            initial_training_model_id=initial_training_model_id,
        )
        # 候補は保有モデルと実体を共有しない。
        held_parameter_storage_addresses = {
            parameter.data_ptr()
            for state in resolution_arguments[
                "held_model_training_state_registry"
            ].snapshot_ordered_held_model_training_states()
            for parameter in state.classifier.parameters()
        }
        for parameter in alarm_change_interval_resolution.started_validation_session.candidate_training_state.candidate_classifier.parameters():
            assert parameter.data_ptr() not in held_parameter_storage_addresses


class TupleSubclass(tuple):
    pass


class IntSubclass(int):
    pass


def replace_change_interval_sample(*, resolution_arguments, sample_index, **replaced_fields):
    """標本列のsample_index番目だけを、指定のfieldを差し替えた標本へ置き換えた標本列を返す。"""
    change_interval_training_samples = list(
        resolution_arguments["change_interval_training_samples"]
    )
    training_sample = change_interval_training_samples[sample_index]
    change_interval_training_samples[sample_index] = ObservedTrainingSample(
        **{
            "input_features": training_sample.input_features,
            "observed_class_labels": training_sample.observed_class_labels,
            **replaced_fields,
        }
    )
    return tuple(change_interval_training_samples)


# 共通入力の不正。値は、正常なresolution_argumentsから不正な引数値を作る関数。
INVALID_COMMON_INPUT_CASES = {
    "samples_list": (
        "change_interval_training_samples",
        lambda resolution_arguments: list(resolution_arguments["change_interval_training_samples"]),
        TypeError,
    ),
    "samples_tuple_subclass": (
        "change_interval_training_samples",
        lambda resolution_arguments: TupleSubclass(
            resolution_arguments["change_interval_training_samples"]
        ),
        TypeError,
    ),
    "samples_empty": (
        "change_interval_training_samples",
        lambda resolution_arguments: (),
        ValueError,
    ),
    "sample_plain_tuple": (
        "change_interval_training_samples",
        lambda resolution_arguments: (
            resolution_arguments["change_interval_training_samples"][:-1]
            + (
                (
                    resolution_arguments["change_interval_training_samples"][-1].input_features,
                    resolution_arguments["change_interval_training_samples"][
                        -1
                    ].observed_class_labels,
                ),
            )
        ),
        TypeError,
    ),
    "sample_features_list": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments, sample_index=1, input_features=[[0.0, 1.0]]
        ),
        TypeError,
    ),
    "sample_features_tensor_subclass": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.nn.Parameter(torch.zeros(1, 2), requires_grad=False),
        ),
        TypeError,
    ),
    "sample_labels_list": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments, sample_index=1, observed_class_labels=[[0.0]]
        ),
        TypeError,
    ),
    "sample_labels_tensor_subclass": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            observed_class_labels=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
        ),
        TypeError,
    ),
    "sample_two_rows": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.zeros(2, 2),
            observed_class_labels=torch.zeros(2, 1),
        ),
        ValueError,
    ),
    "sample_two_feature_rows_with_one_label": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.zeros(2, 2),
        ),
        ValueError,
    ),
    "sample_feature_count_mismatch": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.zeros(1, 3),
        ),
        ValueError,
    ),
    "first_sample_feature_count_mismatch": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=0,
            input_features=torch.zeros(1, 3),
        ),
        ValueError,
    ),
    "first_sample_features_one_dimension": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments, sample_index=0, input_features=torch.zeros(2)
        ),
        ValueError,
    ),
    "first_sample_features_zero_dimension": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=0,
            input_features=torch.tensor(0.0),
        ),
        ValueError,
    ),
    "sample_features_sparse_layout": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.zeros(1, 2).to_sparse(),
        ),
        ValueError,
    ),
    "sample_subclass_instance": (
        "change_interval_training_samples",
        lambda resolution_arguments: (
            resolution_arguments["change_interval_training_samples"][:-1]
            + (
                type("InputSubclass", (ObservedTrainingSample,), {})(
                    input_features=resolution_arguments["change_interval_training_samples"][
                        -1
                    ].input_features,
                    observed_class_labels=resolution_arguments["change_interval_training_samples"][
                        -1
                    ].observed_class_labels,
                ),
            )
        ),
        TypeError,
    ),
    "sample_features_one_dimension": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments, sample_index=1, input_features=torch.zeros(2)
        ),
        ValueError,
    ),
    "sample_features_float64": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.zeros(1, 2, dtype=torch.float64),
        ),
        ValueError,
    ),
    "sample_labels_shape": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            observed_class_labels=torch.zeros(1),
        ),
        ValueError,
    ),
    "sample_labels_int64": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            observed_class_labels=torch.zeros(1, 1, dtype=torch.int64),
        ),
        ValueError,
    ),
    "sample_features_infinite": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            input_features=torch.full((1, 2), math.inf),
        ),
        ValueError,
    ),
    "sample_label_out_of_range": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            observed_class_labels=torch.full((1, 1), 4.0),
        ),
        ValueError,
    ),
    "sample_label_not_integer": (
        "change_interval_training_samples",
        lambda resolution_arguments: replace_change_interval_sample(
            resolution_arguments=resolution_arguments,
            sample_index=1,
            observed_class_labels=torch.full((1, 1), 0.5),
        ),
        ValueError,
    ),
    "concept_ids_list": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: list(
            resolution_arguments["change_interval_sample_concept_ids"]
        ),
        TypeError,
    ),
    "concept_ids_tuple_subclass": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: TupleSubclass(
            resolution_arguments["change_interval_sample_concept_ids"]
        ),
        TypeError,
    ),
    "concept_ids_too_short": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: resolution_arguments["change_interval_sample_concept_ids"][
            :-1
        ],
        ValueError,
    ),
    "concept_ids_too_long": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: (
            resolution_arguments["change_interval_sample_concept_ids"] + (0,)
        ),
        ValueError,
    ),
    "concept_id_bool": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: (
            (True,) + resolution_arguments["change_interval_sample_concept_ids"][1:]
        ),
        TypeError,
    ),
    "concept_id_float": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: (
            resolution_arguments["change_interval_sample_concept_ids"][:-1] + (1.0,)
        ),
        TypeError,
    ),
    "concept_id_int_subclass": (
        "change_interval_sample_concept_ids",
        lambda resolution_arguments: (
            resolution_arguments["change_interval_sample_concept_ids"][:-1] + (IntSubclass(1),)
        ),
        TypeError,
    ),
    **{
        "owner_none:" + field_name: (field_name, lambda resolution_arguments: None, TypeError)
        for field_name in (
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
            "current_training_model_assignment",
        )
    },
    **{
        "owner_subclass:" + field_name: (field_name, "subclass", TypeError)
        for field_name in (
            "held_model_training_state_registry",
            "loss_statistics_store",
            "training_sample_store",
            "model_training_and_assignment_counts_store",
            "current_training_model_assignment",
        )
    },
    "no_held_model": (
        "held_model_training_state_registry",
        lambda resolution_arguments: HeldModelTrainingStateRegistry(),
        LookupError,
    ),
    "current_model_not_held": (
        "current_training_model_assignment",
        lambda resolution_arguments: CurrentTrainingModelAssignment(initial_model_id=77),
        LookupError,
    ),
    "increase_bool": (
        "maximum_alarm_interval_mean_loss_increase",
        lambda resolution_arguments: True,
        TypeError,
    ),
    "increase_nan": (
        "maximum_alarm_interval_mean_loss_increase",
        lambda resolution_arguments: math.nan,
        ValueError,
    ),
    "increase_negative": (
        "maximum_alarm_interval_mean_loss_increase",
        lambda resolution_arguments: -0.125,
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case", list(INVALID_COMMON_INPUT_CASES))
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "current_model_maintained", "no_model_fits"],
)
def test_alarm_change_interval_resolution_rejects_common_inputs_without_mutation(
    invalid_case, alarm_interval_resolution_case, monkeypatch
):
    field_name, field_value, expected_exception = INVALID_COMMON_INPUT_CASES[invalid_case]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(843)
        (
            resolution_arguments,
            shared_optimizer_owners,
            _,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=4,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            monkeypatch=monkeypatch,
        )
        valid_state_snapshot = snapshot_alarm_change_interval_resolution_state(
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        if field_value == "subclass":
            field_value = type("InputSubclass", (type(resolution_arguments[field_name]),), {})
            field_value = object.__new__(field_value)
            field_value.__dict__.update(resolution_arguments[field_name].__dict__)
            resolution_arguments[field_name] = field_value
        else:
            resolution_arguments[field_name] = field_value(resolution_arguments)
        with patch.object(
            resolution_module,
            "evaluate_held_models_for_alarm_interval_reuse",
            wraps=resolution_module.evaluate_held_models_for_alarm_interval_reuse,
        ) as operation_calls:
            with pytest.raises((TypeError, ValueError, LookupError)) as exception_info:
                resolve_alarm_change_interval(**resolution_arguments)
        assert type(exception_info.value) is expected_exception
        # 区間の値と許容損失増加量だけは区間評価が拒否する。それ以外は評価より前に拒否する。
        assert operation_calls.call_count == (
            1
            if invalid_case
            in (
                "sample_features_infinite",
                "sample_label_out_of_range",
                "sample_label_not_integer",
                "increase_bool",
                "increase_nan",
                "increase_negative",
            )
            else 0
        )
        # 差し替えていない元のowner群と3乱数は変わらない。
        assert_alarm_change_interval_resolution_state_unchanged(valid_state_snapshot)


# 候補検証の開始だけに使う入力の不正。
INVALID_START_ONLY_INPUT_CASES = {
    "initialization_settings_none": (
        "candidate_parameter_initialization_settings",
        None,
        TypeError,
    ),
    "architecture_reference_none": ("architecture_reference_classifier", None, TypeError),
    "optimizer_settings_none": ("parameter_optimizer_settings", None, TypeError),
    "epoch_training_settings_none": ("candidate_epoch_training_settings", None, TypeError),
    "acceptance_settings_none": (
        "candidate_model_training_and_acceptance_settings",
        None,
        TypeError,
    ),
    "proposal_index_bool": ("proposal_sample_index", True, TypeError),
    "proposal_index_negative": ("proposal_sample_index", -1, ValueError),
    "change_point_after_proposal": ("estimated_change_point_sample_index", 41, ValueError),
    "change_point_float": ("estimated_change_point_sample_index", 35.0, TypeError),
    "episode_id_negative": ("detection_episode_id", -1, ValueError),
    "detector_name_blank": ("detector_name", " ", ValueError),
    "detector_name_none": ("detector_name", None, TypeError),
}


@pytest.mark.parametrize("invalid_case", list(INVALID_START_ONLY_INPUT_CASES))
def test_alarm_change_interval_resolution_rejects_start_only_inputs_only_without_reuse(
    invalid_case, monkeypatch
):
    field_name, field_value, expected_exception = INVALID_START_ONLY_INPUT_CASES[invalid_case]
    # 再利用先がない条件では、全状態と3乱数を変えずに拒否する。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(844)
        (
            resolution_arguments,
            shared_optimizer_owners,
            _,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=4, alarm_interval_resolution_case="no_model_fits", monkeypatch=monkeypatch
        )
        valid_state_snapshot = snapshot_alarm_change_interval_resolution_state(
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        resolution_arguments[field_name] = field_value
        with pytest.raises((TypeError, ValueError, LookupError)) as exception_info:
            resolve_alarm_change_interval(**resolution_arguments)
        assert type(exception_info.value) is expected_exception
        assert_alarm_change_interval_resolution_state_unchanged(valid_state_snapshot)
    # 再利用・維持の条件では検査されず、正常入力と同じ結果と最終状態になる。
    for alarm_interval_resolution_case in ("other_model_reused", "current_model_maintained"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(844)
            (
                resolution_arguments,
                shared_optimizer_owners,
                legacy_client,
                _,
            ) = build_alarm_change_interval_resolution_oracle(
                class_count=4,
                alarm_interval_resolution_case=alarm_interval_resolution_case,
                monkeypatch=monkeypatch,
            )
            initial_training_model_id = legacy_client.current_model_id
            (
                legacy_drift_type,
                legacy_adaptation_events,
                legacy_epoch_training_calls,
            ) = resolve_alarm_change_interval_in_legacy_client(
                legacy_client=legacy_client,
                resolution_arguments=resolution_arguments,
                monkeypatch=monkeypatch,
            )
            alarm_change_interval_resolution = resolve_alarm_change_interval(
                **{**resolution_arguments, field_name: field_value}
            )
            assert (
                alarm_change_interval_resolution.resolution_outcome
                == ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
                    "expected_resolution_outcome"
                ]
            )
            # 旧は正常入力で実行した結果。新の結果と最終状態が、正常な引数に対する照合と一致する。
            assert_alarm_change_interval_resolution_matches_legacy(
                alarm_change_interval_resolution=alarm_change_interval_resolution,
                resolution_arguments=resolution_arguments,
                shared_optimizer_owners=shared_optimizer_owners,
                legacy_client=legacy_client,
                legacy_drift_type=legacy_drift_type,
                legacy_adaptation_events=legacy_adaptation_events,
                legacy_epoch_training_calls=legacy_epoch_training_calls,
                initial_training_model_id=initial_training_model_id,
            )


@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "current_model_maintained", "no_model_fits"],
)
def test_alarm_change_interval_resolution_calls_only_selected_operations_in_order(
    alarm_interval_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(845)
        (
            resolution_arguments,
            _,
            _,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=4,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            monkeypatch=monkeypatch,
        )
        initial_training_model_id = resolution_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        held_model_ids = ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
            "held_model_ids"
        ]
        operation_calls = []

        def record_operation_call(operation_name, operation):
            """呼出しを記録してから元の部品をそのまま呼ぶwrapperを返す。"""

            def recorded_operation(*operation_arguments, **operation_keyword_arguments):
                operation_calls.append((operation_name, operation_keyword_arguments))
                return operation(*operation_arguments, **operation_keyword_arguments)

            return recorded_operation

        for operation_name in (
            "evaluate_held_models_for_alarm_interval_reuse",
            "absorb_assigned_training_samples_into_held_model",
            "snapshot_classifier_parameters",
            "select_candidate_initial_parameter_snapshot",
            "start_post_alarm_candidate_validation_session",
        ):
            monkeypatch.setattr(
                resolution_module,
                operation_name,
                record_operation_call(operation_name, getattr(resolution_module, operation_name)),
            )
        monkeypatch.setattr(
            CurrentTrainingModelAssignment,
            "assign_model_for_training",
            record_operation_call(
                "assign_model_for_training",
                CurrentTrainingModelAssignment.assign_model_for_training,
            ),
        )
        alarm_change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        change_interval_training_samples = resolution_arguments["change_interval_training_samples"]
        expected_resolution_outcome = ALARM_INTERVAL_RESOLUTION_CASES[
            alarm_interval_resolution_case
        ]["expected_resolution_outcome"]
        assert alarm_change_interval_resolution.resolution_outcome == expected_resolution_outcome
        # 区間評価は最初に1回だけ。標本順に連結した区間全体（実旧のbx/byと同じ値）を受け取る。
        assert operation_calls[0][0] == "evaluate_held_models_for_alarm_interval_reuse"
        assert torch.equal(
            operation_calls[0][1]["input_features"],
            torch.cat(
                [
                    training_sample.input_features
                    for training_sample in change_interval_training_samples
                ]
            ),
        )
        assert torch.equal(
            operation_calls[0][1]["observed_class_labels"],
            torch.cat(
                [
                    training_sample.observed_class_labels
                    for training_sample in change_interval_training_samples
                ]
            ),
        )
        assert operation_calls[0][1]["input_features"].shape == (
            len(change_interval_training_samples),
            2,
        )
        assert (
            operation_calls[0][1]["maximum_alarm_interval_mean_loss_increase"]
            == resolution_arguments["maximum_alarm_interval_mean_loss_increase"]
        )
        if expected_resolution_outcome == "alarm_interval_candidate_validation_started":
            # 保有順の全モデルのsnapshot→初期値選択→session開始。吸収と帰属切替えは呼ばない。
            assert [operation_name for operation_name, _ in operation_calls] == (
                ["evaluate_held_models_for_alarm_interval_reuse"]
                + ["snapshot_classifier_parameters"] * len(held_model_ids)
                + [
                    "select_candidate_initial_parameter_snapshot",
                    "start_post_alarm_candidate_validation_session",
                ]
            )
            registry = resolution_arguments["held_model_training_state_registry"]
            for (_, operation_keyword_arguments), state in zip(
                operation_calls[1:-2], registry.snapshot_ordered_held_model_training_states()
            ):
                assert operation_keyword_arguments["classifier"] is state.classifier
            assert (
                operation_calls[-2][1]["evaluated_mean_losses_by_model_id"]
                == alarm_change_interval_resolution.alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id
            )
            assert operation_calls[-2][1]["current_training_model_id"] == initial_training_model_id
            assert tuple(operation_calls[-2][1]["available_parameter_snapshots_by_model_id"]) == (
                held_model_ids
            )
            assert (
                operation_calls[-2][1]["settings"]
                is resolution_arguments["candidate_parameter_initialization_settings"]
            )
            # session開始は、評価と同じ区間のTensorと、渡した標本列そのものを保留標本として受け取る。
            assert (
                operation_calls[-1][1]["input_features"] is operation_calls[0][1]["input_features"]
            )
            assert (
                operation_calls[-1][1]["observed_class_labels"]
                is operation_calls[0][1]["observed_class_labels"]
            )
            assert (
                operation_calls[-1][1]["pending_assignment_training_samples"]
                is change_interval_training_samples
            )
            for field_name in (
                "architecture_reference_classifier",
                "parameter_optimizer_settings",
                "candidate_epoch_training_settings",
                "candidate_model_training_and_acceptance_settings",
                "held_model_training_state_registry",
                "loss_statistics_store",
                "current_training_model_assignment",
                "proposal_sample_index",
                "estimated_change_point_sample_index",
                "detection_episode_id",
                "detector_name",
            ):
                assert operation_calls[-1][1][field_name] is resolution_arguments[field_name]
            return
        # 吸収→帰属切替えの順に1回ずつ。初期値選択とsession開始は呼ばない。
        assert [operation_name for operation_name, _ in operation_calls] == [
            "evaluate_held_models_for_alarm_interval_reuse",
            "absorb_assigned_training_samples_into_held_model",
            "assign_model_for_training",
        ]
        expected_assigned_model_id = ALARM_INTERVAL_RESOLUTION_CASES[
            alarm_interval_resolution_case
        ]["expected_assigned_model_id"]
        assert operation_calls[1][1]["model_id"] == expected_assigned_model_id
        assert (
            operation_calls[1][1]["assigned_training_samples"] is change_interval_training_samples
        )
        assert (
            operation_calls[1][1]["assigned_sample_concept_ids"]
            is resolution_arguments["change_interval_sample_concept_ids"]
        )
        assert operation_calls[2][1] == {"model_id": expected_assigned_model_id}


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "negative_id_model_reused", "current_model_maintained"],
)
def test_resolved_alarm_change_interval_continues_joint_training(
    class_count,
    optimizer_variant,
    update_shared_features,
    alarm_interval_resolution_case,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(846)
        (
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = resolution_arguments["held_model_training_state_registry"]
        training_sample_store = resolution_arguments["training_sample_store"]
        counts_store = resolution_arguments["model_training_and_assignment_counts_store"]
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        legacy_training_batches = []
        legacy_client.updates_per_sample = 1
        legacy_client.backbone_gradient_diagnostics = defaultdict(float)
        legacy_client._sample_training_batches = lambda: legacy_training_batches
        change_interval_input_features = torch.cat(
            [
                training_sample.input_features
                for training_sample in resolution_arguments["change_interval_training_samples"]
            ]
        )

        def run_joint_update_in_both_implementations():
            # 両実装とも、各自の標本storeにある全標本をモデルごとの固定batchにする。
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            model_training_sample_collections = (
                training_sample_store.snapshot_ordered_model_training_samples()
            )
            legacy_training_batches[:] = [
                (
                    model_id,
                    torch.cat(
                        [
                            legacy_training_sample[0]
                            for legacy_training_sample in legacy_training_samples
                        ]
                    ),
                    torch.cat(
                        [
                            legacy_training_sample[1]
                            for legacy_training_sample in legacy_training_samples
                        ]
                    ),
                )
                for model_id, legacy_training_samples in legacy_client.train_data_store.items()
            ]
            assert tuple(
                training_binding.model_id for training_binding in training_bindings
            ) == tuple(collection.model_id for collection in model_training_sample_collections)
            participating_training_batches = tuple(
                ParticipatingModelTrainingBatch(
                    classifier=training_binding.classifier,
                    concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
                    input_features=torch.cat(
                        [
                            training_sample.input_features
                            for training_sample in collection.training_samples
                        ]
                    ),
                    observed_class_labels=torch.cat(
                        [
                            training_sample.observed_class_labels
                            for training_sample in collection.training_samples
                        ]
                    ),
                )
                for training_binding, collection in zip(
                    training_bindings, model_training_sample_collections
                )
            )
            expected_joint_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            actual_joint_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=training_bindings[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
                participating_training_batches=participating_training_batches,
                update_shared_features=update_shared_features,
            )
            assert actual_joint_loss == expected_joint_loss
            for training_binding, training_batch in zip(
                training_bindings, participating_training_batches
            ):
                counts_store.record_completed_model_training(
                    model_id=training_binding.model_id,
                    trained_sample_count=len(training_batch.input_features),
                    parameter_update_step_count=1,
                )
            assert_held_model_states_match_legacy(
                registry=registry,
                shared_optimizer_owners=shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=change_interval_input_features,
            )

        # 学習でparameterが変わった後のモデルで、区間を評価して解決する。
        run_joint_update_in_both_implementations()
        initial_training_model_id = legacy_client.current_model_id
        (
            legacy_drift_type,
            legacy_adaptation_events,
            legacy_epoch_training_calls,
        ) = resolve_alarm_change_interval_in_legacy_client(
            legacy_client=legacy_client,
            resolution_arguments=resolution_arguments,
            monkeypatch=monkeypatch,
        )
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        alarm_change_interval_resolution = resolve_alarm_change_interval(**resolution_arguments)
        assert (
            alarm_change_interval_resolution.resolution_outcome
            == ALARM_INTERVAL_RESOLUTION_CASES[alarm_interval_resolution_case][
                "expected_resolution_outcome"
            ]
        )
        assert_alarm_change_interval_resolution_matches_legacy(
            alarm_change_interval_resolution=alarm_change_interval_resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            legacy_adaptation_events=legacy_adaptation_events,
            legacy_epoch_training_calls=legacy_epoch_training_calls,
            initial_training_model_id=initial_training_model_id,
        )
        # 吸収した区間の標本を含むbatchで学習を継続する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert_training_samples_match_legacy(
            training_sample_store=training_sample_store, legacy_client=legacy_client
        )
        assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
        assert_store_statistics_match_legacy(
            loss_statistics_store=resolution_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
        )
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
def test_alarm_change_interval_resolution_observes_samples_after_started_validation_like_legacy(
    class_count, optimizer_variant, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(847)
        (
            resolution_arguments,
            _,
            legacy_client,
            _,
        ) = build_alarm_change_interval_resolution_oracle(
            class_count=class_count,
            alarm_interval_resolution_case="no_model_fits",
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        initial_rng_state = torch.get_rng_state().clone()
        resolve_alarm_change_interval_in_legacy_client(
            legacy_client=legacy_client,
            resolution_arguments=resolution_arguments,
            monkeypatch=monkeypatch,
        )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng_state)
        started_validation_session = resolve_alarm_change_interval(
            **resolution_arguments
        ).started_validation_session
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        legacy_session = legacy_client._forward_validation
        # 実観測。旧は損失評価と収集だけを呼び、到達時の採否確定は起動しない。
        validation_samples = tuple(
            (
                started_validation_session.proposal_sample_index + 1 + sample_offset,
                started_validation_session.training_input_features[sample_offset].reshape(1, -1)
                * 0.5
                + 0.1 * sample_offset,
                started_validation_session.training_observed_class_labels[sample_offset].reshape(
                    1, 1
                ),
            )
            for sample_offset in range(legacy_session.target_count)
        )
        for sample_offset, (sample_index, input_features, observed_class_labels) in enumerate(
            validation_samples
        ):
            with torch.no_grad():
                legacy_session.append_losses(
                    legacy_session.candidate.per_sample_error(input_features, observed_class_labels)
                    .reshape(-1)[0]
                    .item(),
                    {
                        model_id: legacy_reference_model.per_sample_error(
                            input_features, observed_class_labels
                        )
                        .reshape(-1)[0]
                        .item()
                        for model_id, legacy_reference_model in legacy_session.reference_models.items()
                    },
                )
            assert observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                candidate_classifier=started_validation_session.candidate_training_state.candidate_classifier,
                reference_classifiers_by_model_id=started_validation_session.fixed_reference_models.reference_classifiers_by_model_id,
                post_alarm_candidate_loss_collection=started_validation_session.post_alarm_candidate_loss_collection,
            ) is (sample_offset == legacy_session.target_count - 1)
            assert_collected_losses_match_legacy(
                post_alarm_candidate_loss_collection=started_validation_session.post_alarm_candidate_loss_collection,
                legacy_session=legacy_session,
            )
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
