"""警報後の候補検証の確定を、実旧の前向き検証確定処理の4分岐へ対照する。"""

import random
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
)
from test_adopted_candidate_local_adoption import (
    assert_adoption_state_unchanged,
    assert_local_adoption_matches_legacy,
    assert_training_samples_match_legacy,
    build_local_adoption_oracle,
    snapshot_adoption_state,
)
from test_joint_model_parameter_update import run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy

import federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution as resolution_module
from federated_drift_experiment import config
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    PostAlarmCandidateLossEvaluation,
    evaluate_candidate_using_post_alarm_losses,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES,
    PostAlarmCandidateValidationResolution,
    apply_post_alarm_candidate_validation_resolution,
)

# 新の結果種別と旧の操作種別の対応。旧名を新実装へ持ち込まず、testで明示する。
LEGACY_ACTION_BY_RESOLUTION_OUTCOME = {
    "candidate_adopted_as_new_model": "create",
    "held_reference_model_reused": "reuse",
    "current_model_maintained": "maintain",
    "candidate_rejected": "create_rejected",
}
LEGACY_DRIFT_TYPE_BY_RESOLUTION_OUTCOME = {
    "candidate_adopted_as_new_model": 2,
    "held_reference_model_reused": 1,
    "current_model_maintained": 0,
    "candidate_rejected": 0,
}
LEGACY_RESOLUTION_CASES = ("create", "reuse", "maintain", "create_rejected")


class IntSubclass(int):
    """IDとして受理しない派生型。"""


def build_resolution_oracle(
    *,
    class_count,
    legacy_resolution_case,
    monkeypatch,
    pending_sample_count=3,
    optimizer_variant="standard",
):
    """上流の採用oracleへ旧分岐を決める損失列を設定し、同じ入力から新の評価結果を作る。"""
    adoption_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
        build_local_adoption_oracle(
            class_count=class_count,
            pending_sample_count=pending_sample_count,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
    )
    legacy_session = legacy_client._forward_validation
    if legacy_resolution_case == "create":
        candidate_losses = (0.01,) * 4
        reference_losses_by_model_id = {4: (0.9,) * 4, 9: (0.9,) * 4}
        reference_historical_mean_losses_by_model_id = {}
    else:
        candidate_losses = (0.9,) * 4
        reference_losses_by_model_id = {4: (0.2,) * 4, 9: (0.3,) * 4}
        reference_historical_mean_losses_by_model_id = {
            "reuse": {4: 0.2},
            "maintain": {9: 0.3},
            "create_rejected": {},
        }[legacy_resolution_case]
    legacy_session.candidate_losses = list(candidate_losses)
    legacy_session.reference_losses = {
        model_id: list(losses) for model_id, losses in reference_losses_by_model_id.items()
    }
    legacy_session.reference_historical_means = dict(reference_historical_mean_losses_by_model_id)
    registry = adoption_arguments["held_model_training_state_registry"]
    # 評価は移植済みの関数へ同じ損失列・履歴平均・閾値を与えるtest-only接続で得る。
    evaluation_arguments = dict(
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy=(
                "current_model_first_reuse_then_two_segment_candidate_validation"
            ),
            candidate_post_alarm_validation_sample_count=4,
        ),
        candidate_losses=candidate_losses,
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ),
        current_training_model_id=adoption_arguments[
            "current_training_model_assignment"
        ].current_training_model_id,
        maximum_reference_mean_loss_increase=legacy_client.distance_threshold,
        minimum_candidate_mean_loss_improvement=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
    )
    resolution_arguments = dict(
        adoption_arguments,
        post_alarm_candidate_loss_evaluation=evaluate_candidate_using_post_alarm_losses(
            **evaluation_arguments
        ),
        # 旧の保留標本は全て真の概念1を持つ（build_local_adoption_oracle）。
        pending_assignment_sample_concept_ids=(1,) * pending_sample_count,
    )
    return resolution_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model


def finalize_forward_validation_in_legacy_client(*, legacy_client, resolution_arguments):
    """実旧の確定処理を実行し、旧判定が新の評価結果と同じであることを確かめる。"""
    legacy_drift_type = legacy_client._finalize_forward_validation(57)
    assert (
        legacy_client.provisional_model_decisions[-1].accepted
        is resolution_arguments["post_alarm_candidate_loss_evaluation"].candidate_accepted
    )
    assert legacy_client._forward_validation is None
    return legacy_drift_type


def assert_resolution_matches_legacy(
    *,
    resolution,
    resolution_arguments,
    shared_optimizer_owners,
    legacy_client,
    legacy_drift_type,
    previous_model_id,
):
    assert type(resolution) is PostAlarmCandidateValidationResolution
    assert resolution.resolution_outcome in POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES
    assert (
        LEGACY_ACTION_BY_RESOLUTION_OUTCOME[resolution.resolution_outcome]
        == legacy_client.adaptation_events[-1].action
    )
    assert (
        LEGACY_DRIFT_TYPE_BY_RESOLUTION_OUTCOME[resolution.resolution_outcome] == legacy_drift_type
    )
    assert resolution.assigned_model_id == legacy_client.current_model_id
    current_training_model_assignment = resolution_arguments["current_training_model_assignment"]
    assert (
        current_training_model_assignment.current_training_model_id
        == legacy_client.current_model_id
    )
    assignment_change = resolution.training_model_assignment_change
    if legacy_client.current_model_id == previous_model_id:
        # 旧は現在IDが変わらなければ変更通知を呼ばず、切替位置も記録しない。
        assert assignment_change is None
        assert legacy_client.local_model_changes == []
        assert legacy_client.local_switch_positions == []
    else:
        assert type(assignment_change) is TrainingModelAssignmentChange
        assert legacy_client.local_model_changes == [
            (assignment_change.previous_model_id, assignment_change.current_model_id)
        ]
        assert assignment_change.previous_model_id == previous_model_id
        assert assignment_change.current_model_id == resolution.assigned_model_id
        assert legacy_client.local_switch_positions == [57]
    assert (
        resolution_arguments["temporary_model_id_allocator"].next_temporary_model_id
        == legacy_client.next_temp_id
    )
    registry = resolution_arguments["held_model_training_state_registry"]
    if resolution.resolution_outcome == "candidate_adopted_as_new_model":
        assert_local_adoption_matches_legacy(
            adoption_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            assignment_change=assignment_change,
        )
    else:
        # 採用以外では候補は登録されず、共有部も送信保留も変わらない。
        assert_held_model_states_match_legacy(
            registry=registry,
            shared_optimizer_owners=shared_optimizer_owners[:-1],
            legacy_client=legacy_client,
            input_features=resolution_arguments["initial_statistics_input_features"],
        )
        assert_model_counts_match_legacy(
            counts_store=resolution_arguments["model_training_and_assignment_counts_store"],
            legacy_client=legacy_client,
        )
        assert_training_samples_match_legacy(
            training_sample_store=resolution_arguments["training_sample_store"],
            legacy_client=legacy_client,
        )
        assert_store_statistics_match_legacy(
            loss_statistics_store=resolution_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
        )
        assert resolution_arguments["pending_model_upload_state"].get_pending_model_upload() is None
        assert legacy_client.pending_model_params is None


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("legacy_resolution_case", LEGACY_RESOLUTION_CASES)
@pytest.mark.parametrize("pending_sample_count", [0, 4])
def test_resolution_matches_actual_legacy_forward_validation_branches(
    class_count, legacy_resolution_case, pending_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        resolution_arguments, shared_optimizer_owners, legacy_client, _ = build_resolution_oracle(
            class_count=class_count,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=pending_sample_count,
            monkeypatch=monkeypatch,
        )
        registry = resolution_arguments["held_model_training_state_registry"]
        previous_model_id = resolution_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        previous_held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        candidate = resolution_arguments["adopted_candidate_classifier"]
        candidate_feature_extractor = candidate.feature_extractor
        candidate_optimizer = resolution_arguments[
            "candidate_concept_specific_parameter_optimizer_state"
        ].parameter_optimizer
        previous_counts = resolution_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())

        legacy_drift_type = finalize_forward_validation_in_legacy_client(
            legacy_client=legacy_client, resolution_arguments=resolution_arguments
        )
        resolution = apply_post_alarm_candidate_validation_resolution(**resolution_arguments)

        assert (
            LEGACY_ACTION_BY_RESOLUTION_OUTCOME[resolution.resolution_outcome]
            == legacy_resolution_case
        )
        assert_resolution_matches_legacy(
            resolution=resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
        )
        current_counts = resolution_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        if legacy_resolution_case == "create":
            # 採用では保留標本の概念を計数しない（実旧と同じ）。
            assert (
                current_counts.assigned_sample_counts_by_model_and_concept_id
                == previous_counts.assigned_sample_counts_by_model_and_concept_id
            )
        else:
            # 採用以外では候補・保有一覧の構成・学習計数・採番を変更しない。
            assert (
                registry.snapshot_ordered_held_model_training_states()
                == previous_held_model_training_states
            )
            assert candidate.feature_extractor is candidate_feature_extractor
            assert (
                resolution_arguments[
                    "candidate_concept_specific_parameter_optimizer_state"
                ].parameter_optimizer
                is candidate_optimizer
            )
            assert (
                current_counts.trained_sample_counts_by_model_id
                == previous_counts.trained_sample_counts_by_model_id
            )
            assert (
                current_counts.parameter_update_step_counts_by_model_id
                == previous_counts.parameter_update_step_counts_by_model_id
            )
            assert (
                sum(
                    current_counts.assigned_sample_counts_by_model_and_concept_id.get(
                        resolution.assigned_model_id, {}
                    ).values()
                )
                == sum(
                    previous_counts.assigned_sample_counts_by_model_and_concept_id.get(
                        resolution.assigned_model_id, {}
                    ).values()
                )
                + pending_sample_count
            )
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]


def test_resolution_record_is_immutable_and_validates_outcome():
    resolution = PostAlarmCandidateValidationResolution(
        resolution_outcome="candidate_rejected",
        assigned_model_id=9,
        training_model_assignment_change=None,
    )
    with pytest.raises(FrozenInstanceError):
        resolution.assigned_model_id = 4
    with pytest.raises(ValueError):
        PostAlarmCandidateValidationResolution(
            resolution_outcome="create",
            assigned_model_id=9,
            training_model_assignment_change=None,
        )
    assert POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES == (
        "candidate_adopted_as_new_model",
        "held_reference_model_reused",
        "current_model_maintained",
        "candidate_rejected",
    )


@pytest.mark.parametrize("legacy_resolution_case", LEGACY_RESOLUTION_CASES)
@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("evaluation_none", TypeError),
        ("evaluation_dict", TypeError),
        ("evaluation_subclass", TypeError),
        ("evaluation_accepted_not_bool", TypeError),
        ("evaluation_reusable_id_bool", TypeError),
        ("evaluation_reusable_id_subclass", TypeError),
        ("evaluation_accepted_with_reusable_reference", ValueError),
        ("current_assignment_none", TypeError),
        ("current_assignment_subclass", TypeError),
        ("pending_samples_list", TypeError),
        ("concept_ids_list", TypeError),
        ("concept_ids_none", TypeError),
        ("concept_ids_shorter", ValueError),
        ("concept_id_bool", TypeError),
        ("concept_id_float", TypeError),
    ],
)
def test_invalid_resolution_inputs_change_no_state_for_any_outcome(
    legacy_resolution_case, invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        resolution_arguments, shared_optimizer_owners, _, _ = build_resolution_oracle(
            class_count=4, legacy_resolution_case=legacy_resolution_case, monkeypatch=monkeypatch
        )
        valid_resolution_arguments = dict(resolution_arguments)
        evaluation = resolution_arguments["post_alarm_candidate_loss_evaluation"]
        if invalid_case == "evaluation_none":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = None
        elif invalid_case == "evaluation_dict":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = dict(
                candidate_accepted=False, reusable_reference_model_id=None
            )
        elif invalid_case == "evaluation_subclass":
            evaluation_subclass = type(
                "EvaluationSubclass", (PostAlarmCandidateLossEvaluation,), {}
            )
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = evaluation_subclass(
                **{
                    field_name: getattr(evaluation, field_name)
                    for field_name in evaluation.__dataclass_fields__
                }
            )
        elif invalid_case == "evaluation_accepted_not_bool":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = replace(
                evaluation, candidate_accepted=1
            )
        elif invalid_case == "evaluation_reusable_id_bool":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = replace(
                evaluation, candidate_accepted=False, reusable_reference_model_id=True
            )
        elif invalid_case == "evaluation_reusable_id_subclass":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = replace(
                evaluation, candidate_accepted=False, reusable_reference_model_id=IntSubclass(4)
            )
        elif invalid_case == "evaluation_accepted_with_reusable_reference":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = replace(
                evaluation, candidate_accepted=True, reusable_reference_model_id=4
            )
        elif invalid_case == "current_assignment_none":
            resolution_arguments["current_training_model_assignment"] = None
        elif invalid_case == "current_assignment_subclass":
            resolution_arguments["current_training_model_assignment"] = type(
                "OwnerSubclass", (CurrentTrainingModelAssignment,), {}
            )(initial_model_id=9)
        elif invalid_case == "pending_samples_list":
            resolution_arguments["pending_assignment_training_samples"] = list(
                resolution_arguments["pending_assignment_training_samples"]
            )
        elif invalid_case == "concept_ids_list":
            resolution_arguments["pending_assignment_sample_concept_ids"] = [1, 1, 1]
        elif invalid_case == "concept_ids_none":
            resolution_arguments["pending_assignment_sample_concept_ids"] = None
        elif invalid_case == "concept_ids_shorter":
            resolution_arguments["pending_assignment_sample_concept_ids"] = (1, 1)
        elif invalid_case == "concept_id_bool":
            resolution_arguments["pending_assignment_sample_concept_ids"] = (1, True, 1)
        elif invalid_case == "concept_id_float":
            resolution_arguments["pending_assignment_sample_concept_ids"] = (1, 1.0, None)
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=valid_resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with pytest.raises(expected_exception):
            apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
        assert_adoption_state_unchanged(
            previous_snapshot=previous_snapshot, adoption_arguments=valid_resolution_arguments
        )


@pytest.mark.parametrize(
    ("legacy_resolution_case", "invalid_case", "expected_exception"),
    [
        ("create", "upload_delay_zero", ValueError),
        ("create", "trained_sample_count_negative", ValueError),
        ("create", "candidate_object", (TypeError, ValueError)),
        ("create", "labels_out_of_range", ValueError),
        ("create", "temporary_id_used_in_training_sample_store", ValueError),
        ("create", "allocator_object", TypeError),
        ("reuse", "reusable_model_not_held", KeyError),
        ("reuse", "pending_sample_with_two_rows", ValueError),
        ("reuse", "pending_sample_label_out_of_range", ValueError),
        ("reuse", "registry_object", TypeError),
        ("maintain", "pending_sample_feature_count", ValueError),
        ("maintain", "statistics_store_none", TypeError),
        ("create_rejected", "pending_sample_feature_count", ValueError),
        ("create_rejected", "pending_sample_element_tuple", TypeError),
        ("create_rejected", "counts_store_object", TypeError),
    ],
)
def test_rejection_by_selected_operation_changes_no_state_including_current_assignment(
    legacy_resolution_case, invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        resolution_arguments, shared_optimizer_owners, _, _ = build_resolution_oracle(
            class_count=4, legacy_resolution_case=legacy_resolution_case, monkeypatch=monkeypatch
        )
        valid_resolution_arguments = dict(resolution_arguments)
        pending_assignment_training_samples = resolution_arguments[
            "pending_assignment_training_samples"
        ]

        def replace_middle_pending_sample(*, input_features=None, observed_class_labels=None):
            training_sample = pending_assignment_training_samples[1]
            resolution_arguments["pending_assignment_training_samples"] = (
                pending_assignment_training_samples[0],
                ObservedTrainingSample(
                    input_features=training_sample.input_features
                    if input_features is None
                    else input_features,
                    observed_class_labels=training_sample.observed_class_labels
                    if observed_class_labels is None
                    else observed_class_labels,
                ),
                pending_assignment_training_samples[2],
            )

        if invalid_case == "upload_delay_zero":
            resolution_arguments["upload_delay_round_count"] = 0
        elif invalid_case == "trained_sample_count_negative":
            resolution_arguments["candidate_trained_sample_count"] = -1
        elif invalid_case == "candidate_object":
            resolution_arguments["adopted_candidate_classifier"] = object()
        elif invalid_case == "labels_out_of_range":
            resolution_arguments["initial_statistics_observed_class_labels"][0, 0] = 4.0
        elif invalid_case == "temporary_id_used_in_training_sample_store":
            resolution_arguments["training_sample_store"].append_model_training_samples(
                model_id=resolution_arguments[
                    "temporary_model_id_allocator"
                ].next_temporary_model_id,
                training_samples=(),
            )
        elif invalid_case == "allocator_object":
            resolution_arguments["temporary_model_id_allocator"] = object()
        elif invalid_case == "reusable_model_not_held":
            resolution_arguments["post_alarm_candidate_loss_evaluation"] = replace(
                resolution_arguments["post_alarm_candidate_loss_evaluation"],
                reusable_reference_model_id=77,
            )
        elif invalid_case == "pending_sample_with_two_rows":
            replace_middle_pending_sample(
                input_features=torch.cat(
                    [pending_assignment_training_samples[1].input_features] * 2
                ),
                observed_class_labels=torch.cat(
                    [pending_assignment_training_samples[1].observed_class_labels] * 2
                ),
            )
        elif invalid_case == "pending_sample_label_out_of_range":
            replace_middle_pending_sample(observed_class_labels=torch.tensor([[4.0]]))
        elif invalid_case == "pending_sample_feature_count":
            replace_middle_pending_sample(input_features=torch.ones((1, 3)))
        elif invalid_case == "pending_sample_element_tuple":
            resolution_arguments["pending_assignment_training_samples"] = (
                pending_assignment_training_samples[0],
                (
                    pending_assignment_training_samples[1].input_features,
                    pending_assignment_training_samples[1].observed_class_labels,
                ),
                pending_assignment_training_samples[2],
            )
        elif invalid_case == "registry_object":
            resolution_arguments["held_model_training_state_registry"] = object()
        elif invalid_case == "statistics_store_none":
            resolution_arguments["loss_statistics_store"] = None
        elif invalid_case == "counts_store_object":
            resolution_arguments["model_training_and_assignment_counts_store"] = object()
        previous_snapshot = snapshot_adoption_state(
            adoption_arguments=valid_resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with pytest.raises(expected_exception):
            apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
        assert_adoption_state_unchanged(
            previous_snapshot=previous_snapshot, adoption_arguments=valid_resolution_arguments
        )


@pytest.mark.parametrize("legacy_resolution_case", LEGACY_RESOLUTION_CASES)
def test_each_outcome_calls_only_its_operations_in_order(legacy_resolution_case, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        resolution_arguments, _, _, _ = build_resolution_oracle(
            class_count=2, legacy_resolution_case=legacy_resolution_case, monkeypatch=monkeypatch
        )
        actual_call_order = []
        for operation_name in (
            "adopt_candidate_as_current_training_model",
            "absorb_assigned_training_samples_into_held_model",
        ):
            original_operation = getattr(resolution_module, operation_name)

            def record_then_call(
                *, operation_name=operation_name, original_operation=original_operation, **arguments
            ):
                actual_call_order.append((operation_name, arguments.get("model_id")))
                return original_operation(**arguments)

            monkeypatch.setattr(resolution_module, operation_name, record_then_call)
        current_training_model_assignment = resolution_arguments[
            "current_training_model_assignment"
        ]
        original_assignment = current_training_model_assignment.assign_model_for_training

        def record_then_assign(**arguments):
            # 採用の組立が内部で行う切替えは、採用の呼出しの一部として数えない。
            if actual_call_order[-1][0] != "adopt_candidate_as_current_training_model":
                actual_call_order.append(("assign_model_for_training", arguments["model_id"]))
            return original_assignment(**arguments)

        monkeypatch.setattr(
            current_training_model_assignment, "assign_model_for_training", record_then_assign
        )
        apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
        expected_call_order = {
            "create": [("adopt_candidate_as_current_training_model", None)],
            "reuse": [
                ("absorb_assigned_training_samples_into_held_model", 4),
                ("assign_model_for_training", 4),
            ],
            # 維持は切替え先が現在IDと同じで、ownerは変更なし（None）を返す。
            "maintain": [
                ("absorb_assigned_training_samples_into_held_model", 9),
                ("assign_model_for_training", 9),
            ],
            "create_rejected": [("absorb_assigned_training_samples_into_held_model", 9)],
        }[legacy_resolution_case]
        assert actual_call_order == expected_call_order


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
@pytest.mark.parametrize("legacy_resolution_case", ["create", "reuse"])
def test_resolved_validation_continues_actual_joint_training(
    class_count, optimizer_variant, update_shared_features, legacy_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(607)
        resolution_arguments, shared_optimizer_owners, legacy_client, _ = build_resolution_oracle(
            class_count=class_count,
            legacy_resolution_case=legacy_resolution_case,
            pending_sample_count=4,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = resolution_arguments["held_model_training_state_registry"]
        training_sample_store = resolution_arguments["training_sample_store"]
        counts_store = resolution_arguments["model_training_and_assignment_counts_store"]
        active = registry.get_held_model_training_state(model_id=9).classifier.feature_extractor
        input_features = resolution_arguments["initial_statistics_input_features"]
        previous_model_id = resolution_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        legacy_training_batches = []
        legacy_client.updates_per_sample = 1
        legacy_client.backbone_gradient_diagnostics = defaultdict(float)
        legacy_client.phase_seconds = defaultdict(float)
        legacy_client._sample_training_batches = lambda: legacy_training_batches

        def run_joint_update_in_both_implementations():
            # 両実装とも、各自の標本storeにある全標本をモデルごとの固定batchにする。
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            model_training_sample_collections = (
                training_sample_store.snapshot_ordered_model_training_samples()
            )
            assert tuple(binding.model_id for binding in training_bindings) == tuple(
                collection.model_id for collection in model_training_sample_collections
            )
            legacy_training_batches[:] = [
                (
                    model_id,
                    torch.cat([legacy_training_sample[0] for legacy_training_sample in samples]),
                    torch.cat([legacy_training_sample[1] for legacy_training_sample in samples]),
                )
                for model_id, samples in legacy_client.train_data_store.items()
            ]
            participating_training_batches = tuple(
                ParticipatingModelTrainingBatch(
                    classifier=training_binding.classifier,
                    concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
                    input_features=torch.cat(
                        [sample.input_features for sample in collection.training_samples]
                    ),
                    observed_class_labels=torch.cat(
                        [sample.observed_class_labels for sample in collection.training_samples]
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
                shared_feature_extractor=active,
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
                shared_optimizer_owners=shared_optimizer_owners
                if len(training_bindings) == 3
                else shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=input_features,
            )
            assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
            assert_training_samples_match_legacy(
                training_sample_store=training_sample_store, legacy_client=legacy_client
            )
            assert_store_statistics_match_legacy(
                loss_statistics_store=resolution_arguments["loss_statistics_store"],
                legacy_client=legacy_client,
            )

        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        # 確定前: 保有2モデルを学習し、parameterが変わった状態で確定する。
        run_joint_update_in_both_implementations()
        legacy_drift_type = finalize_forward_validation_in_legacy_client(
            legacy_client=legacy_client, resolution_arguments=resolution_arguments
        )
        resolution = apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
        assert (
            LEGACY_ACTION_BY_RESOLUTION_OUTCOME[resolution.resolution_outcome]
            == legacy_resolution_case
        )
        assert_resolution_matches_legacy(
            resolution=resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
        )
        # 確定後: 採用した新モデル、または保留標本を吸収した再利用先を含めて学習を継続する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
