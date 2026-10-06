"""採用候補のローカル採用を、実旧の前向き検証確定処理の採用分岐へ対照する。"""

import random
from collections import Counter, defaultdict

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
    assert_initial_registration_matches_legacy,
    assert_registration_state_unchanged,
    build_initial_registration_oracle,
    snapshot_registration_state,
)
from test_joint_model_parameter_update import run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy

import federated_learning_experiments.runtime.adopted_candidate_local_adoption as adoption_module
from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.detection_episode import DetectionEpisodeController
from federated_drift_experiment.provisional_model import ForwardValidationSession
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
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
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.runtime.adopted_candidate_local_adoption import (
    adopt_candidate_as_current_training_model,
)
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)


class IntSubclass(int):
    """計数として受理しない派生型。"""


def build_local_adoption_oracle(
    *,
    class_count,
    monkeypatch,
    held_model_ids=(4, 9),
    current_model_is_held=True,
    pending_sample_count=3,
    candidate_training_counts=(11, 3),
    optimizer_variant="standard",
    existing_pending_model_id=None,
):
    """上流の登録oracleへ、採番・標本・計数ownerと実旧の前向き検証sessionを加える。"""
    registration_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
        build_initial_registration_oracle(
            class_count=class_count,
            held_model_ids=held_model_ids,
            current_model_is_held=current_model_is_held,
            optimizer_variant=optimizer_variant,
            existing_pending_model_id=existing_pending_model_id,
            monkeypatch=monkeypatch,
        )
    )
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_persistent")
    temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=3)
    training_sample_store = ModelTrainingSampleStore()
    counts_store = ModelTrainingAndAssignmentCountsStore()
    legacy_client.client_id = 3
    # 初期値の一致は採番specで実旧__init__と対照済み。ここでは同じ値から始める。
    legacy_client.next_temp_id = temporary_model_id_allocator.next_temporary_model_id
    legacy_client.train_data_store = defaultdict(list)
    legacy_client.model_training_examples = defaultdict(int)
    legacy_client.model_optimizer_steps = defaultdict(int)
    legacy_client.model_concept_counts = defaultdict(Counter)
    input_features = registration_arguments["initial_statistics_input_features"]
    observed_class_labels = registration_arguments["initial_statistics_observed_class_labels"]
    # 既存モデルの標本・学習計数・割当概念計数。採用で変わらないことを観測する。
    for model_index, model_id in enumerate(held_model_ids):
        training_samples = tuple(
            ObservedTrainingSample(
                input_features=input_features[sample_index : sample_index + 1] + model_index,
                observed_class_labels=observed_class_labels[sample_index : sample_index + 1],
            )
            for sample_index in range(3)
        )
        training_sample_store.append_model_training_samples(
            model_id=model_id, training_samples=training_samples
        )
        legacy_client.train_data_store[model_id].extend(
            (training_sample.input_features, training_sample.observed_class_labels, 0)
            for training_sample in training_samples
        )
        counts_store.record_completed_model_training(
            model_id=model_id, trained_sample_count=5 + model_index, parameter_update_step_count=2
        )
        legacy_client.model_training_examples[model_id] += 5 + model_index
        legacy_client.model_optimizer_steps[model_id] += 2
        counts_store.record_assigned_sample_concept(model_id=model_id, observed_concept_id=0)
        legacy_client.model_concept_counts[model_id][0] += 1
    pending_assignment_training_samples = tuple(
        ObservedTrainingSample(
            input_features=input_features[sample_index % len(input_features)].reshape(1, -1) - 0.5,
            observed_class_labels=observed_class_labels[
                sample_index % len(observed_class_labels)
            ].reshape(1, 1),
        )
        for sample_index in range(pending_sample_count)
    )
    legacy_session = ForwardValidationSession(
        proposal_position=40,
        estimated_change_point=35,
        episode_id=None,
        old_model_id=legacy_client.current_model_id,
        detector="class_esr",
        candidate=legacy_candidate_model,
        training_x=input_features,
        training_y=observed_class_labels,
        # 旧の保留標本は(特徴, ラベル, 真の概念)。採用分岐は概念を読まない。
        held_data=[
            (training_sample.input_features, training_sample.observed_class_labels, 1)
            for training_sample in pending_assignment_training_samples
        ],
        reference_models=dict(legacy_client.models),
        target_count=4,
        candidate_training_examples=candidate_training_counts[0],
        candidate_optimizer_steps=candidate_training_counts[1],
    )
    # 候補の損失を十分小さくし、実旧の判定を採用にする。
    for _ in range(4):
        legacy_session.append_losses(0.01, dict.fromkeys(legacy_client.models, 0.9))
    legacy_client._forward_validation = legacy_session
    legacy_client.distance_threshold = 0.1
    legacy_client.provisional_model_decisions = []
    legacy_client.local_switch_positions = []
    legacy_client.adaptation_events = []
    legacy_client.model_upload_delay_rounds = registration_arguments["upload_delay_round_count"]
    legacy_client.detection_episodes = DetectionEpisodeController(enabled=False, length=10)
    legacy_client.local_model_changes = []
    legacy_client._on_local_model_change = lambda previous_model_id, current_model_id: (
        legacy_client.local_model_changes.append((previous_model_id, current_model_id))
    )
    adoption_arguments = {
        argument_name: argument_value
        for argument_name, argument_value in registration_arguments.items()
        if argument_name != "temporary_model_id"
    }
    adoption_arguments.update(
        temporary_model_id_allocator=temporary_model_id_allocator,
        candidate_trained_sample_count=candidate_training_counts[0],
        candidate_parameter_update_step_count=candidate_training_counts[1],
        pending_assignment_training_samples=pending_assignment_training_samples,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=counts_store,
    )
    return adoption_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model


def finalize_adoption_in_legacy_client(*, legacy_client):
    """実旧の確定処理を実行する。採用と判定されたことだけを前提として確認する。"""
    assert legacy_client._finalize_forward_validation(57) == 2
    assert legacy_client.provisional_model_decisions[-1].accepted
    assert legacy_client._forward_validation is None


def assert_training_samples_match_legacy(*, training_sample_store, legacy_client):
    model_training_sample_collections = (
        training_sample_store.snapshot_ordered_model_training_samples()
    )
    assert tuple(collection.model_id for collection in model_training_sample_collections) == tuple(
        legacy_client.train_data_store
    )
    for collection in model_training_sample_collections:
        legacy_training_samples = legacy_client.train_data_store[collection.model_id]
        assert len(collection.training_samples) == len(legacy_training_samples)
        for training_sample, legacy_training_sample in zip(
            collection.training_samples, legacy_training_samples
        ):
            assert training_sample.input_features is legacy_training_sample[0]
            assert training_sample.observed_class_labels is legacy_training_sample[1]


def assert_local_adoption_matches_legacy(
    *, adoption_arguments, shared_optimizer_owners, legacy_client, assignment_change
):
    temporary_model_id = legacy_client.current_model_id
    assert temporary_model_id < 0
    assert type(assignment_change) is TrainingModelAssignmentChange
    assert legacy_client.local_model_changes[-1] == (
        assignment_change.previous_model_id,
        assignment_change.current_model_id,
    )
    assert assignment_change.current_model_id == temporary_model_id
    assert (
        adoption_arguments["current_training_model_assignment"].current_training_model_id
        == temporary_model_id
    )
    assert (
        adoption_arguments["temporary_model_id_allocator"].next_temporary_model_id
        == legacy_client.next_temp_id
    )
    assert_model_counts_match_legacy(
        counts_store=adoption_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert_training_samples_match_legacy(
        training_sample_store=adoption_arguments["training_sample_store"],
        legacy_client=legacy_client,
    )
    # 一覧・共有部と全モデルの値・統計・送信保留と待機は上流の登録対照を使う。
    assert_initial_registration_matches_legacy(
        registration_arguments={**adoption_arguments, "temporary_model_id": temporary_model_id},
        shared_optimizer_owners=shared_optimizer_owners,
        legacy_client=legacy_client,
    )


def snapshot_adoption_state(*, adoption_arguments, shared_optimizer_owners):
    training_sample_store = adoption_arguments["training_sample_store"]
    counts_store = adoption_arguments["model_training_and_assignment_counts_store"]
    return dict(
        registration_state=snapshot_registration_state(
            registration_arguments=adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        ),
        next_temporary_model_id=adoption_arguments[
            "temporary_model_id_allocator"
        ].next_temporary_model_id,
        training_samples=tuple(
            (collection.model_id, collection.training_samples)
            for collection in training_sample_store.snapshot_ordered_model_training_samples()
        ),
        counts=counts_store.snapshot_model_training_and_assignment_counts(),
    )


def assert_adoption_state_unchanged(*, previous_snapshot, adoption_arguments):
    assert_registration_state_unchanged(
        previous_snapshot=previous_snapshot["registration_state"],
        registry=adoption_arguments["held_model_training_state_registry"],
        loss_statistics_store=adoption_arguments["loss_statistics_store"],
        pending_upload_state=adoption_arguments["pending_model_upload_state"],
        current_training_model_assignment=adoption_arguments["current_training_model_assignment"],
    )
    assert (
        adoption_arguments["temporary_model_id_allocator"].next_temporary_model_id
        == previous_snapshot["next_temporary_model_id"]
    )
    model_training_sample_collections = adoption_arguments[
        "training_sample_store"
    ].snapshot_ordered_model_training_samples()
    assert len(model_training_sample_collections) == len(previous_snapshot["training_samples"])
    for collection, (model_id, training_samples) in zip(
        model_training_sample_collections, previous_snapshot["training_samples"]
    ):
        assert collection.model_id == model_id
        assert len(collection.training_samples) == len(training_samples)
        assert all(
            training_sample is previous_training_sample
            for training_sample, previous_training_sample in zip(
                collection.training_samples, training_samples
            )
        )
    assert (
        adoption_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        == previous_snapshot["counts"]
    )


def assert_adoption_rejected_without_any_change(
    *, adoption_arguments, valid_adoption_arguments, shared_optimizer_owners, expected_exception
):
    previous_snapshot = snapshot_adoption_state(
        adoption_arguments=valid_adoption_arguments,
        shared_optimizer_owners=shared_optimizer_owners,
    )
    with pytest.raises(expected_exception):
        adopt_candidate_as_current_training_model(**adoption_arguments)
    assert_adoption_state_unchanged(
        previous_snapshot=previous_snapshot, adoption_arguments=valid_adoption_arguments
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("held_model_ids", [(4,), (9, -3, 4)])
@pytest.mark.parametrize("pending_sample_count", [0, 1, 4])
@pytest.mark.parametrize("candidate_training_counts", [(0, 0), (11, 3)])
@pytest.mark.parametrize("existing_pending_model_id", [None, 9])
def test_local_adoption_matches_actual_legacy_forward_validation_acceptance(
    class_count,
    held_model_ids,
    pending_sample_count,
    candidate_training_counts,
    existing_pending_model_id,
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(401)
        adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
            class_count=class_count,
            held_model_ids=held_model_ids,
            pending_sample_count=pending_sample_count,
            candidate_training_counts=candidate_training_counts,
            existing_pending_model_id=existing_pending_model_id,
            monkeypatch=monkeypatch,
        )
        previous_model_id = adoption_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        expected_temporary_model_id = adoption_arguments[
            "temporary_model_id_allocator"
        ].next_temporary_model_id
        loss_statistics_store = adoption_arguments["loss_statistics_store"]
        previous_loss_statistics = loss_statistics_store.get_state_snapshot()
        counts_store = adoption_arguments["model_training_and_assignment_counts_store"]
        previous_counts = counts_store.snapshot_model_training_and_assignment_counts()
        previous_training_samples = adoption_arguments[
            "training_sample_store"
        ].snapshot_ordered_model_training_samples()
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())

        finalize_adoption_in_legacy_client(legacy_client=legacy_client)
        assignment_change = adopt_candidate_as_current_training_model(**adoption_arguments)

        assert (assignment_change.previous_model_id, assignment_change.current_model_id) == (
            previous_model_id,
            expected_temporary_model_id,
        )
        assert_local_adoption_matches_legacy(
            adoption_arguments=adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            assignment_change=assignment_change,
        )
        # 保留標本の追加は、既存モデルの統計・標本・計数と、全モデルの割当概念計数を変えない。
        assert loss_statistics_store.get_state_snapshot()[:-1] == previous_loss_statistics
        current_counts = counts_store.snapshot_model_training_and_assignment_counts()
        assert (
            current_counts.assigned_sample_counts_by_model_and_concept_id
            == previous_counts.assigned_sample_counts_by_model_and_concept_id
        )
        assert expected_temporary_model_id not in legacy_client.model_concept_counts
        for model_id in held_model_ids:
            assert (
                current_counts.trained_sample_counts_by_model_id[model_id]
                == previous_counts.trained_sample_counts_by_model_id[model_id]
            )
            assert (
                current_counts.parameter_update_step_counts_by_model_id[model_id]
                == previous_counts.parameter_update_step_counts_by_model_id[model_id]
            )
        current_training_samples = adoption_arguments[
            "training_sample_store"
        ].snapshot_ordered_model_training_samples()
        for collection, previous_collection in zip(
            current_training_samples, previous_training_samples
        ):
            assert collection.model_id == previous_collection.model_id
            assert all(
                training_sample is previous_training_sample
                for training_sample, previous_training_sample in zip(
                    collection.training_samples, previous_collection.training_samples
                )
            )
        assert current_training_samples[-1].model_id == expected_temporary_model_id
        assert all(
            training_sample is pending_training_sample
            for training_sample, pending_training_sample in zip(
                current_training_samples[-1].training_samples,
                adoption_arguments["pending_assignment_training_samples"],
            )
        )
        assert len(current_training_samples[-1].training_samples) == pending_sample_count
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]


@pytest.mark.parametrize("class_count", [2, 4])
def test_two_consecutive_adoptions_match_actual_legacy(class_count, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(409)
        adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
            class_count=class_count, monkeypatch=monkeypatch
        )
        finalize_adoption_in_legacy_client(legacy_client=legacy_client)
        first_assignment_change = adopt_candidate_as_current_training_model(**adoption_arguments)
        assert_local_adoption_matches_legacy(
            adoption_arguments=adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            assignment_change=first_assignment_change,
        )
        # 2回目の候補を、同じ初期値の実旧候補と新候補として別のoracleから取り出す。
        (
            second_adoption_arguments,
            second_shared_optimizer_owners,
            second_legacy_client,
            second_legacy_candidate_model,
        ) = build_local_adoption_oracle(
            class_count=class_count, held_model_ids=(4,), monkeypatch=monkeypatch
        )
        second_legacy_session = second_legacy_client._forward_validation
        second_legacy_session.old_model_id = legacy_client.current_model_id
        second_legacy_session.reference_models = dict(legacy_client.models)
        second_legacy_session.reference_losses = {
            model_id: [0.9] * 4 for model_id in legacy_client.models
        }
        assert second_legacy_session.candidate is second_legacy_candidate_model
        legacy_client._forward_validation = second_legacy_session
        for argument_name in (
            "adopted_candidate_classifier",
            "candidate_concept_specific_parameter_optimizer_state",
            "pending_assignment_training_samples",
        ):
            adoption_arguments[argument_name] = second_adoption_arguments[argument_name]
        finalize_adoption_in_legacy_client(legacy_client=legacy_client)
        second_assignment_change = adopt_candidate_as_current_training_model(**adoption_arguments)
        assert (
            second_assignment_change.previous_model_id == first_assignment_change.current_model_id
        )
        assert (
            second_assignment_change.current_model_id
            == first_assignment_change.current_model_id - 1
        )
        assert_local_adoption_matches_legacy(
            adoption_arguments=adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners + second_shared_optimizer_owners[-1:],
            legacy_client=legacy_client,
            assignment_change=second_assignment_change,
        )


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("allocator_none", TypeError),
        ("allocator_object", TypeError),
        ("allocator_subclass", TypeError),
        ("training_sample_store_none", TypeError),
        ("training_sample_store_object", TypeError),
        ("training_sample_store_subclass", TypeError),
        ("counts_store_none", TypeError),
        ("counts_store_object", TypeError),
        ("counts_store_subclass", TypeError),
        ("current_assignment_none", TypeError),
        ("current_assignment_subclass", TypeError),
        ("trained_sample_count_negative", ValueError),
        ("trained_sample_count_bool", TypeError),
        ("trained_sample_count_float", TypeError),
        ("trained_sample_count_subclass", TypeError),
        ("update_step_count_negative", ValueError),
        ("update_step_count_bool", TypeError),
        ("update_step_count_none", TypeError),
        ("update_step_count_subclass", TypeError),
        ("pending_samples_list", TypeError),
        ("pending_samples_element_tuple", TypeError),
        ("pending_samples_none", TypeError),
        ("temporary_id_used_in_training_sample_store", ValueError),
        ("temporary_id_used_in_trained_sample_counts", ValueError),
        ("temporary_id_used_in_assigned_concept_counts", ValueError),
        ("temporary_id_is_current_training_model_id", ValueError),
        ("registration_temporary_id_used_in_registry", ValueError),
        ("registration_temporary_id_used_in_statistics", ValueError),
        ("registration_temporary_id_used_in_pending_upload", ValueError),
        ("registration_no_held_model", LookupError),
        ("registration_upload_delay_zero", ValueError),
        ("registration_upload_delay_bool", TypeError),
        ("registration_registry_object", TypeError),
        ("registration_statistics_store_none", TypeError),
        ("registration_pending_upload_state_object", TypeError),
        ("registration_candidate_object", (TypeError, ValueError)),
        ("registration_optimizer_owner_mismatch", ValueError),
        ("registration_features_empty", ValueError),
        ("registration_labels_out_of_range", ValueError),
        ("registration_candidate_parameter_non_finite", ValueError),
    ],
)
def test_rejected_adoption_changes_no_state_including_allocator(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        used_temporary_model_id = TemporaryModelIdAllocator(client_id=3).next_temporary_model_id
        adoption_arguments, shared_optimizer_owners, _, _ = build_local_adoption_oracle(
            class_count=4,
            held_model_ids=()
            if invalid_case == "registration_no_held_model"
            else (4, used_temporary_model_id)
            if invalid_case == "registration_temporary_id_used_in_registry"
            else (4, 9),
            current_model_is_held=invalid_case != "registration_temporary_id_used_in_registry",
            existing_pending_model_id=used_temporary_model_id
            if invalid_case == "registration_temporary_id_used_in_pending_upload"
            else None,
            monkeypatch=monkeypatch,
        )
        valid_adoption_arguments = dict(adoption_arguments)
        training_sample_store = adoption_arguments["training_sample_store"]
        counts_store = adoption_arguments["model_training_and_assignment_counts_store"]
        loss_statistics_store = adoption_arguments["loss_statistics_store"]
        candidate = adoption_arguments["adopted_candidate_classifier"]
        owner_name_by_case_prefix = {
            "allocator": "temporary_model_id_allocator",
            "training_sample_store": "training_sample_store",
            "counts_store": "model_training_and_assignment_counts_store",
            "current_assignment": "current_training_model_assignment",
        }
        for case_prefix, owner_name in owner_name_by_case_prefix.items():
            if invalid_case == case_prefix + "_none":
                adoption_arguments[owner_name] = None
            elif invalid_case == case_prefix + "_object":
                adoption_arguments[owner_name] = object()
            elif invalid_case == case_prefix + "_subclass":
                owner_type = type(valid_adoption_arguments[owner_name])
                owner_subclass = type("OwnerSubclass", (owner_type,), {})
                adoption_arguments[owner_name] = (
                    owner_subclass(client_id=3)
                    if owner_type is TemporaryModelIdAllocator
                    else owner_subclass(initial_model_id=9)
                    if owner_type is CurrentTrainingModelAssignment
                    else owner_subclass()
                )
        invalid_count_by_case_suffix = {
            "negative": -1,
            "bool": True,
            "float": 1.0,
            "none": None,
            "subclass": IntSubclass(1),
        }
        for case_prefix, argument_name in (
            ("trained_sample_count_", "candidate_trained_sample_count"),
            ("update_step_count_", "candidate_parameter_update_step_count"),
        ):
            if invalid_case.startswith(case_prefix):
                adoption_arguments[argument_name] = invalid_count_by_case_suffix[
                    invalid_case.removeprefix(case_prefix)
                ]
        pending_assignment_training_samples = adoption_arguments[
            "pending_assignment_training_samples"
        ]
        if invalid_case == "pending_samples_list":
            adoption_arguments["pending_assignment_training_samples"] = list(
                pending_assignment_training_samples
            )
        elif invalid_case == "pending_samples_element_tuple":
            adoption_arguments["pending_assignment_training_samples"] = (
                pending_assignment_training_samples[0],
                (
                    pending_assignment_training_samples[1].input_features,
                    pending_assignment_training_samples[1].observed_class_labels,
                ),
            )
        elif invalid_case == "pending_samples_none":
            adoption_arguments["pending_assignment_training_samples"] = None
        elif invalid_case == "temporary_id_used_in_training_sample_store":
            training_sample_store.append_model_training_samples(
                model_id=used_temporary_model_id, training_samples=()
            )
        elif invalid_case == "temporary_id_used_in_trained_sample_counts":
            counts_store.record_completed_model_training(
                model_id=used_temporary_model_id,
                trained_sample_count=0,
                parameter_update_step_count=0,
            )
        elif invalid_case == "temporary_id_used_in_assigned_concept_counts":
            counts_store.record_assigned_sample_concept(
                model_id=used_temporary_model_id, observed_concept_id=1
            )
        elif invalid_case == "temporary_id_is_current_training_model_id":
            adoption_arguments["current_training_model_assignment"].assign_model_for_training(
                model_id=used_temporary_model_id
            )
        elif invalid_case == "registration_temporary_id_used_in_registry":
            # 一覧だけの重複にするため、同IDの統計・標本・計数は別IDへ移す。
            loss_statistics_store.reassign_model_loss_statistics_id(
                original_model_id=used_temporary_model_id, reassigned_model_id=30
            )
            training_sample_store.reassign_model_training_samples_id(
                original_model_id=used_temporary_model_id, reassigned_model_id=30
            )
            counts_store.transfer_model_training_and_assignment_counts(
                original_model_id=used_temporary_model_id, receiving_model_id=30
            )
        elif invalid_case == "registration_temporary_id_used_in_statistics":
            loss_statistics_store.set_model_loss_statistics(
                model_id=used_temporary_model_id,
                loss_statistics=loss_statistics_store.get_model_loss_statistics(model_id=4),
            )
        elif invalid_case == "registration_upload_delay_zero":
            adoption_arguments["upload_delay_round_count"] = 0
        elif invalid_case == "registration_upload_delay_bool":
            adoption_arguments["upload_delay_round_count"] = True
        elif invalid_case == "registration_registry_object":
            adoption_arguments["held_model_training_state_registry"] = object()
        elif invalid_case == "registration_statistics_store_none":
            adoption_arguments["loss_statistics_store"] = None
        elif invalid_case == "registration_pending_upload_state_object":
            adoption_arguments["pending_model_upload_state"] = object()
        elif invalid_case == "registration_candidate_object":
            adoption_arguments["adopted_candidate_classifier"] = object()
        elif invalid_case == "registration_optimizer_owner_mismatch":
            adoption_arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                shared_optimizer_owners[-1]
            )
        elif invalid_case == "registration_features_empty":
            adoption_arguments["initial_statistics_input_features"] = adoption_arguments[
                "initial_statistics_input_features"
            ][:0]
            adoption_arguments["initial_statistics_observed_class_labels"] = adoption_arguments[
                "initial_statistics_observed_class_labels"
            ][:0]
        elif invalid_case == "registration_labels_out_of_range":
            adoption_arguments["initial_statistics_observed_class_labels"][0, 0] = 4.0
        elif invalid_case == "registration_candidate_parameter_non_finite":
            with torch.no_grad():
                candidate.classification_layer.bias[0] = float("-inf")
        assert_adoption_rejected_without_any_change(
            adoption_arguments=adoption_arguments,
            valid_adoption_arguments=valid_adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            expected_exception=expected_exception,
        )


def test_registration_completes_before_allocation_counts_samples_and_assignment(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        adoption_arguments, _, _, _ = build_local_adoption_oracle(
            class_count=2, monkeypatch=monkeypatch
        )
        actual_call_order = []
        original_registration = adoption_module.register_adopted_candidate_as_temporary_held_model

        def record_then_register(**arguments):
            actual_call_order.append("register_adopted_candidate_as_temporary_held_model")
            return original_registration(**arguments)

        monkeypatch.setattr(
            adoption_module,
            "register_adopted_candidate_as_temporary_held_model",
            record_then_register,
        )
        for owner_name, operation_name in (
            ("temporary_model_id_allocator", "allocate_temporary_model_id"),
            ("model_training_and_assignment_counts_store", "record_completed_model_training"),
            ("training_sample_store", "append_model_training_samples"),
            ("current_training_model_assignment", "assign_model_for_training"),
        ):
            original_operation = getattr(adoption_arguments[owner_name], operation_name)

            def record_then_update(
                *, operation_name=operation_name, original_operation=original_operation, **arguments
            ):
                actual_call_order.append(operation_name)
                return original_operation(**arguments)

            monkeypatch.setattr(adoption_arguments[owner_name], operation_name, record_then_update)
        adopt_candidate_as_current_training_model(**adoption_arguments)
        expected_call_order = [
            "register_adopted_candidate_as_temporary_held_model",
            "allocate_temporary_model_id",
            "record_completed_model_training",
            "append_model_training_samples",
            "assign_model_for_training",
        ]
        assert actual_call_order == expected_call_order


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_adopted_candidate_continues_actual_joint_training_and_confirmation(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(419)
        adoption_arguments, shared_optimizer_owners, legacy_client, legacy_candidate_model = (
            build_local_adoption_oracle(
                class_count=class_count,
                pending_sample_count=4,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = adoption_arguments["held_model_training_state_registry"]
        training_sample_store = adoption_arguments["training_sample_store"]
        counts_store = adoption_arguments["model_training_and_assignment_counts_store"]
        loss_statistics_store = adoption_arguments["loss_statistics_store"]
        candidate = adoption_arguments["adopted_candidate_classifier"]
        candidate_optimizer_owner = adoption_arguments[
            "candidate_concept_specific_parameter_optimizer_state"
        ]
        active = registry.get_held_model_training_state(model_id=9).classifier.feature_extractor
        input_features = adoption_arguments["initial_statistics_input_features"]
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
                shared_optimizer_owners=shared_optimizer_owners,
                legacy_client=legacy_client,
                input_features=input_features,
            )
            assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
            assert_training_samples_match_legacy(
                training_sample_store=training_sample_store, legacy_client=legacy_client
            )

        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        # 採用前: 保有2モデルの共同更新と、独立した共有部を持つ候補だけの学習。
        run_joint_update_in_both_implementations()
        candidate_features = torch.cat(
            [
                sample.input_features
                for sample in adoption_arguments["pending_assignment_training_samples"]
            ]
        )
        candidate_labels = torch.cat(
            [
                sample.observed_class_labels
                for sample in adoption_arguments["pending_assignment_training_samples"]
            ]
        )
        legacy_candidate_model.update(candidate_features.clone(), candidate_labels.clone())
        perform_joint_model_parameter_update(
            local_training_settings=local_training_settings,
            shared_feature_extractor=candidate.feature_extractor,
            shared_parameter_optimizer=shared_optimizer_owners[-1].parameter_optimizer,
            participating_training_batches=(
                ParticipatingModelTrainingBatch(
                    classifier=candidate,
                    concept_specific_parameter_optimizer=candidate_optimizer_owner.parameter_optimizer,
                    input_features=candidate_features,
                    observed_class_labels=candidate_labels,
                ),
            ),
            update_shared_features=True,
        )

        finalize_adoption_in_legacy_client(legacy_client=legacy_client)
        assignment_change = adopt_candidate_as_current_training_model(**adoption_arguments)
        temporary_model_id = assignment_change.current_model_id
        assert_local_adoption_matches_legacy(
            adoption_arguments=adoption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            assignment_change=assignment_change,
        )

        # 採用後: 一時IDの新モデルを含む3モデルを、追加された保留標本で共同更新する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        registered_candidate_optimizer = candidate_optimizer_owner.parameter_optimizer
        if optimizer_variant != "sgd":
            assert registered_candidate_optimizer.state
        assert_store_statistics_match_legacy(
            loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
        )

        # 採用で現在IDは一時IDになっているので、切替えを補わずに正式ID確認へ接続できる。
        legacy_client.stored_data = {}
        confirmation_change = confirm_held_model_registration(
            registered_global_model_id=12,
            held_model_training_state_registry=registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=training_sample_store,
            evaluation_sample_store=ModelEvaluationSampleStore(
                maximum_stored_sample_count_per_model=3, added_batch_sample_count=1
            ),
            model_training_and_assignment_counts_store=counts_store,
            current_training_model_assignment=adoption_arguments[
                "current_training_model_assignment"
            ],
            pending_model_upload_state=adoption_arguments["pending_model_upload_state"],
        )
        BaseClient.confirm_model_registration(legacy_client, 12)
        assert (confirmation_change.previous_model_id, confirmation_change.current_model_id) == (
            temporary_model_id,
            12,
        )
        assert legacy_client.current_model_id == 12
        assert tuple(legacy_client.models) == (4, 9, 12)
        assert registry.get_held_model_training_state(model_id=12).classifier is candidate
        assert candidate_optimizer_owner.parameter_optimizer is registered_candidate_optimizer
        assert adoption_arguments["pending_model_upload_state"].get_pending_model_upload() is None
        assert legacy_client.pending_model_params is None
        assert_store_statistics_match_legacy(
            loss_statistics_store=loss_statistics_store, legacy_client=legacy_client
        )

        # 正式ID確認後: 付け替えた標本・計数・optimizer stateで学習を継続する。
        run_joint_update_in_both_implementations()
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
