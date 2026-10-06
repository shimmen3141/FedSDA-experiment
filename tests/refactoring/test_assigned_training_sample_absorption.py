"""帰属確定標本の吸収を、実旧の吸収処理と前向き検証確定の非採用分岐へ対照する。"""

import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
)
from test_adopted_candidate_local_adoption import (
    assert_training_samples_match_legacy,
    build_local_adoption_oracle,
)
from test_joint_model_parameter_update import assert_nested_state_equal, run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

import federated_learning_experiments.runtime.assigned_training_sample_absorption as absorption_module
from federated_drift_experiment import config
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
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)

OWNER_NAMES = (
    "held_model_training_state_registry",
    "training_sample_store",
    "model_training_and_assignment_counts_store",
    "loss_statistics_store",
)


class IntSubclass(int):
    """IDとして受理しない派生型。"""


def build_absorption_oracle(
    *,
    class_count,
    monkeypatch,
    target_model_case="current",
    absorbed_sample_count=3,
    concept_id_case="all",
    optimizer_variant="standard",
):
    """上流の採用oracleから新ownerと実旧clientを取り出し、吸収する標本列を両形式で作る。"""
    adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
        class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
    )
    loss_statistics_store = adoption_arguments["loss_statistics_store"]
    model_id = 9 if target_model_case == "current" else 4
    if target_model_case == "without_statistics":
        # 吸収先の統計を両実装で別IDへ退避し、未登録モデルが件数0から始まることを観測する。
        loss_statistics_store.reassign_model_loss_statistics_id(
            original_model_id=4, reassigned_model_id=999
        )
        legacy_client.model_stats[999] = legacy_client.model_stats.pop(4)
    if target_model_case == "without_training_samples":
        # 吸収先の標本列を両実装で別IDへ退避し、空列の吸収が標本列を作らないことを観測する。
        adoption_arguments["training_sample_store"].reassign_model_training_samples_id(
            original_model_id=4, reassigned_model_id=999
        )
        legacy_client.train_data_store[999] = legacy_client.train_data_store.pop(4)
    input_features = adoption_arguments["initial_statistics_input_features"]
    observed_class_labels = adoption_arguments["initial_statistics_observed_class_labels"]
    assigned_training_samples = tuple(
        ObservedTrainingSample(
            input_features=input_features[sample_index % len(input_features)].reshape(1, -1) + 0.25,
            observed_class_labels=observed_class_labels[
                (sample_index * 2) % len(observed_class_labels)
            ].reshape(1, 1),
        )
        for sample_index in range(absorbed_sample_count)
    )
    assigned_sample_concept_ids = tuple(
        None
        if concept_id_case == "absent" or (concept_id_case == "mixed" and sample_index % 2)
        else sample_index % 2
        for sample_index in range(absorbed_sample_count)
    )
    # 旧標本は(特徴, ラベル[, 真の概念])。概念要素なしと概念Noneの両方を旧形式で表す。
    legacy_training_samples = [
        (training_sample.input_features, training_sample.observed_class_labels)
        if concept_id_case == "absent"
        else (
            training_sample.input_features,
            training_sample.observed_class_labels,
            observed_concept_id,
        )
        for training_sample, observed_concept_id in zip(
            assigned_training_samples, assigned_sample_concept_ids
        )
    ]
    absorption_arguments = dict(
        model_id=model_id,
        assigned_training_samples=assigned_training_samples,
        assigned_sample_concept_ids=assigned_sample_concept_ids,
        held_model_training_state_registry=adoption_arguments["held_model_training_state_registry"],
        training_sample_store=adoption_arguments["training_sample_store"],
        model_training_and_assignment_counts_store=adoption_arguments[
            "model_training_and_assignment_counts_store"
        ],
        loss_statistics_store=loss_statistics_store,
    )
    return (
        absorption_arguments,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_training_samples,
    )


def assert_absorption_matches_legacy(*, absorption_arguments, legacy_client):
    assert_training_samples_match_legacy(
        training_sample_store=absorption_arguments["training_sample_store"],
        legacy_client=legacy_client,
    )
    assert_model_counts_match_legacy(
        counts_store=absorption_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert_store_statistics_match_legacy(
        loss_statistics_store=absorption_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
    )


def snapshot_absorption_state(*, absorption_arguments, shared_optimizer_owners):
    registry = absorption_arguments["held_model_training_state_registry"]
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    optimizer_owners = tuple(
        state.concept_specific_parameter_optimizer_state for state in held_model_training_states
    ) + tuple(shared_optimizer_owners)
    return dict(
        held_model_training_states=held_model_training_states,
        parameter_snapshots=snapshot_parameter_values_and_gradients(
            parameter
            for state in held_model_training_states
            for parameter in state.classifier.parameters()
        ),
        optimizers=tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in optimizer_owners
        ),
        training_samples=tuple(
            (collection.model_id, collection.training_samples)
            for collection in absorption_arguments[
                "training_sample_store"
            ].snapshot_ordered_model_training_samples()
        ),
        counts=absorption_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts(),
        loss_statistics=absorption_arguments["loss_statistics_store"].get_state_snapshot(),
        random_states=(torch.get_rng_state().clone(), random.getstate(), np.random.get_state()),
    )


def assert_models_optimizers_and_random_state_unchanged(*, previous_snapshot, registry):
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    assert len(held_model_training_states) == len(previous_snapshot["held_model_training_states"])
    assert all(
        state is previous_state
        for state, previous_state in zip(
            held_model_training_states, previous_snapshot["held_model_training_states"]
        )
    )
    assert_parameter_values_and_gradients_unchanged(previous_snapshot["parameter_snapshots"])
    for owner, previous_optimizer, previous_optimizer_state in previous_snapshot["optimizers"]:
        assert owner.parameter_optimizer is previous_optimizer
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
    previous_torch_state, previous_python_state, previous_numpy_state = previous_snapshot[
        "random_states"
    ]
    assert torch.equal(torch.get_rng_state(), previous_torch_state)
    assert random.getstate() == previous_python_state
    numpy_state = np.random.get_state()
    assert numpy_state[0] == previous_numpy_state[0]
    assert np.array_equal(numpy_state[1], previous_numpy_state[1])
    assert numpy_state[2:] == previous_numpy_state[2:]


def assert_absorption_state_unchanged(*, previous_snapshot, valid_absorption_arguments):
    assert_models_optimizers_and_random_state_unchanged(
        previous_snapshot=previous_snapshot,
        registry=valid_absorption_arguments["held_model_training_state_registry"],
    )
    model_training_sample_collections = valid_absorption_arguments[
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
        valid_absorption_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        == previous_snapshot["counts"]
    )
    assert (
        valid_absorption_arguments["loss_statistics_store"].get_state_snapshot()
        == previous_snapshot["loss_statistics"]
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "target_model_case", ["current", "other", "without_statistics", "without_training_samples"]
)
@pytest.mark.parametrize("absorbed_sample_count", [0, 1, 5])
@pytest.mark.parametrize("concept_id_case", ["all", "mixed", "absent"])
def test_absorption_matches_actual_legacy_absorption(
    class_count, target_model_case, absorbed_sample_count, concept_id_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(503)
        (
            absorption_arguments,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_training_samples,
        ) = build_absorption_oracle(
            class_count=class_count,
            target_model_case=target_model_case,
            absorbed_sample_count=absorbed_sample_count,
            concept_id_case=concept_id_case,
            monkeypatch=monkeypatch,
        )
        model_id = absorption_arguments["model_id"]
        previous_snapshot = snapshot_absorption_state(
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        current_model_id = adoption_arguments[
            "current_training_model_assignment"
        ].current_training_model_id

        legacy_client._absorb_into_store(model_id, legacy_training_samples)
        assert absorb_assigned_training_samples_into_held_model(**absorption_arguments) is None

        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        assert_models_optimizers_and_random_state_unchanged(
            previous_snapshot=previous_snapshot,
            registry=absorption_arguments["held_model_training_state_registry"],
        )
        # 吸収先以外の標本・統計と、全モデルの学習計数は変わらない。
        current_counts = absorption_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts()
        assert (
            current_counts.trained_sample_counts_by_model_id
            == previous_snapshot["counts"].trained_sample_counts_by_model_id
        )
        assert (
            current_counts.parameter_update_step_counts_by_model_id
            == previous_snapshot["counts"].parameter_update_step_counts_by_model_id
        )
        current_training_samples = absorption_arguments[
            "training_sample_store"
        ].snapshot_ordered_model_training_samples()
        if target_model_case == "without_training_samples":
            # 標本列を持たないモデルは、1件以上吸収したときだけ末尾に列ができる。
            assert len(current_training_samples) == len(previous_snapshot["training_samples"]) + (
                1 if absorbed_sample_count else 0
            )
            assert (model_id in legacy_client.train_data_store) is bool(absorbed_sample_count)
            if absorbed_sample_count:
                assert current_training_samples[-1].model_id == model_id
                assert all(
                    training_sample is assigned_training_sample
                    for training_sample, assigned_training_sample in zip(
                        current_training_samples[-1].training_samples,
                        absorption_arguments["assigned_training_samples"],
                    )
                )
        for collection, (previous_model_id, previous_training_samples) in zip(
            current_training_samples,
            previous_snapshot["training_samples"],
        ):
            assert collection.model_id == previous_model_id
            expected_training_samples = previous_training_samples + (
                absorption_arguments["assigned_training_samples"]
                if previous_model_id == model_id
                else ()
            )
            assert len(collection.training_samples) == len(expected_training_samples)
            assert all(
                training_sample is expected_training_sample
                for training_sample, expected_training_sample in zip(
                    collection.training_samples, expected_training_samples
                )
            )
        for (other_model_id, loss_statistics), (_, previous_loss_statistics) in zip(
            absorption_arguments["loss_statistics_store"].get_state_snapshot(),
            previous_snapshot["loss_statistics"],
        ):
            if other_model_id != model_id:
                assert loss_statistics == previous_loss_statistics
        assert (
            adoption_arguments["current_training_model_assignment"].current_training_model_id
            == current_model_id
        )
        if absorbed_sample_count == 0:
            assert_absorption_state_unchanged(
                previous_snapshot=previous_snapshot,
                valid_absorption_arguments=absorption_arguments,
            )


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("model_id_bool", TypeError),
        ("model_id_float", TypeError),
        ("model_id_none", TypeError),
        ("model_id_subclass", TypeError),
        ("samples_list", TypeError),
        ("samples_none", TypeError),
        ("samples_element_tuple", TypeError),
        ("concept_ids_list", TypeError),
        ("concept_ids_none", TypeError),
        ("concept_ids_shorter", ValueError),
        ("concept_ids_longer", ValueError),
        ("concept_id_bool", TypeError),
        ("concept_id_float", TypeError),
        ("concept_id_subclass", TypeError),
        ("model_not_held", KeyError),
        ("model_not_held_with_empty_samples", KeyError),
        ("sample_with_two_rows_in_middle", ValueError),
        ("features_feature_count_in_middle", ValueError),
        ("features_non_finite_last", ValueError),
        ("features_not_tensor_last", TypeError),
        ("labels_out_of_range_last", ValueError),
        ("labels_fractional_first", ValueError),
        ("labels_shape_in_middle", ValueError),
    ]
    + [
        (owner_name + "_" + invalid_owner, TypeError)
        for owner_name in OWNER_NAMES
        for invalid_owner in ("none", "object", "subclass")
    ],
)
def test_rejected_absorption_changes_no_state_for_any_sample(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        absorption_arguments, _, shared_optimizer_owners, _, _ = build_absorption_oracle(
            class_count=4, absorbed_sample_count=3, monkeypatch=monkeypatch
        )
        valid_absorption_arguments = dict(absorption_arguments)
        assigned_training_samples = absorption_arguments["assigned_training_samples"]

        def replace_training_sample(
            sample_index, *, input_features=None, observed_class_labels=None
        ):
            training_sample = assigned_training_samples[sample_index]
            absorption_arguments["assigned_training_samples"] = (
                assigned_training_samples[:sample_index]
                + (
                    ObservedTrainingSample(
                        input_features=training_sample.input_features
                        if input_features is None
                        else input_features,
                        observed_class_labels=training_sample.observed_class_labels
                        if observed_class_labels is None
                        else observed_class_labels,
                    ),
                )
                + assigned_training_samples[sample_index + 1 :]
            )

        if invalid_case.startswith("model_id_"):
            absorption_arguments["model_id"] = {
                "bool": True,
                "float": 9.0,
                "none": None,
                "subclass": IntSubclass(9),
            }[invalid_case.removeprefix("model_id_")]
        elif invalid_case == "samples_list":
            absorption_arguments["assigned_training_samples"] = list(assigned_training_samples)
        elif invalid_case == "samples_none":
            absorption_arguments["assigned_training_samples"] = None
        elif invalid_case == "samples_element_tuple":
            absorption_arguments["assigned_training_samples"] = assigned_training_samples[:2] + (
                (
                    assigned_training_samples[2].input_features,
                    assigned_training_samples[2].observed_class_labels,
                ),
            )
        elif invalid_case == "concept_ids_list":
            absorption_arguments["assigned_sample_concept_ids"] = [0, 1, 0]
        elif invalid_case == "concept_ids_none":
            absorption_arguments["assigned_sample_concept_ids"] = None
        elif invalid_case == "concept_ids_shorter":
            absorption_arguments["assigned_sample_concept_ids"] = (0, 1)
        elif invalid_case == "concept_ids_longer":
            absorption_arguments["assigned_sample_concept_ids"] = (0, 1, 0, None)
        elif invalid_case.startswith("concept_id_"):
            absorption_arguments["assigned_sample_concept_ids"] = (
                0,
                None,
                {"bool": True, "float": 1.0, "subclass": IntSubclass(1)}[
                    invalid_case.removeprefix("concept_id_")
                ],
            )
        elif invalid_case == "model_not_held":
            absorption_arguments["model_id"] = 77
        elif invalid_case == "model_not_held_with_empty_samples":
            absorption_arguments["model_id"] = 77
            absorption_arguments["assigned_training_samples"] = ()
            absorption_arguments["assigned_sample_concept_ids"] = ()
        elif invalid_case == "sample_with_two_rows_in_middle":
            replace_training_sample(
                1,
                input_features=torch.cat([assigned_training_samples[1].input_features] * 2),
                observed_class_labels=torch.cat(
                    [assigned_training_samples[1].observed_class_labels] * 2
                ),
            )
        elif invalid_case == "features_feature_count_in_middle":
            replace_training_sample(1, input_features=torch.ones((1, 3)))
        elif invalid_case == "features_non_finite_last":
            replace_training_sample(2, input_features=torch.tensor([[float("nan"), 0.0]]))
        elif invalid_case == "features_not_tensor_last":
            replace_training_sample(2, input_features=[[0.0, 1.0]])
        elif invalid_case == "labels_out_of_range_last":
            replace_training_sample(2, observed_class_labels=torch.tensor([[4.0]]))
        elif invalid_case == "labels_fractional_first":
            replace_training_sample(0, observed_class_labels=torch.tensor([[0.5]]))
        elif invalid_case == "labels_shape_in_middle":
            replace_training_sample(1, observed_class_labels=torch.tensor([1.0]))
        else:
            owner_name, invalid_owner = invalid_case.rsplit("_", 1)
            if invalid_owner == "subclass":
                owner_type = type(valid_absorption_arguments[owner_name])
                absorption_arguments[owner_name] = type("OwnerSubclass", (owner_type,), {})()
            else:
                absorption_arguments[owner_name] = None if invalid_owner == "none" else object()
        previous_snapshot = snapshot_absorption_state(
            absorption_arguments=valid_absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with pytest.raises(expected_exception):
            absorb_assigned_training_samples_into_held_model(**absorption_arguments)
        assert_absorption_state_unchanged(
            previous_snapshot=previous_snapshot,
            valid_absorption_arguments=valid_absorption_arguments,
        )


def test_actual_legacy_absorption_keeps_earlier_samples_when_a_later_sample_fails(monkeypatch):
    """旧は途中の標本で失敗しても先行標本の更新を残す。新は同じ入力で何も変更しない。"""
    with torch.random.fork_rng(devices=[]):
        absorption_arguments, _, shared_optimizer_owners, legacy_client, legacy_training_samples = (
            build_absorption_oracle(class_count=4, absorbed_sample_count=3, monkeypatch=monkeypatch)
        )
        invalid_input_features = torch.ones((1, 3))
        legacy_training_samples[1] = (invalid_input_features, *legacy_training_samples[1][1:])
        assigned_training_samples = absorption_arguments["assigned_training_samples"]
        absorption_arguments["assigned_training_samples"] = (
            assigned_training_samples[0],
            ObservedTrainingSample(
                input_features=invalid_input_features,
                observed_class_labels=assigned_training_samples[1].observed_class_labels,
            ),
            assigned_training_samples[2],
        )
        previous_legacy_sample_count = len(legacy_client.train_data_store[9])
        previous_legacy_statistics_count = legacy_client.model_stats[9]["n"]
        previous_legacy_concept_counts = dict(legacy_client.model_concept_counts[9])
        with pytest.raises(RuntimeError):
            legacy_client._absorb_into_store(9, legacy_training_samples)
        # 1件目は全更新済み、2件目は標本と概念だけ追加済みで統計は未更新、3件目は未処理。
        assert len(legacy_client.train_data_store[9]) == previous_legacy_sample_count + 2
        assert legacy_client.model_stats[9]["n"] == previous_legacy_statistics_count + 1
        assert (
            sum(legacy_client.model_concept_counts[9].values())
            == sum(previous_legacy_concept_counts.values()) + 2
        )
        previous_snapshot = snapshot_absorption_state(
            absorption_arguments=absorption_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
        )
        with pytest.raises(ValueError):
            absorb_assigned_training_samples_into_held_model(**absorption_arguments)
        assert_absorption_state_unchanged(
            previous_snapshot=previous_snapshot, valid_absorption_arguments=absorption_arguments
        )


def test_all_losses_are_evaluated_before_per_sample_owner_updates(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        absorption_arguments, _, _, _, _ = build_absorption_oracle(
            class_count=2, absorbed_sample_count=2, monkeypatch=monkeypatch
        )
        actual_call_order = []
        original_evaluation = absorption_module.evaluate_classifier_per_sample_bounded_losses

        def record_then_evaluate(**arguments):
            actual_call_order.append("evaluate_classifier_per_sample_bounded_losses")
            return original_evaluation(**arguments)

        monkeypatch.setattr(
            absorption_module,
            "evaluate_classifier_per_sample_bounded_losses",
            record_then_evaluate,
        )
        for owner_name, operation_name in (
            ("training_sample_store", "append_model_training_samples"),
            ("model_training_and_assignment_counts_store", "record_assigned_sample_concept"),
            ("loss_statistics_store", "record_assigned_loss"),
        ):
            original_operation = getattr(absorption_arguments[owner_name], operation_name)

            def record_then_update(
                *, operation_name=operation_name, original_operation=original_operation, **arguments
            ):
                actual_call_order.append(operation_name)
                return original_operation(**arguments)

            monkeypatch.setattr(
                absorption_arguments[owner_name], operation_name, record_then_update
            )
        absorb_assigned_training_samples_into_held_model(**absorption_arguments)
        expected_call_order = ["evaluate_classifier_per_sample_bounded_losses"] * 2 + [
            "append_model_training_samples",
            "record_assigned_sample_concept",
            "record_assigned_loss",
        ] * 2
        assert actual_call_order == expected_call_order


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "legacy_resolution_case", ["create_rejected", "maintain_current", "reuse_other"]
)
def test_absorption_matches_actual_legacy_non_adoption_branches_of_forward_validation(
    class_count, legacy_resolution_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(509)
        adoption_arguments, _, legacy_client, _ = build_local_adoption_oracle(
            class_count=class_count, pending_sample_count=4, monkeypatch=monkeypatch
        )
        legacy_session = legacy_client._forward_validation
        # 候補の損失を参照より大きくし、実旧の判定を非採用にする。
        legacy_session.candidate_losses = [0.9] * 4
        legacy_session.reference_losses = {4: [0.2] * 4, 9: [0.3] * 4}
        if legacy_resolution_case == "maintain_current":
            legacy_session.reference_historical_means = {9: 0.3}
        elif legacy_resolution_case == "reuse_other":
            legacy_session.reference_historical_means = {4: 0.2}
        registry = adoption_arguments["held_model_training_state_registry"]
        current_training_model_assignment = adoption_arguments["current_training_model_assignment"]
        previous_held_model_ids = tuple(legacy_client.models)
        previous_next_temporary_model_id = legacy_client.next_temp_id
        parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for state in registry.snapshot_ordered_held_model_training_states()
            for parameter in state.classifier.parameters()
        )

        legacy_drift_type = legacy_client._finalize_forward_validation(57)
        assert not legacy_client.provisional_model_decisions[-1].accepted
        expected_model_id = {"create_rejected": 9, "maintain_current": 9, "reuse_other": 4}[
            legacy_resolution_case
        ]
        assert legacy_client.current_model_id == expected_model_id
        assert legacy_drift_type == (1 if legacy_resolution_case == "reuse_other" else 0)
        assert (
            legacy_client.adaptation_events[-1].action
            == {
                "create_rejected": "create_rejected",
                "maintain_current": "maintain",
                "reuse_other": "reuse",
            }[legacy_resolution_case]
        )
        assert tuple(legacy_client.models) == previous_held_model_ids
        assert legacy_client.next_temp_id == previous_next_temporary_model_id

        # 帰属先の決定と現在IDの切替えは後続specの責務。ここでは旧が決めた帰属先へtest-only接続する。
        current_training_model_assignment.assign_model_for_training(model_id=expected_model_id)
        absorption_arguments = dict(
            model_id=expected_model_id,
            assigned_training_samples=adoption_arguments["pending_assignment_training_samples"],
            assigned_sample_concept_ids=(1,) * 4,
            held_model_training_state_registry=registry,
            training_sample_store=adoption_arguments["training_sample_store"],
            model_training_and_assignment_counts_store=adoption_arguments[
                "model_training_and_assignment_counts_store"
            ],
            loss_statistics_store=adoption_arguments["loss_statistics_store"],
        )
        absorb_assigned_training_samples_into_held_model(**absorption_arguments)
        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert adoption_arguments["pending_model_upload_state"].get_pending_model_upload() is None
        assert legacy_client.pending_model_params is None


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_absorbed_samples_continue_actual_joint_training(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(521)
        (
            absorption_arguments,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_training_samples,
        ) = build_absorption_oracle(
            class_count=class_count,
            absorbed_sample_count=4,
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = absorption_arguments["held_model_training_state_registry"]
        training_sample_store = absorption_arguments["training_sample_store"]
        counts_store = absorption_arguments["model_training_and_assignment_counts_store"]
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
                shared_optimizer_owners=shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=input_features,
            )

        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        # 学習でparameterが変わった後のモデルで、吸収時の損失を評価する。
        run_joint_update_in_both_implementations()
        legacy_client._absorb_into_store(absorption_arguments["model_id"], legacy_training_samples)
        absorb_assigned_training_samples_into_held_model(**absorption_arguments)
        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        # 吸収した標本を含むbatchで学習を継続する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert_absorption_matches_legacy(
            absorption_arguments=absorption_arguments, legacy_client=legacy_client
        )
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
