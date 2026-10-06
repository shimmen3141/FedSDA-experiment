"""警報後の候補検証標本の観測を、実旧の観測処理と到達時の確定へ対照する。"""

import random
from collections import OrderedDict, defaultdict

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
    convert_legacy_parameter_name,
)
from test_adopted_candidate_local_adoption import (
    assert_training_samples_match_legacy,
    build_local_adoption_oracle,
)
from test_joint_model_parameter_update import run_legacy_joint_update
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_post_alarm_candidate_validation_resolution import (
    LEGACY_ACTION_BY_RESOLUTION_OUTCOME,
    assert_resolution_matches_legacy,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

import federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation as observation_module
from federated_drift_experiment import config
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    apply_post_alarm_candidate_validation_resolution,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)

# 実旧の確定処理が切替位置として記録する、最後の検証標本の位置（上流の確定testと同じ値）。
LAST_VALIDATION_SAMPLE_INDEX = 57


class IntSubclass(int):
    """位置やIDとして受理しない派生型。"""


def make_acceptance_settings(*, required_validation_sample_count):
    return CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy=(
            "current_model_first_reuse_then_two_segment_candidate_validation"
        ),
        candidate_post_alarm_validation_sample_count=required_validation_sample_count,
    )


def prepare_validation_session_in_both_implementations(
    *,
    adoption_arguments,
    legacy_client,
    class_count,
    required_validation_sample_count,
    historical_mean_case="none",
):
    """現在の保有モデルの値で、実旧の参照snapshotと同じ値の新の参照分類器・損失収集を作る。"""
    legacy_client.model_cls = lambda: ResidualAdapterMLP(input_dim=2, dataset="sine2")
    legacy_session = legacy_client._forward_validation
    legacy_session.target_count = required_validation_sample_count
    legacy_session.proposal_position = (
        LAST_VALIDATION_SAMPLE_INDEX - required_validation_sample_count
    )
    # 実旧の複製処理で、警報時点の値を持つ独立した参照モデルを作る。
    legacy_session.reference_models = legacy_client._snapshot_reference_models()
    legacy_session.candidate_losses = []
    legacy_session.reference_losses = {model_id: [] for model_id in legacy_session.reference_models}
    # 履歴平均を十分大きくすると、そのモデルは前向き損失が履歴の範囲内と評価される。
    legacy_session.reference_historical_means = {
        "none": {},
        "current": {9: 1.0},
        "other": {4: 1.0},
    }[historical_mean_case]
    reference_classifiers_by_model_id = {}
    for model_id, legacy_reference_model in legacy_session.reference_models.items():
        reference_classifier = ResidualAdapterClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=2,
            ),
            input_feature_count=2,
            hidden_layer_widths=(5, 4),
            class_count=class_count,
        )
        reference_classifier.load_state_dict(
            {
                convert_legacy_parameter_name(parameter_name): parameter_value
                for parameter_name, parameter_value in legacy_reference_model.state_dict().items()
            }
        )
        reference_classifiers_by_model_id[model_id] = reference_classifier
    observation_arguments = dict(
        candidate_classifier=adoption_arguments["adopted_candidate_classifier"],
        reference_classifiers_by_model_id=reference_classifiers_by_model_id,
        post_alarm_candidate_loss_collection=PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=make_acceptance_settings(
                required_validation_sample_count=required_validation_sample_count
            ),
            proposal_sample_index=legacy_session.proposal_position,
            reference_model_ids=tuple(reference_classifiers_by_model_id),
        ),
    )
    input_features = adoption_arguments["initial_statistics_input_features"]
    observed_class_labels = adoption_arguments["initial_statistics_observed_class_labels"]
    validation_samples = tuple(
        (
            legacy_session.proposal_position + 1 + sample_offset,
            input_features[sample_offset % len(input_features)].reshape(1, -1) * 0.5
            + 0.1 * sample_offset,
            observed_class_labels[(sample_offset * 2) % len(observed_class_labels)].reshape(1, 1),
        )
        for sample_offset in range(required_validation_sample_count)
    )
    return observation_arguments, validation_samples, legacy_session


def build_observation_oracle(
    *,
    class_count,
    monkeypatch,
    required_validation_sample_count=4,
    historical_mean_case="none",
    optimizer_variant="standard",
):
    adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
        class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
    )
    observation_arguments, validation_samples, legacy_session = (
        prepare_validation_session_in_both_implementations(
            adoption_arguments=adoption_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            required_validation_sample_count=required_validation_sample_count,
            historical_mean_case=historical_mean_case,
        )
    )
    return (
        observation_arguments,
        validation_samples,
        adoption_arguments,
        shared_optimizer_owners,
        legacy_client,
        legacy_session,
    )


def assert_collected_losses_match_legacy(*, post_alarm_candidate_loss_collection, legacy_session):
    collection_state = post_alarm_candidate_loss_collection.get_state_snapshot()
    assert collection_state.candidate_losses == tuple(legacy_session.candidate_losses)
    assert tuple(
        model_id for model_id, _ in collection_state.reference_losses_by_model_id
    ) == tuple(legacy_session.reference_losses)
    for model_id, reference_losses in collection_state.reference_losses_by_model_id:
        assert reference_losses == tuple(legacy_session.reference_losses[model_id])
    assert collection_state.ready_for_acceptance_evaluation is legacy_session.ready
    assert collection_state.validation_sample_count == legacy_session.validation_count


def observe_and_resolve_in_both_implementations(
    *, observation_arguments, validation_samples, adoption_arguments, legacy_client, legacy_session
):
    """全検証標本を両実装へ渡す。実旧は到達時に確定まで進み、新は評価と確定をtest-only接続する。"""
    registry = adoption_arguments["held_model_training_state_registry"]
    current_training_model_assignment = adoption_arguments["current_training_model_assignment"]
    previous_model_id = current_training_model_assignment.current_training_model_id
    legacy_drift_type = None
    ready_for_acceptance_evaluation = False
    for sample_index, input_features, observed_class_labels in validation_samples:
        assert legacy_client._forward_validation is legacy_session
        legacy_drift_type = legacy_client._observe_forward_validation(
            input_features, observed_class_labels, sample_index
        )
        ready_for_acceptance_evaluation = observe_post_alarm_candidate_validation_sample(
            sample_index=sample_index,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
            **observation_arguments,
        )
        assert_collected_losses_match_legacy(
            post_alarm_candidate_loss_collection=observation_arguments[
                "post_alarm_candidate_loss_collection"
            ],
            legacy_session=legacy_session,
        )
    assert ready_for_acceptance_evaluation is True
    assert legacy_client._forward_validation is None
    collection_state = observation_arguments[
        "post_alarm_candidate_loss_collection"
    ].get_state_snapshot()
    evaluation = evaluate_candidate_using_post_alarm_losses(
        candidate_model_training_and_acceptance_settings=make_acceptance_settings(
            required_validation_sample_count=collection_state.required_validation_sample_count
        ),
        candidate_losses=collection_state.candidate_losses,
        reference_losses_by_model_id=dict(collection_state.reference_losses_by_model_id),
        reference_historical_mean_losses_by_model_id=dict(
            legacy_session.reference_historical_means
        ),
        available_reference_model_ids=tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ),
        current_training_model_id=previous_model_id,
        maximum_reference_mean_loss_increase=legacy_client.distance_threshold,
        minimum_candidate_mean_loss_improvement=config.NEW_MODEL_EARLY_STOPPING_MIN_DELTA,
    )
    assert legacy_client.provisional_model_decisions[-1].accepted is evaluation.candidate_accepted
    resolution_arguments = dict(
        adoption_arguments,
        post_alarm_candidate_loss_evaluation=evaluation,
        pending_assignment_sample_concept_ids=(1,)
        * len(adoption_arguments["pending_assignment_training_samples"]),
    )
    resolution = apply_post_alarm_candidate_validation_resolution(**resolution_arguments)
    return resolution, resolution_arguments, legacy_drift_type, previous_model_id


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("required_validation_sample_count", [2, 4])
def test_observed_loss_sequences_match_actual_legacy_observation(
    class_count, required_validation_sample_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(701)
        observation_arguments, validation_samples, _, _, legacy_client, legacy_session = (
            build_observation_oracle(
                class_count=class_count,
                required_validation_sample_count=required_validation_sample_count,
                monkeypatch=monkeypatch,
            )
        )
        classifiers = (observation_arguments["candidate_classifier"],) + tuple(
            observation_arguments["reference_classifiers_by_model_id"].values()
        )
        for classifier_index, classifier in enumerate(classifiers):
            # 学習modeと評価modeを混在させ、観測がmodeを変えないことも確かめる。
            classifier.train(classifier_index % 2 == 0)
        training_modes = tuple(classifier.training for classifier in classifiers)
        parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter for classifier in classifiers for parameter in classifier.parameters()
        )
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        collection = observation_arguments["post_alarm_candidate_loss_collection"]
        for observation_index, (sample_index, input_features, observed_class_labels) in enumerate(
            validation_samples
        ):
            legacy_result = legacy_client._observe_forward_validation(
                input_features, observed_class_labels, sample_index
            )
            ready_for_acceptance_evaluation = observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                **observation_arguments,
            )
            is_last_observation = observation_index == required_validation_sample_count - 1
            assert ready_for_acceptance_evaluation is is_last_observation
            # 実旧は到達前に0を返し、到達時だけ確定してsessionを破棄する。
            assert (legacy_client._forward_validation is None) is is_last_observation
            if not is_last_observation:
                assert legacy_result == 0
            assert_collected_losses_match_legacy(
                post_alarm_candidate_loss_collection=collection, legacy_session=legacy_session
            )
            assert collection.get_state_snapshot().last_validation_sample_index == sample_index
        assert len(legacy_session.candidate_losses) == required_validation_sample_count
        assert len(set(legacy_session.candidate_losses)) > 1
        # 候補と2つの参照が互いに異なる損失列を持ち、取り違えを検出できる入力であること。
        assert legacy_session.reference_losses[4] != legacy_session.reference_losses[9]
        assert legacy_session.candidate_losses != legacy_session.reference_losses[4]
        assert tuple(classifier.training for classifier in classifiers) == training_modes
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("collection_none", TypeError),
        ("collection_object", TypeError),
        ("collection_subclass", TypeError),
        ("reference_classifiers_list", TypeError),
        ("reference_classifiers_none", TypeError),
        ("reference_classifiers_ordered_dict", TypeError),
        ("features_two_rows", ValueError),
        ("features_one_dimension", ValueError),
        ("features_list", TypeError),
        ("features_feature_count", ValueError),
        ("features_non_finite", ValueError),
        ("labels_shape", ValueError),
        ("labels_out_of_range", ValueError),
        ("labels_list", TypeError),
        ("candidate_object", (TypeError, ValueError)),
        ("reference_classifier_object", (TypeError, ValueError)),
        ("sample_index_not_consecutive", ValueError),
        ("sample_index_repeated", ValueError),
        ("sample_index_bool", TypeError),
        ("sample_index_subclass", TypeError),
        ("reference_model_missing", ValueError),
        ("reference_model_extra", ValueError),
        ("reference_model_id_bool", TypeError),
        ("collection_already_ready", RuntimeError),
    ],
)
def test_rejected_observation_does_not_change_collection(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        observation_arguments, validation_samples, _, _, _, _ = build_observation_oracle(
            class_count=4,
            required_validation_sample_count=2 if invalid_case == "collection_already_ready" else 4,
            monkeypatch=monkeypatch,
        )
        collection = observation_arguments["post_alarm_candidate_loss_collection"]
        reference_classifiers_by_model_id = observation_arguments[
            "reference_classifiers_by_model_id"
        ]
        # 1件目を正常に観測し、拒否で既存の系列が変わらないことを観測する。
        first_sample_index, first_input_features, first_observed_class_labels = validation_samples[
            0
        ]
        observe_post_alarm_candidate_validation_sample(
            sample_index=first_sample_index,
            input_features=first_input_features,
            observed_class_labels=first_observed_class_labels,
            **observation_arguments,
        )
        sample_index, input_features, observed_class_labels = validation_samples[1]
        if invalid_case == "collection_already_ready":
            observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                **observation_arguments,
            )
            sample_index += 1
        forward_call_count_before_rejection = None
        if invalid_case in ("features_two_rows", "features_one_dimension"):
            # 1標本契約の違反は、どの分類器のforwardよりも前に拒否する。
            forward_call_count_before_rejection = []
            original_evaluation = observation_module.evaluate_classifier_per_sample_bounded_losses

            def count_then_evaluate(**arguments):
                forward_call_count_before_rejection.append(1)
                return original_evaluation(**arguments)

            monkeypatch.setattr(
                observation_module,
                "evaluate_classifier_per_sample_bounded_losses",
                count_then_evaluate,
            )
        if invalid_case == "collection_none":
            observation_arguments["post_alarm_candidate_loss_collection"] = None
        elif invalid_case == "collection_object":
            observation_arguments["post_alarm_candidate_loss_collection"] = object()
        elif invalid_case == "collection_subclass":
            observation_arguments["post_alarm_candidate_loss_collection"] = type(
                "CollectionSubclass", (PostAlarmCandidateLossCollection,), {}
            )(
                candidate_model_training_and_acceptance_settings=make_acceptance_settings(
                    required_validation_sample_count=4
                ),
                proposal_sample_index=sample_index - 1,
                reference_model_ids=tuple(reference_classifiers_by_model_id),
            )
        elif invalid_case == "reference_classifiers_list":
            observation_arguments["reference_classifiers_by_model_id"] = list(
                reference_classifiers_by_model_id.items()
            )
        elif invalid_case == "reference_classifiers_none":
            observation_arguments["reference_classifiers_by_model_id"] = None
        elif invalid_case == "reference_classifiers_ordered_dict":
            observation_arguments["reference_classifiers_by_model_id"] = OrderedDict(
                reference_classifiers_by_model_id
            )
        elif invalid_case == "features_two_rows":
            input_features = torch.cat([input_features] * 2)
            observed_class_labels = torch.cat([observed_class_labels] * 2)
        elif invalid_case == "features_one_dimension":
            input_features = input_features.reshape(-1)
        elif invalid_case == "features_list":
            input_features = input_features.tolist()
        elif invalid_case == "features_feature_count":
            input_features = torch.ones((1, 3))
        elif invalid_case == "features_non_finite":
            input_features = torch.tensor([[float("nan"), 0.0]])
        elif invalid_case == "labels_shape":
            observed_class_labels = observed_class_labels.reshape(-1)
        elif invalid_case == "labels_out_of_range":
            observed_class_labels = torch.tensor([[4.0]])
        elif invalid_case == "labels_list":
            observed_class_labels = observed_class_labels.tolist()
        elif invalid_case == "candidate_object":
            observation_arguments["candidate_classifier"] = object()
        elif invalid_case == "reference_classifier_object":
            observation_arguments["reference_classifiers_by_model_id"] = {
                **reference_classifiers_by_model_id,
                9: object(),
            }
        elif invalid_case == "sample_index_not_consecutive":
            sample_index += 1
        elif invalid_case == "sample_index_repeated":
            sample_index -= 1
        elif invalid_case == "sample_index_bool":
            sample_index = True
        elif invalid_case == "sample_index_subclass":
            sample_index = IntSubclass(sample_index)
        elif invalid_case == "reference_model_missing":
            observation_arguments["reference_classifiers_by_model_id"] = {
                4: reference_classifiers_by_model_id[4]
            }
        elif invalid_case == "reference_model_extra":
            observation_arguments["reference_classifiers_by_model_id"] = {
                **reference_classifiers_by_model_id,
                12: reference_classifiers_by_model_id[4],
            }
        elif invalid_case == "reference_model_id_bool":
            observation_arguments["reference_classifiers_by_model_id"] = {
                4: reference_classifiers_by_model_id[4],
                True: reference_classifiers_by_model_id[9],
            }
        previous_collection_state = collection.get_state_snapshot()
        with pytest.raises(expected_exception):
            observe_post_alarm_candidate_validation_sample(
                sample_index=sample_index,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                **observation_arguments,
            )
        assert collection.get_state_snapshot() == previous_collection_state
        if forward_call_count_before_rejection is not None:
            assert forward_call_count_before_rejection == []


def test_all_losses_are_evaluated_before_single_collection_update(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        observation_arguments, validation_samples, _, _, _, _ = build_observation_oracle(
            class_count=2, monkeypatch=monkeypatch
        )
        actual_call_order = []
        original_evaluation = observation_module.evaluate_classifier_per_sample_bounded_losses

        def record_then_evaluate(**arguments):
            actual_call_order.append(arguments["classifier"])
            return original_evaluation(**arguments)

        monkeypatch.setattr(
            observation_module,
            "evaluate_classifier_per_sample_bounded_losses",
            record_then_evaluate,
        )
        collection = observation_arguments["post_alarm_candidate_loss_collection"]
        original_observation = collection.observe_losses_after_label_observation

        def record_then_observe(**arguments):
            actual_call_order.append("observe_losses_after_label_observation")
            return original_observation(**arguments)

        monkeypatch.setattr(
            collection, "observe_losses_after_label_observation", record_then_observe
        )
        sample_index, input_features, observed_class_labels = validation_samples[0]
        observe_post_alarm_candidate_validation_sample(
            sample_index=sample_index,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
            **observation_arguments,
        )
        expected_call_order = [
            observation_arguments["candidate_classifier"],
            *observation_arguments["reference_classifiers_by_model_id"].values(),
            "observe_losses_after_label_observation",
        ]
        assert len(actual_call_order) == len(expected_call_order)
        assert all(
            actual is expected for actual, expected in zip(actual_call_order, expected_call_order)
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("historical_mean_case", ["none", "current", "other"])
def test_observation_then_evaluation_and_resolution_match_actual_legacy_observation(
    class_count, historical_mean_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(709)
        (
            observation_arguments,
            validation_samples,
            adoption_arguments,
            shared_optimizer_owners,
            legacy_client,
            legacy_session,
        ) = build_observation_oracle(
            class_count=class_count,
            historical_mean_case=historical_mean_case,
            monkeypatch=monkeypatch,
        )
        resolution, resolution_arguments, legacy_drift_type, previous_model_id = (
            observe_and_resolve_in_both_implementations(
                observation_arguments=observation_arguments,
                validation_samples=validation_samples,
                adoption_arguments=adoption_arguments,
                legacy_client=legacy_client,
                legacy_session=legacy_session,
            )
        )
        if historical_mean_case == "current":
            assert resolution.resolution_outcome == "current_model_maintained"
        elif historical_mean_case == "other":
            assert resolution.resolution_outcome == "held_reference_model_reused"
        else:
            assert resolution.resolution_outcome in (
                "candidate_adopted_as_new_model",
                "candidate_rejected",
            )
        assert (
            LEGACY_ACTION_BY_RESOLUTION_OUTCOME[resolution.resolution_outcome]
            == legacy_client.adaptation_events[-1].action
        )
        assert_resolution_matches_legacy(
            resolution=resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_observed_and_resolved_validation_continues_actual_joint_training(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = adoption_arguments["held_model_training_state_registry"]
        training_sample_store = adoption_arguments["training_sample_store"]
        counts_store = adoption_arguments["model_training_and_assignment_counts_store"]
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
                loss_statistics_store=adoption_arguments["loss_statistics_store"],
                legacy_client=legacy_client,
            )

        # 学習でparameterが変わった後に警報が起きたとして、その時点の値で参照を固定する。
        run_joint_update_in_both_implementations()
        observation_arguments, validation_samples, legacy_session = (
            prepare_validation_session_in_both_implementations(
                adoption_arguments=adoption_arguments,
                legacy_client=legacy_client,
                class_count=class_count,
                required_validation_sample_count=4,
                historical_mean_case="other",
            )
        )
        # 参照モデルの生成は初期化で乱数を消費する（実旧の複製処理も同じ）。
        # 観測・評価・確定・学習が乱数を消費しないことは、生成後の状態から確かめる。
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        resolution, resolution_arguments, legacy_drift_type, previous_model_id = (
            observe_and_resolve_in_both_implementations(
                observation_arguments=observation_arguments,
                validation_samples=validation_samples,
                adoption_arguments=adoption_arguments,
                legacy_client=legacy_client,
                legacy_session=legacy_session,
            )
        )
        assert resolution.resolution_outcome == "held_reference_model_reused"
        assert_resolution_matches_legacy(
            resolution=resolution,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_drift_type=legacy_drift_type,
            previous_model_id=previous_model_id,
        )
        # 確定後: 保留標本を吸収した再利用先を含めて学習を継続する。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
