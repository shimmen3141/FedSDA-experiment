"""警報時点の参照モデルの固定を、実旧の参照複製とsession開始へ対照する。"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
    convert_legacy_parameter_name,
)
from test_adopted_candidate_local_adoption import build_local_adoption_oracle
from test_joint_model_parameter_update import assert_nested_state_equal, run_legacy_joint_update
from test_post_alarm_candidate_validation_resolution import assert_resolution_matches_legacy
from test_post_alarm_candidate_validation_sample_observation import (
    LAST_VALIDATION_SAMPLE_INDEX,
    make_acceptance_settings,
    observe_and_resolve_in_both_implementations,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
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
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.participating_model_training_batch import (
    ParticipatingModelTrainingBatch,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import (
    FixedPostAlarmReferenceModels,
    fix_reference_models_at_alarm,
)

# 統計の与え方。モデルIDごとの(全体件数, 保存平均)。Noneは統計未登録。
STATISTICS_CASES = {
    "both_have_history": {4: (5, 0.4), 9: (3, 0.2)},
    "single_observation_is_excluded_and_zero_mean_is_kept": {4: (1, 0.9), 9: (2, 0.0)},
    "no_observation_and_single_observation": {4: (0, 0.0), 9: (1, 0.5)},
    "unregistered_statistics": {4: None, 9: (7, 1.0)},
}


def set_overall_loss_statistics_in_both_implementations(
    *, loss_statistics_store, legacy_client, statistics_by_model_id
):
    for model_id, statistics in statistics_by_model_id.items():
        if statistics is None:
            # 統計を両実装で保有していないIDへ退避し、未登録のモデルにする。
            loss_statistics_store.reassign_model_loss_statistics_id(
                original_model_id=model_id, reassigned_model_id=999
            )
            legacy_client.model_stats[999] = legacy_client.model_stats.pop(model_id)
            continue
        observed_loss_count, mean_loss = statistics
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=observed_loss_count,
                    mean_loss=mean_loss,
                    sum_squared_loss_deviations=0.0,
                )
            ),
        )
        # 旧の比較helperが読むため、クラス別統計は空で持たせる（新の統計もクラス別は空）。
        legacy_client.model_stats[model_id] = {
            "n": observed_loss_count,
            "mean": mean_loss,
            "M2": 0.0,
            "class_stats": {},
        }


def build_fixation_oracle(
    *, class_count, monkeypatch, held_model_ids=(4, 9), optimizer_variant="standard"
):
    adoption_arguments, shared_optimizer_owners, legacy_client, _ = build_local_adoption_oracle(
        class_count=class_count,
        held_model_ids=held_model_ids,
        optimizer_variant=optimizer_variant,
        monkeypatch=monkeypatch,
    )
    # 実旧のモデル生成。実験では設定の既定値から同じ構造を作る。
    legacy_client.model_cls = lambda: ResidualAdapterMLP(input_dim=2, dataset="sine2")
    fixation_arguments = dict(
        held_model_training_state_registry=adoption_arguments["held_model_training_state_registry"],
        loss_statistics_store=adoption_arguments["loss_statistics_store"],
    )
    return fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client


def begin_forward_validation_in_legacy_client(*, legacy_client, adoption_arguments):
    """実旧のsession開始を、候補の学習だけ無効化して実行する。"""
    legacy_client._forward_validation = None
    legacy_client.phase_seconds = defaultdict(float)
    legacy_client.forward_validation_samples = 4
    legacy_client._train_new_model = lambda model, input_features, observed_class_labels: None
    legacy_client._begin_forward_validation(
        adoption_arguments["initial_statistics_input_features"],
        adoption_arguments["initial_statistics_observed_class_labels"],
        [
            (training_sample.input_features, training_sample.observed_class_labels, 1)
            for training_sample in adoption_arguments["pending_assignment_training_samples"]
        ],
        legacy_client.models[legacy_client.current_model_id].get_params(),
        LAST_VALIDATION_SAMPLE_INDEX - 4,
        35,
        None,
    )
    return legacy_client._forward_validation


def assert_fixed_references_match_legacy(
    *, fixed_reference_models, legacy_reference_models, registry, input_features
):
    reference_classifiers_by_model_id = fixed_reference_models.reference_classifiers_by_model_id
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    assert type(reference_classifiers_by_model_id) is dict
    assert tuple(reference_classifiers_by_model_id) == tuple(legacy_reference_models)
    assert tuple(reference_classifiers_by_model_id) == tuple(
        state.model_id for state in held_model_training_states
    )
    held_parameter_storage_addresses = {
        parameter.data_ptr()
        for state in held_model_training_states
        for parameter in state.classifier.parameters()
    }
    reference_parameter_storage_addresses = set()
    for state in held_model_training_states:
        reference_classifier = reference_classifiers_by_model_id[state.model_id]
        legacy_reference_model = legacy_reference_models[state.model_id]
        assert type(reference_classifier) is ResidualAdapterClassifier
        assert reference_classifier is not state.classifier
        assert reference_classifier.feature_extractor is not state.classifier.feature_extractor
        legacy_parameters = dict(legacy_reference_model.named_parameters())
        assert tuple(
            parameter_name for parameter_name, _ in reference_classifier.named_parameters()
        ) == tuple(
            convert_legacy_parameter_name(parameter_name) for parameter_name in legacy_parameters
        )
        for (_, parameter), legacy_parameter, held_parameter in zip(
            reference_classifier.named_parameters(),
            legacy_parameters.values(),
            state.classifier.parameters(),
        ):
            assert torch.equal(parameter, legacy_parameter)
            assert torch.equal(parameter, held_parameter)
            assert parameter.requires_grad is legacy_parameter.requires_grad
            assert parameter.grad is None and legacy_parameter.grad is None
            assert parameter.data_ptr() not in held_parameter_storage_addresses
            # 参照どうしも実体を共有しない。
            assert parameter.data_ptr() not in reference_parameter_storage_addresses
            reference_parameter_storage_addresses.add(parameter.data_ptr())
        assert reference_classifier.training is legacy_reference_model.training
        with torch.no_grad():
            assert torch.equal(
                reference_classifier(input_features), legacy_reference_model(input_features)
            )
    feature_extractors = [
        reference_classifier.feature_extractor
        for reference_classifier in reference_classifiers_by_model_id.values()
    ]
    assert len({id(feature_extractor) for feature_extractor in feature_extractors}) == len(
        feature_extractors
    )
    legacy_backbones = [model.backbone for model in legacy_reference_models.values()]
    assert len({id(backbone) for backbone in legacy_backbones}) == len(legacy_backbones)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("held_model_ids", [(4,), (4, 9), (9, -3, 4)])
def test_fixed_references_match_actual_legacy_reference_snapshots_and_random_state(
    class_count, held_model_ids, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(801)
        fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client = (
            build_fixation_oracle(
                class_count=class_count, held_model_ids=held_model_ids, monkeypatch=monkeypatch
            )
        )
        registry = fixation_arguments["held_model_training_state_registry"]
        loss_statistics_store = fixation_arguments["loss_statistics_store"]
        input_features = adoption_arguments["initial_statistics_input_features"]
        previous_held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        held_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for state in previous_held_model_training_states
            for parameter in state.classifier.parameters()
        )
        optimizer_owners = tuple(
            state.concept_specific_parameter_optimizer_state
            for state in previous_held_model_training_states
        ) + tuple(shared_optimizer_owners)
        previous_optimizers = tuple(
            (owner, owner.parameter_optimizer, deepcopy(owner.parameter_optimizer.state_dict()))
            for owner in optimizer_owners
        )
        previous_loss_statistics = loss_statistics_store.get_state_snapshot()
        initial_torch_random_state = torch.get_rng_state().clone()
        other_random_states = (random.getstate(), np.random.get_state())

        legacy_reference_models = legacy_client._snapshot_reference_models()
        legacy_torch_random_state = torch.get_rng_state().clone()
        # 同じ乱数状態から新の固定を行い、モデル生成による消費が実旧と同じであることを確かめる。
        torch.set_rng_state(initial_torch_random_state)
        fixed_reference_models = fix_reference_models_at_alarm(**fixation_arguments)

        assert type(fixed_reference_models) is FixedPostAlarmReferenceModels
        assert not torch.equal(legacy_torch_random_state, initial_torch_random_state)
        assert torch.equal(torch.get_rng_state(), legacy_torch_random_state)
        assert random.getstate() == other_random_states[0]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == other_random_states[1][0]
        assert np.array_equal(numpy_state[1], other_random_states[1][1])
        assert numpy_state[2:] == other_random_states[1][2:]
        assert_fixed_references_match_legacy(
            fixed_reference_models=fixed_reference_models,
            legacy_reference_models=legacy_reference_models,
            registry=registry,
            input_features=input_features,
        )
        # 保有モデル・optimizer・統計は変わらない。
        current_held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        assert len(current_held_model_training_states) == len(previous_held_model_training_states)
        assert all(
            state is previous_state
            for state, previous_state in zip(
                current_held_model_training_states, previous_held_model_training_states
            )
        )
        assert_parameter_values_and_gradients_unchanged(held_parameter_snapshots)
        for owner, previous_optimizer, previous_optimizer_state in previous_optimizers:
            assert owner.parameter_optimizer is previous_optimizer
            assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert loss_statistics_store.get_state_snapshot() == previous_loss_statistics
        # 固定後に保有モデルを書き換えても、参照の値は警報時点のまま。
        reference_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for reference_classifier in fixed_reference_models.reference_classifiers_by_model_id.values()
            for parameter in reference_classifier.parameters()
        )
        with torch.no_grad():
            for state in current_held_model_training_states:
                for parameter in state.classifier.parameters():
                    parameter.add_(0.5)
        assert_parameter_values_and_gradients_unchanged(reference_parameter_snapshots)
        with pytest.raises(FrozenInstanceError):
            fixed_reference_models.reference_classifiers_by_model_id = {}


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("statistics_case", list(STATISTICS_CASES))
def test_historical_mean_losses_match_actual_legacy_session_start(
    class_count, statistics_case, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(809)
        fixation_arguments, adoption_arguments, _, legacy_client = build_fixation_oracle(
            class_count=class_count, monkeypatch=monkeypatch
        )
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=fixation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id=STATISTICS_CASES[statistics_case],
        )
        legacy_session = begin_forward_validation_in_legacy_client(
            legacy_client=legacy_client, adoption_arguments=adoption_arguments
        )
        fixed_reference_models = fix_reference_models_at_alarm(**fixation_arguments)
        historical_mean_losses = fixed_reference_models.reference_historical_mean_losses_by_model_id
        assert type(historical_mean_losses) is dict
        assert historical_mean_losses == legacy_session.reference_historical_means
        assert tuple(historical_mean_losses) == tuple(legacy_session.reference_historical_means)
        assert all(type(mean_loss) is float for mean_loss in historical_mean_losses.values())
        expected_model_ids = {
            "both_have_history": (4, 9),
            "single_observation_is_excluded_and_zero_mean_is_kept": (9,),
            "no_observation_and_single_observation": (),
            "unregistered_statistics": (9,),
        }[statistics_case]
        assert tuple(historical_mean_losses) == expected_model_ids
        assert tuple(fixed_reference_models.reference_classifiers_by_model_id) == tuple(
            legacy_session.reference_models
        )


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception"),
    [
        ("registry_none", TypeError),
        ("registry_object", TypeError),
        ("registry_subclass", TypeError),
        ("statistics_store_none", TypeError),
        ("statistics_store_object", TypeError),
        ("statistics_store_subclass", TypeError),
        ("no_held_model", LookupError),
        ("first_model_parameter_non_finite", ValueError),
        ("last_model_parameter_non_finite", ValueError),
        ("shared_parameter_non_finite", ValueError),
    ],
)
def test_rejected_fixation_consumes_no_random_state_and_changes_nothing(
    invalid_case, expected_exception, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        fixation_arguments, _, _, _ = build_fixation_oracle(
            class_count=4,
            held_model_ids=() if invalid_case == "no_held_model" else (4, 9),
            monkeypatch=monkeypatch,
        )
        registry = fixation_arguments["held_model_training_state_registry"]
        loss_statistics_store = fixation_arguments["loss_statistics_store"]
        held_model_training_states = registry.snapshot_ordered_held_model_training_states()
        if invalid_case == "registry_none":
            fixation_arguments["held_model_training_state_registry"] = None
        elif invalid_case == "registry_object":
            fixation_arguments["held_model_training_state_registry"] = object()
        elif invalid_case == "registry_subclass":
            fixation_arguments["held_model_training_state_registry"] = type(
                "OwnerSubclass", (HeldModelTrainingStateRegistry,), {}
            )()
        elif invalid_case == "statistics_store_none":
            fixation_arguments["loss_statistics_store"] = None
        elif invalid_case == "statistics_store_object":
            fixation_arguments["loss_statistics_store"] = object()
        elif invalid_case == "statistics_store_subclass":
            fixation_arguments["loss_statistics_store"] = type(
                "OwnerSubclass", (ModelAndClassLossStatisticsStore,), {}
            )()
        elif invalid_case == "first_model_parameter_non_finite":
            with torch.no_grad():
                held_model_training_states[0].classifier.classification_layer.bias[0] = float("inf")
        elif invalid_case == "last_model_parameter_non_finite":
            with torch.no_grad():
                held_model_training_states[-1].classifier.classification_layer.bias[0] = float(
                    "-inf"
                )
        elif invalid_case == "shared_parameter_non_finite":
            with torch.no_grad():
                next(iter(held_model_training_states[0].classifier.feature_extractor.parameters()))[
                    0, 0
                ] = float("inf")
        held_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for state in held_model_training_states
            for parameter in state.classifier.parameters()
        )
        previous_loss_statistics = loss_statistics_store.get_state_snapshot()
        random_states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        with pytest.raises(expected_exception):
            fix_reference_models_at_alarm(**fixation_arguments)
        # 拒否では分類器を1つも生成せず、torch乱数を消費しない。
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
        assert_parameter_values_and_gradients_unchanged(held_parameter_snapshots)
        assert registry.snapshot_ordered_held_model_training_states() == held_model_training_states
        assert loss_statistics_store.get_state_snapshot() == previous_loss_statistics


def start_validation_in_both_implementations(
    *, fixation_arguments, adoption_arguments, legacy_client, class_count
):
    """実旧はsession開始、新は候補（test側で用意）→参照の固定→損失収集の開始を行う。"""
    registry = fixation_arguments["held_model_training_state_registry"]
    current_model_id = adoption_arguments[
        "current_training_model_assignment"
    ].current_training_model_id
    initial_torch_random_state = torch.get_rng_state().clone()
    legacy_session = begin_forward_validation_in_legacy_client(
        legacy_client=legacy_client, adoption_arguments=adoption_arguments
    )
    legacy_torch_random_state = torch.get_rng_state().clone()
    torch.set_rng_state(initial_torch_random_state)
    # 候補の生成は後続specの責務。ここでは実旧と同じ順（候補→参照）で、
    # 現行モデルと同じ値の独立した候補をtest側で用意する。
    current_classifier = registry.get_held_model_training_state(
        model_id=current_model_id
    ).classifier
    candidate_classifier = ResidualAdapterClassifier(
        model_architecture_settings=current_classifier.model_architecture_settings,
        input_feature_count=current_classifier.feature_extractor.input_feature_count,
        hidden_layer_widths=current_classifier.feature_extractor.hidden_layer_widths,
        class_count=class_count,
    )
    candidate_classifier.load_state_dict(
        snapshot_classifier_parameters(classifier=current_classifier)
    )
    fixed_reference_models = fix_reference_models_at_alarm(**fixation_arguments)
    # 候補の生成と参照の固定を合わせた乱数の消費も、実旧のsession開始と同じになる。
    assert torch.equal(torch.get_rng_state(), legacy_torch_random_state)
    assert (
        fixed_reference_models.reference_historical_mean_losses_by_model_id
        == legacy_session.reference_historical_means
    )
    adoption_arguments.update(
        adopted_candidate_classifier=candidate_classifier,
        candidate_concept_specific_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(candidate_classifier.residual_adapter.parameters())
            + tuple(candidate_classifier.classification_layer.parameters()),
            optimizer_settings=AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
            ),
        ),
        candidate_trained_sample_count=legacy_session.candidate_training_examples,
        candidate_parameter_update_step_count=legacy_session.candidate_optimizer_steps,
    )
    observation_arguments = dict(
        candidate_classifier=candidate_classifier,
        reference_classifiers_by_model_id=fixed_reference_models.reference_classifiers_by_model_id,
        post_alarm_candidate_loss_collection=PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=make_acceptance_settings(
                required_validation_sample_count=legacy_session.target_count
            ),
            proposal_sample_index=legacy_session.proposal_position,
            reference_model_ids=tuple(fixed_reference_models.reference_classifiers_by_model_id),
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
        for sample_offset in range(legacy_session.target_count)
    )
    return fixed_reference_models, observation_arguments, validation_samples, legacy_session


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    ("statistics_by_model_id", "expected_resolution_outcome"),
    [
        ({4: (1, 0.5), 9: (5, 1.0)}, "current_model_maintained"),
        ({4: (5, 1.0), 9: (1, 0.5)}, "held_reference_model_reused"),
        ({4: (1, 0.5), 9: (1, 0.5)}, "candidate_rejected"),
    ],
)
def test_fixation_then_observation_and_resolution_match_actual_legacy_session(
    class_count, statistics_by_model_id, expected_resolution_outcome, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client = (
            build_fixation_oracle(class_count=class_count, monkeypatch=monkeypatch)
        )
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=fixation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id=statistics_by_model_id,
        )
        fixed_reference_models, observation_arguments, validation_samples, legacy_session = (
            start_validation_in_both_implementations(
                fixation_arguments=fixation_arguments,
                adoption_arguments=adoption_arguments,
                legacy_client=legacy_client,
                class_count=class_count,
            )
        )
        assert_fixed_references_match_legacy(
            fixed_reference_models=fixed_reference_models,
            legacy_reference_models=legacy_session.reference_models,
            registry=fixation_arguments["held_model_training_state_registry"],
            input_features=adoption_arguments["initial_statistics_input_features"],
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
        assert resolution.resolution_outcome == expected_resolution_outcome
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
def test_references_stay_fixed_while_held_models_continue_actual_joint_training(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(821)
        fixation_arguments, adoption_arguments, shared_optimizer_owners, legacy_client = (
            build_fixation_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
        registry = fixation_arguments["held_model_training_state_registry"]
        training_sample_store = adoption_arguments["training_sample_store"]
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
            expected_joint_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            actual_joint_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=active,
                shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
                participating_training_batches=tuple(
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
                ),
                update_shared_features=update_shared_features,
            )
            assert actual_joint_loss == expected_joint_loss
            # 実旧の共同学習は学習計数を加算する。新側も同じ量を記録して照合を続ける。
            for training_binding, collection in zip(
                training_bindings, model_training_sample_collections
            ):
                adoption_arguments[
                    "model_training_and_assignment_counts_store"
                ].record_completed_model_training(
                    model_id=training_binding.model_id,
                    trained_sample_count=len(collection.training_samples),
                    parameter_update_step_count=1,
                )
            assert_held_model_states_match_legacy(
                registry=registry,
                shared_optimizer_owners=shared_optimizer_owners[:-1],
                legacy_client=legacy_client,
                input_features=input_features,
            )

        # 学習でparameterが変わった後に警報が起きたとして、その時点の値で参照を固定する。
        run_joint_update_in_both_implementations()
        set_overall_loss_statistics_in_both_implementations(
            loss_statistics_store=fixation_arguments["loss_statistics_store"],
            legacy_client=legacy_client,
            statistics_by_model_id={4: (5, 1.0), 9: (1, 0.5)},
        )
        fixed_reference_models, observation_arguments, validation_samples, legacy_session = (
            start_validation_in_both_implementations(
                fixation_arguments=fixation_arguments,
                adoption_arguments=adoption_arguments,
                legacy_client=legacy_client,
                class_count=class_count,
            )
        )
        reference_classifiers = tuple(
            fixed_reference_models.reference_classifiers_by_model_id.values()
        )
        reference_parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for reference_classifier in reference_classifiers
            for parameter in reference_classifier.parameters()
        )
        # 検証中も保有モデルの学習は続く。参照は固定時の値のまま、実旧の参照とも一致し続ける。
        for _ in range(2):
            run_joint_update_in_both_implementations()
        assert_parameter_values_and_gradients_unchanged(reference_parameter_snapshots)
        for (
            model_id,
            reference_classifier,
        ) in fixed_reference_models.reference_classifiers_by_model_id.items():
            held_classifier = registry.get_held_model_training_state(model_id=model_id).classifier
            assert any(
                not torch.equal(parameter, held_parameter)
                for parameter, held_parameter in zip(
                    reference_classifier.parameters(), held_classifier.parameters()
                )
            )
            for parameter, legacy_parameter in zip(
                reference_classifier.parameters(),
                legacy_session.reference_models[model_id].parameters(),
            ):
                assert torch.equal(parameter, legacy_parameter)
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
        assert torch.equal(torch.get_rng_state(), random_states[0])
        assert random.getstate() == random_states[1]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == random_states[2][0]
        assert np.array_equal(numpy_state[1], random_states[2][1])
        assert numpy_state[2:] == random_states[2][2:]
