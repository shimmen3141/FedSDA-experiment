"""警報時点の参照モデルの固定を、実旧の参照複製とsession開始へ対照する。"""

import random

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    convert_legacy_parameter_name,
)
from test_adopted_candidate_local_adoption import build_local_adoption_oracle
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.runtime.post_alarm_reference_model_fixation import (
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
        assert type(reference_classifier) is ModelArchitectureClassifier
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
