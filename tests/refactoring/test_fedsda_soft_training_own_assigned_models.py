"""構成`fedsda-soft-training-own-assigned-models`（柔らかい学習の責任度の対象を、そのclientが自分で固く割り当てた
標本を持つモデルと、いま割り当てる先に限った構成）を、設定・読取り・提供の関数・run・掃引の順に確かめる。"""

from dataclasses import replace

import pytest
from test_fedsda_soft_training_assignment_count_aggregation_weights import (
    client_settings,
    make_store,
    observed_samples,
)

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)

NAME = "fedsda-soft-training-own-assigned-models"
OWN = "models_with_own_assigned_samples"
ALL = "all_held_models"


# 設定と読取り（task 1）


def test_default_scope_is_all_held_models_and_is_not_saved():
    for name in ("fedsda", "fedsda-soft-training", "fedsda-training-mass-diagnostics"):
        settings = client_settings(name)
        assert settings.training_responsibility_model_scope == ALL
        assert "training_responsibility_model_scope" not in convert_settings_to_plain_mapping(
            settings
        )


def test_own_scope_is_accepted_with_soft_training_and_saved():
    changed = replace(
        client_settings("fedsda-soft-training"), training_responsibility_model_scope=OWN
    )
    assert changed.training_responsibility_model_scope == OWN
    assert convert_settings_to_plain_mapping(changed)["training_responsibility_model_scope"] == OWN


@pytest.mark.parametrize("value", ["unknown", True, None, 1])
def test_invalid_scope_values_are_rejected(value):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings("fedsda-soft-training"), training_responsibility_model_scope=value)


@pytest.mark.parametrize(
    "name",
    ["fedsda", "fedsda-training-mass-diagnostics", "fedsda-soft-training-single-model-prediction"],
)
def test_own_scope_is_rejected_without_soft_training_sharing_prediction_weights(name):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings(name), training_responsibility_model_scope=OWN)


def test_store_returns_only_model_ids_with_assigned_samples():
    store = make_store(None, with_responsibility_store=False)
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset()
    store.append_model_training_samples(model_id=3, training_samples=())
    store.append_model_training_samples(model_id=5, training_samples=observed_samples(2))
    store.append_model_training_samples(model_id=-1, training_samples=observed_samples(1))
    ids = store.snapshot_model_ids_with_assigned_samples()
    assert ids == frozenset({5, -1})
    assert type(ids) is frozenset
    store.reassign_model_training_samples_id(original_model_id=-1, reassigned_model_id=7)
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset({5, 7})
    store.remap_model_training_sample_collections(model_id_mapping={5: 7, 3: 9})
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset({7})


# 責任度の提供の関数（task 2）


def build_three_model_provider_inputs(monkeypatch):
    """既存の`test_release_responsibilities_use_fresh_local_parameters_and_new_pool`と同じ組立てで、
    真のラベルの確率が0.8・0.2・0.5の3モデルを用意する。"""
    from copy import deepcopy
    from math import log
    from types import SimpleNamespace

    import torch
    from test_joint_model_parameter_update import build_joint_update_oracle_pair

    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )

    batches, _, _ = build_joint_update_oracle_pair(
        class_count=2,
        batch_sample_counts=(4, 4),
        optimizer_variant="sgd",
        monkeypatch=monkeypatch,
    )
    states = [SimpleNamespace(model_id=m, classifier=b.classifier) for m, b in enumerate(batches)]
    states.append(SimpleNamespace(model_id=2, classifier=deepcopy(states[1].classifier)))
    for state, probability in zip(states, (0.8, 0.2, 0.5), strict=True):
        with torch.no_grad():
            state.classifier.classification_layer.weight.zero_()
            state.classifier.classification_layer.bias.fill_(log(probability / (1 - probability)))
    registry = SimpleNamespace(snapshot_ordered_held_model_training_states=lambda: tuple(states))
    observation = ObservedTrainingSample(
        input_features=torch.tensor([[0.0, 0.0]]), observed_class_labels=torch.tensor([[1.0]])
    )
    return registry, observation


def new_controller():
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )

    return FixedShareWeightController(
        prediction_combination_settings=client_settings("fedsda").prediction_combination_settings
    )


def build_provider(registry, controller, store=None):
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    if store is None:
        return build_released_sample_responsibility_provider(
            registry=registry, weight_controller=controller
        )
    return build_released_sample_responsibility_provider(
        registry=registry, weight_controller=controller, own_assignment_sample_store=store
    )


def own_store(model_ids):
    store = make_store(None, with_responsibility_store=False)
    for model_id in model_ids:
        store.append_model_training_samples(model_id=model_id, training_samples=observed_samples(1))
    return store


def test_restricted_responsibilities_renormalize_full_values_over_own_models(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    full_controller, own_controller = new_controller(), new_controller()
    full = build_provider(registry, full_controller)(0, observation)
    restricted = build_provider(registry, own_controller, own_store((0, 1)))(0, observation)
    # (a) 対象外のモデル2は辞書に入らない。
    assert set(restricted) == {0, 1}
    # (b) 全保有モデルで求めた値を、対象の中で正規化し直した値。
    total = full[0] + full[1]
    assert restricted == pytest.approx({0: full[0] / total, 1: full[1] / total}, abs=1e-12)
    assert restricted == pytest.approx({0: 0.8, 1: 0.2}, abs=1e-6)
    # (c) Fixed-Shareの重みは全保有モデルで同期され、制限なしと同じ状態。
    assert own_controller.weights_by_model_id == full_controller.weights_by_model_id
    assert set(own_controller.weights_by_model_id) == {0, 1, 2}


def test_first_assignment_target_is_always_included(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    # (d) モデル2は標本を持たないが、いま割り当てる先なので含まれる。
    restricted = build_provider(registry, new_controller(), own_store((0,)))(2, observation)
    assert set(restricted) == {0, 2}
    assert restricted == pytest.approx({0: 0.8 / 1.3, 2: 0.5 / 1.3}, abs=1e-6)
    only_target = build_provider(registry, new_controller(), own_store(()))(1, observation)
    assert only_target == {1: 1.0}


def test_all_held_models_own_gives_identical_responsibilities(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    full = build_provider(registry, new_controller())(1, observation)
    # (e) 全保有モデルが対象なら、制限なしと==で同じ（保有していないIDが集合にあっても引かない）。
    restricted = build_provider(registry, new_controller(), own_store((0, 1, 2, 99)))(
        1, observation
    )
    assert restricted == full
    assert list(restricted) == list(full)


@pytest.mark.parametrize("store", [object(), {}, 0])
def test_invalid_store_type_is_rejected_when_building(monkeypatch, store):
    registry, _ = build_three_model_provider_inputs(monkeypatch)
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    # (f) 関数を作る時点で拒否する。
    with pytest.raises(TypeError, match="exact ModelTrainingSampleStore"):
        build_released_sample_responsibility_provider(
            registry=registry, weight_controller=new_controller(), own_assignment_sample_store=store
        )
