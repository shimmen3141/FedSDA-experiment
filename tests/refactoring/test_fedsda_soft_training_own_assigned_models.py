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
