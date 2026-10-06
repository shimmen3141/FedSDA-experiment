"""独立候補の生成を実旧生成・optimizer・乱数へ対照する。"""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    build_initial_registration_oracle,
    convert_legacy_parameter_name,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.runtime.candidate_classifier_construction import (
    IndependentCandidateTrainingState,
    create_independent_candidate_training_state,
)


def build_candidate_construction_oracle(*, class_count, optimizer_variant, monkeypatch):
    construction_arguments, _, legacy_client, legacy_candidate = build_initial_registration_oracle(
        class_count=class_count,
        held_model_ids=(4,),
        current_model_is_held=True,
        monkeypatch=monkeypatch,
        optimizer_variant=optimizer_variant,
    )
    architecture_reference_classifier = construction_arguments["adopted_candidate_classifier"]
    initial_candidate_parameter_snapshot = snapshot_classifier_parameters(
        classifier=architecture_reference_classifier
    )
    # 構造参照の値を選択値と区別し、参照を初期値に使う誤りを検出する。
    with torch.no_grad():
        for parameter_values in architecture_reference_classifier.parameters():
            parameter_values.add_(0.375)
    parameter_optimizer_settings = (
        SgdParameterOptimizerSettings(learning_rate=0.01)
        if optimizer_variant == "sgd"
        else AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
        )
    )
    legacy_client.model_cls = lambda: ResidualAdapterMLP(input_dim=2, dataset="sine2")
    construction_arguments = dict(
        architecture_reference_classifier=architecture_reference_classifier,
        initial_candidate_parameter_snapshot=dict(
            reversed(tuple(initial_candidate_parameter_snapshot.items()))
        ),
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    return construction_arguments, legacy_client, legacy_candidate.get_params()


def create_legacy_candidate(*, legacy_client, legacy_initial_parameter_snapshot):
    legacy_candidate = BaseClient._new_model(legacy_client)
    legacy_candidate.set_params(legacy_initial_parameter_snapshot)
    legacy_candidate.reset_optimizer(lr=0.01)
    return legacy_candidate


def assert_candidate_matches_legacy(*, candidate_training_state, legacy_candidate):
    assert type(candidate_training_state) is IndependentCandidateTrainingState
    candidate_classifier = candidate_training_state.candidate_classifier
    assert tuple(dict(candidate_classifier.named_parameters())) == tuple(
        convert_legacy_parameter_name(parameter_name)
        for parameter_name, _ in legacy_candidate.named_parameters()
    )
    for parameter_values, expected_parameter_values in zip(
        candidate_classifier.parameters(), legacy_candidate.parameters()
    ):
        assert torch.equal(parameter_values, expected_parameter_values)
        assert parameter_values.grad is None
    for (
        candidate_parameter_optimizer_manager,
        concept_specific_parameters,
        legacy_parameter_optimizer,
    ) in (
        (
            candidate_training_state.candidate_shared_parameter_optimizer_state,
            tuple(candidate_classifier.feature_extractor.parameters()),
            legacy_candidate.backbone.optimizer,
        ),
        (
            candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            tuple(candidate_classifier.residual_adapter.parameters())
            + tuple(candidate_classifier.classification_layer.parameters()),
            legacy_candidate.head_optimizer,
        ),
    ):
        candidate_parameter_optimizer = candidate_parameter_optimizer_manager.parameter_optimizer
        assert type(candidate_parameter_optimizer) is type(legacy_parameter_optimizer)
        assert candidate_parameter_optimizer.defaults == legacy_parameter_optimizer.defaults
        assert candidate_parameter_optimizer.state_dict() == legacy_parameter_optimizer.state_dict()
        assert not candidate_parameter_optimizer.state
        assert len(candidate_parameter_optimizer.param_groups) == 1
        assert all(
            parameter_values is expected_parameter_values
            for parameter_values, expected_parameter_values in zip(
                candidate_parameter_optimizer.param_groups[0]["params"],
                concept_specific_parameters,
                strict=True,
            )
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
def test_candidate_construction_matches_legacy(class_count, optimizer_variant, monkeypatch):
    construction_arguments, legacy_client, legacy_initial_parameter_snapshot = (
        build_candidate_construction_oracle(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
    )
    for parameter_values in construction_arguments["initial_candidate_parameter_snapshot"].values():
        parameter_values.requires_grad_(True)
        parameter_values.grad = torch.full_like(parameter_values, 0.75)
    reference_parameter_values_and_gradients = snapshot_parameter_values_and_gradients(
        tuple(construction_arguments["architecture_reference_classifier"].parameters())
        + tuple(construction_arguments["initial_candidate_parameter_snapshot"].values())
    )
    initial_rng_state = torch.get_rng_state().clone()
    legacy_candidate = create_legacy_candidate(
        legacy_client=legacy_client,
        legacy_initial_parameter_snapshot=legacy_initial_parameter_snapshot,
    )
    expected_rng_state = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng_state)
    candidate_training_state = create_independent_candidate_training_state(**construction_arguments)
    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    assert not torch.equal(torch.get_rng_state(), initial_rng_state)
    assert_candidate_matches_legacy(
        candidate_training_state=candidate_training_state, legacy_candidate=legacy_candidate
    )
    assert_parameter_values_and_gradients_unchanged(reference_parameter_values_and_gradients)
    parameter_storage_addresses = {
        parameter_values.untyped_storage().data_ptr()
        for parameter_values, _, _ in reference_parameter_values_and_gradients
    }
    for parameter_values in candidate_training_state.candidate_classifier.parameters():
        assert parameter_values.untyped_storage().data_ptr() not in parameter_storage_addresses
        parameter_storage_addresses.add(parameter_values.untyped_storage().data_ptr())
    second_candidate_training_state = create_independent_candidate_training_state(
        **construction_arguments
    )
    for parameter_values in second_candidate_training_state.candidate_classifier.parameters():
        assert parameter_values.untyped_storage().data_ptr() not in parameter_storage_addresses
    assert (
        second_candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer
        is not candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer
    )
    assert (
        second_candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer
        is not candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "reference_type",
        "reference_dtype",
        "reference_nonfinite",
        "reference_empty_features",
        "snapshot_type",
        "snapshot_subclass",
        "key_type",
        "missing_key",
        "extra_key",
        "tensor_type",
        "parameter_subclass",
        "shape",
        "dtype",
        "device",
        "nan",
        "inf",
        "sparse",
        "nested",
        "optimizer_type",
        "adam_learning_rate",
        "adam_weight_decay",
        "adam_variant",
        "sgd_learning_rate",
    ],
)
def test_candidate_construction_rejects_invalid_inputs(invalid_case, monkeypatch):
    construction_arguments, _, _ = build_candidate_construction_oracle(
        class_count=4, optimizer_variant="standard", monkeypatch=monkeypatch
    )
    initial_candidate_parameter_snapshot = construction_arguments[
        "initial_candidate_parameter_snapshot"
    ]
    parameter_name = next(iter(initial_candidate_parameter_snapshot))
    parameter_values = initial_candidate_parameter_snapshot[parameter_name]
    if invalid_case == "reference_type":
        construction_arguments["architecture_reference_classifier"] = object()
    elif invalid_case == "reference_empty_features":
        architecture_reference_classifier = construction_arguments[
            "architecture_reference_classifier"
        ]
        architecture_reference_classifier = type(architecture_reference_classifier)(
            model_architecture_settings=architecture_reference_classifier.model_architecture_settings,
            input_feature_count=2,
            hidden_layer_widths=(),
            class_count=4,
        )
        construction_arguments["architecture_reference_classifier"] = (
            architecture_reference_classifier
        )
        construction_arguments["initial_candidate_parameter_snapshot"] = (
            snapshot_classifier_parameters(classifier=architecture_reference_classifier)
        )
    elif invalid_case == "reference_dtype":
        construction_arguments["architecture_reference_classifier"].double()
    elif invalid_case == "reference_nonfinite":
        with torch.no_grad():
            next(construction_arguments["architecture_reference_classifier"].parameters()).fill_(
                float("nan")
            )
    elif invalid_case == "snapshot_type":
        construction_arguments["initial_candidate_parameter_snapshot"] = list(
            initial_candidate_parameter_snapshot.items()
        )
    elif invalid_case == "snapshot_subclass":
        construction_arguments["initial_candidate_parameter_snapshot"] = type(
            "DictSubclass", (dict,), {}
        )(initial_candidate_parameter_snapshot)
    elif invalid_case == "key_type":
        initial_candidate_parameter_snapshot[type("StrSubclass", (str,), {})(parameter_name)] = (
            initial_candidate_parameter_snapshot.pop(parameter_name)
        )
    elif invalid_case == "missing_key":
        del initial_candidate_parameter_snapshot[parameter_name]
    elif invalid_case == "extra_key":
        initial_candidate_parameter_snapshot["unexpected"] = parameter_values
    elif invalid_case == "tensor_type":
        initial_candidate_parameter_snapshot[parameter_name] = parameter_values.tolist()
    elif invalid_case == "parameter_subclass":
        initial_candidate_parameter_snapshot[parameter_name] = torch.nn.Parameter(
            parameter_values.clone()
        )
    elif invalid_case == "shape":
        initial_candidate_parameter_snapshot[parameter_name] = parameter_values.flatten()[:1]
    elif invalid_case == "dtype":
        initial_candidate_parameter_snapshot[parameter_name] = parameter_values.double()
    elif invalid_case == "device":
        initial_candidate_parameter_snapshot[parameter_name] = parameter_values.to("meta")
    elif invalid_case in ("nan", "inf"):
        initial_candidate_parameter_snapshot[parameter_name] = torch.full_like(
            parameter_values, float(invalid_case)
        )
    elif invalid_case == "sparse":
        initial_candidate_parameter_snapshot[parameter_name] = parameter_values.to_sparse()
    elif invalid_case == "nested":
        initial_candidate_parameter_snapshot[parameter_name] = torch.nested.nested_tensor(
            [parameter_values]
        )
    elif invalid_case == "optimizer_type":
        construction_arguments["parameter_optimizer_settings"] = object()
    elif invalid_case.startswith("adam_"):
        object.__setattr__(
            construction_arguments["parameter_optimizer_settings"],
            invalid_case.removeprefix("adam_")
            if invalid_case != "adam_variant"
            else "adam_variant",
            "invalid" if invalid_case == "adam_variant" else float("nan"),
        )
    elif invalid_case == "sgd_learning_rate":
        construction_arguments["parameter_optimizer_settings"] = SgdParameterOptimizerSettings(
            learning_rate=0.01
        )
        object.__setattr__(
            construction_arguments["parameter_optimizer_settings"], "learning_rate", -1
        )
    reference_parameter_values_and_gradients = snapshot_parameter_values_and_gradients(
        tuple(construction_arguments["architecture_reference_classifier"].parameters())
        if invalid_case != "reference_type"
        else ()
    )
    initial_rng_state = torch.get_rng_state().clone()
    with pytest.raises(
        TypeError
        if invalid_case
        in (
            "reference_type",
            "snapshot_type",
            "snapshot_subclass",
            "key_type",
            "tensor_type",
            "parameter_subclass",
            "optimizer_type",
        )
        else ValueError
    ):
        create_independent_candidate_training_state(**construction_arguments)
    assert torch.equal(torch.get_rng_state(), initial_rng_state)
    # NaN参照の値比較はequal_nanで行い、勾配の同一参照も確認する。
    if invalid_case == "reference_nonfinite":
        for (
            parameter_values,
            expected_parameter_values,
            previous_gradients,
        ) in reference_parameter_values_and_gradients:
            torch.testing.assert_close(parameter_values, expected_parameter_values, equal_nan=True)
            assert parameter_values.grad is previous_gradients[0]
    else:
        assert_parameter_values_and_gradients_unchanged(reference_parameter_values_and_gradients)


def test_candidate_training_state_record_contract(monkeypatch):
    construction_arguments, _, _ = build_candidate_construction_oracle(
        class_count=2, optimizer_variant="standard", monkeypatch=monkeypatch
    )
    candidate_training_state = create_independent_candidate_training_state(**construction_arguments)
    assert tuple(parameter_name.name for parameter_name in fields(candidate_training_state)) == (
        "candidate_classifier",
        "candidate_shared_parameter_optimizer_state",
        "candidate_concept_specific_parameter_optimizer_state",
    )
    assert all(parameter_name.kw_only for parameter_name in fields(candidate_training_state))
    with pytest.raises(FrozenInstanceError):
        candidate_training_state.candidate_classifier = construction_arguments[
            "architecture_reference_classifier"
        ]
