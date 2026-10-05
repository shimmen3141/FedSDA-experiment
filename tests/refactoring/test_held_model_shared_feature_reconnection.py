"""保有モデルの共有再接続を実旧の選択・参照・optimizerへ照合する。"""

import random
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    build_attachment_classifier,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_drift_experiment.clients.shared_backbone import (
    _SharedRepresentationFedSDAClientMixin,
)
from federated_drift_experiment.data.specs import DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP, SharedFeatureBackbone
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.training.held_model_shared_feature_reconnection import (
    HeldModelOptimizerBinding,
    reconnect_held_models_to_shared_feature_extractor,
)
from federated_learning_experiments.learning.training.held_model_training_binding import (
    HeldModelTrainingBinding,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


class ModelIdSubclass(int):
    pass


class HeldModelOptimizerBindingSubclass(HeldModelOptimizerBinding):
    pass


def build_held_model_optimizer_binding(
    model_id, *, shared_feature_extractor=None, shared_optimizer_state=None, classifier=None
):
    if classifier is None:
        classifier = build_attachment_classifier()
    if shared_feature_extractor is not None:
        classifier.attach_shared_feature_extractor(
            shared_feature_extractor=shared_feature_extractor
        )
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
    )
    return HeldModelOptimizerBinding(
        model_id=model_id,
        classifier=classifier,
        shared_parameter_optimizer_state=shared_optimizer_state
        if shared_optimizer_state is not None
        else ParameterOptimizerState(
            parameters=tuple(classifier.feature_extractor.parameters()),
            optimizer_settings=optimizer_settings,
        ),
        concept_specific_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(classifier.residual_adapter.parameters())
            + tuple(classifier.classification_layer.parameters()),
            optimizer_settings=optimizer_settings,
        ),
    )


def snapshot_reconnection_input_states(previous_bindings):
    parameter_snapshots = tuple(
        (
            binding.classifier.feature_extractor,
            snapshot_parameter_values_and_gradients(binding.classifier.parameters()),
            binding.classifier.residual_adapter,
            binding.classifier.classification_layer,
            binding.classifier.output_activation,
            binding.classifier.class_count,
        )
        for binding in previous_bindings
    )
    optimizer_snapshots = tuple(
        tuple(
            (
                optimizer_state.parameter_optimizer,
                deepcopy(optimizer_state.parameter_optimizer.state_dict()),
            )
            for optimizer_state in (
                binding.shared_parameter_optimizer_state,
                binding.concept_specific_parameter_optimizer_state,
            )
        )
        for binding in previous_bindings
    )
    return parameter_snapshots, optimizer_snapshots


def assert_reconnection_input_states_unchanged(
    previous_bindings, parameter_snapshots, optimizer_snapshots
):
    for binding_index, binding in enumerate(previous_bindings):
        assert binding.classifier.feature_extractor is parameter_snapshots[binding_index][0]
        assert binding.classifier.residual_adapter is parameter_snapshots[binding_index][2]
        assert binding.classifier.classification_layer is parameter_snapshots[binding_index][3]
        assert binding.classifier.output_activation is parameter_snapshots[binding_index][4]
        assert binding.classifier.class_count == parameter_snapshots[binding_index][5]
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots[binding_index][1])
        for optimizer_state, (previous_optimizer, previous_state_dict) in zip(
            (
                binding.shared_parameter_optimizer_state,
                binding.concept_specific_parameter_optimizer_state,
            ),
            optimizer_snapshots[binding_index],
            strict=True,
        ):
            assert optimizer_state.parameter_optimizer is previous_optimizer
            torch.testing.assert_close(
                previous_optimizer.state_dict(), previous_state_dict, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "model_ids_in_order,expected_source_model_id",
    [
        ((), None),
        ((0,), 0),
        ((-8,), -8),
        ((7, -2, 3, 0), 0),
        ((5, 1, -7), 1),
        ((-7, -4, -9), -7),
        ((2**90, -1, 2**90 - 1), 2**90 - 1),
    ],
)
@pytest.mark.parametrize("initially_shared", [False, True])
def test_reconnection_source_and_optimizer_states_match_legacy(
    model_ids_in_order, expected_source_model_id, initially_shared, monkeypatch
):
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset: DatasetSpec(
            input_dim=2, num_concepts=2, num_classes=2, hidden_dims=(5, 4), learning_rate=0.01
        ),
    )
    monkeypatch.setattr(config, "SHARED_ADAPTER_RANK", 2)
    monkeypatch.setattr(config, "OPTIMIZER", "adam")
    monkeypatch.setattr(config, "WEIGHT_DECAY", 0.001)
    monkeypatch.setattr(config, "AMSGRAD", False)
    with torch.random.fork_rng(devices=[]):
        previous_bindings = []
        legacy_models = {}
        for model_id in model_ids_in_order:
            binding = build_held_model_optimizer_binding(
                model_id,
                shared_feature_extractor=previous_bindings[0].classifier.feature_extractor
                if previous_bindings and initially_shared
                else None,
                shared_optimizer_state=previous_bindings[0].shared_parameter_optimizer_state
                if previous_bindings and initially_shared
                else None,
            )
            legacy_model = ResidualAdapterMLP(
                input_dim=2,
                dataset="sine2",
                backbone=next(iter(legacy_models.values())).backbone
                if legacy_models and initially_shared
                else None,
            )
            legacy_model.load_state_dict(
                {
                    parameter_name.replace("feature_extractor.hidden_layers.", "backbone.net.")
                    .replace("residual_adapter.feature_compression.", "adapter.down.")
                    .replace("residual_adapter.feature_expansion.", "adapter.up.")
                    .replace("classification_layer.", "head."): parameter_value
                    for parameter_name, parameter_value in binding.classifier.state_dict().items()
                }
            )
            for parameter in tuple(binding.classifier.parameters()) + tuple(
                legacy_model.parameters()
            ):
                parameter.grad = torch.ones_like(parameter)
            binding.shared_parameter_optimizer_state.parameter_optimizer.step()
            binding.concept_specific_parameter_optimizer_state.parameter_optimizer.step()
            legacy_model.backbone.optimizer.step()
            legacy_model.head_optimizer.step()
            previous_bindings.append(binding)
            legacy_models[model_id] = legacy_model
        previous_bindings = tuple(previous_bindings)
        parameter_snapshots, optimizer_snapshots = snapshot_reconnection_input_states(
            previous_bindings
        )
        legacy_client = SimpleNamespace(models=legacy_models)
        python_random_state = random.getstate()
        torch_random_state = torch.get_rng_state()
        _SharedRepresentationFedSDAClientMixin._share_model_backbones(legacy_client)
        result_bindings = reconnect_held_models_to_shared_feature_extractor(
            held_model_optimizer_bindings=previous_bindings
        )
        assert tuple(binding.model_id for binding in result_bindings) == model_ids_in_order
        assert random.getstate() == python_random_state
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        if not previous_bindings:
            assert result_bindings == ()
            return
        source_binding = next(
            binding for binding in previous_bindings if binding.model_id == expected_source_model_id
        )
        for binding_index, binding in enumerate(result_bindings):
            assert binding.classifier is previous_bindings[binding_index].classifier
            assert binding.classifier.residual_adapter is parameter_snapshots[binding_index][2]
            assert binding.classifier.classification_layer is parameter_snapshots[binding_index][3]
            assert binding.classifier.output_activation is parameter_snapshots[binding_index][4]
            assert binding.classifier.class_count == parameter_snapshots[binding_index][5]
            assert (
                binding.classifier.feature_extractor is source_binding.classifier.feature_extractor
            )
            assert (
                binding.shared_parameter_optimizer_state
                is source_binding.shared_parameter_optimizer_state
            )
            assert_parameter_values_and_gradients_unchanged(parameter_snapshots[binding_index][1])
            assert (
                previous_bindings[
                    binding_index
                ].shared_parameter_optimizer_state.parameter_optimizer
                is optimizer_snapshots[binding_index][0][0]
            )
            for previous_optimizer, previous_state_dict in optimizer_snapshots[binding_index]:
                torch.testing.assert_close(
                    previous_optimizer.state_dict(), previous_state_dict, rtol=0, atol=0
                )
            if binding.model_id == expected_source_model_id:
                assert binding is source_binding
                assert (
                    binding.concept_specific_parameter_optimizer_state.parameter_optimizer
                    is optimizer_snapshots[binding_index][1][0]
                )
            else:
                assert binding is not previous_bindings[binding_index]
                assert (
                    binding.concept_specific_parameter_optimizer_state.parameter_optimizer
                    is not optimizer_snapshots[binding_index][1][0]
                )
                assert (
                    binding.concept_specific_parameter_optimizer_state.parameter_optimizer.state
                    == {}
                )
            legacy_model = legacy_models[binding.model_id]
            torch.testing.assert_close(
                binding.shared_parameter_optimizer_state.parameter_optimizer.state_dict(),
                legacy_model.backbone.optimizer.state_dict(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                binding.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
                legacy_model.head_optimizer.state_dict(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "list",
        "record",
        "record_subclass",
        "bool_id",
        "id_subclass",
        "duplicate_id",
        "classifier",
        "duplicate_classifier",
        "shared_owner",
        "concept_owner",
        "shared_parameters",
        "concept_parameters",
        "input_width",
        "layers",
        "float64",
        "meta",
        "connection_input",
        "connection_hidden",
        "parameter_order",
        "shared_concept_parameters",
    ],
)
def test_invalid_tail_rejects_before_any_reconnection(invalid_case):
    with torch.random.fork_rng(devices=[]):
        previous_bindings = tuple(
            build_held_model_optimizer_binding(model_id) for model_id in (4, -7, -8)
        )
        invalid_value = previous_bindings[-1]
        if invalid_case == "list":
            invalid_value = list(previous_bindings)
        elif invalid_case == "record":
            invalid_value = None
        elif invalid_case == "record_subclass":
            invalid_value = HeldModelOptimizerBindingSubclass(
                model_id=-8,
                classifier=invalid_value.classifier,
                shared_parameter_optimizer_state=invalid_value.shared_parameter_optimizer_state,
                concept_specific_parameter_optimizer_state=invalid_value.concept_specific_parameter_optimizer_state,
            )
        elif invalid_case == "bool_id":
            invalid_value = replace(invalid_value, model_id=True)
        elif invalid_case == "id_subclass":
            invalid_value = replace(invalid_value, model_id=ModelIdSubclass(-8))
        elif invalid_case == "duplicate_id":
            invalid_value = replace(invalid_value, model_id=4)
        elif invalid_case == "classifier":
            invalid_value = replace(invalid_value, classifier=None)
        elif invalid_case == "duplicate_classifier":
            invalid_value = replace(previous_bindings[0], model_id=-8)
        elif invalid_case == "shared_owner":
            invalid_value = replace(invalid_value, shared_parameter_optimizer_state=None)
        elif invalid_case == "concept_owner":
            invalid_value = replace(invalid_value, concept_specific_parameter_optimizer_state=None)
        elif invalid_case == "shared_parameters":
            invalid_value = replace(
                invalid_value,
                shared_parameter_optimizer_state=previous_bindings[
                    0
                ].shared_parameter_optimizer_state,
            )
        elif invalid_case == "concept_parameters":
            invalid_value = replace(
                invalid_value,
                concept_specific_parameter_optimizer_state=previous_bindings[
                    0
                ].concept_specific_parameter_optimizer_state,
            )
        elif invalid_case == "input_width":
            invalid_value.classifier.feature_extractor.input_feature_count = 3
        elif invalid_case == "layers":
            invalid_value.classifier.feature_extractor.hidden_layers[1] = torch.nn.Identity()
        elif invalid_case == "float64":
            invalid_value.classifier.feature_extractor.to(dtype=torch.float64)
        elif invalid_case == "meta":
            invalid_value.classifier.feature_extractor.to(device="meta")
        elif invalid_case in ("connection_input", "connection_hidden"):
            invalid_value = build_held_model_optimizer_binding(
                -8,
                classifier=ResidualAdapterClassifier(
                    model_architecture_settings=invalid_value.classifier.model_architecture_settings,
                    input_feature_count=3 if invalid_case == "connection_input" else 2,
                    hidden_layer_widths=(5, 3) if invalid_case == "connection_hidden" else (5, 4),
                    class_count=2,
                ),
            )
        elif invalid_case == "parameter_order":
            invalid_value.concept_specific_parameter_optimizer_state.parameter_optimizer.param_groups[
                0
            ]["params"].reverse()
        elif invalid_case == "shared_concept_parameters":
            invalid_value.classifier.residual_adapter = previous_bindings[
                0
            ].classifier.residual_adapter
            invalid_value = build_held_model_optimizer_binding(
                -8, classifier=invalid_value.classifier
            )
        if invalid_case in ("connection_input", "connection_hidden", "shared_concept_parameters"):
            previous_bindings = previous_bindings[:-1] + (invalid_value,)
        parameter_snapshots, optimizer_snapshots = snapshot_reconnection_input_states(
            previous_bindings
        )
        python_random_state = random.getstate()
        torch_random_state = torch.get_rng_state()
        with pytest.raises(ValueError):
            reconnect_held_models_to_shared_feature_extractor(
                held_model_optimizer_bindings=invalid_value
                if invalid_case == "list"
                else previous_bindings[:-1] + (invalid_value,)
            )
        assert_reconnection_input_states_unchanged(
            previous_bindings, parameter_snapshots, optimizer_snapshots
        )
        assert random.getstate() == python_random_state
        assert torch.equal(torch.get_rng_state(), torch_random_state)


def test_unexpected_reset_failure_preserves_prefix_and_current_attachment(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        previous_bindings = tuple(
            build_held_model_optimizer_binding(model_id) for model_id in (-7, -8, 4, -9)
        )
        parameter_snapshots, optimizer_snapshots = snapshot_reconnection_input_states(
            previous_bindings
        )
        failure = RuntimeError("reset failed")
        monkeypatch.setattr(
            previous_bindings[1].concept_specific_parameter_optimizer_state,
            "reset_parameter_optimizer",
            Mock(side_effect=failure),
        )
        with pytest.raises(RuntimeError) as invalid_value:
            reconnect_held_models_to_shared_feature_extractor(
                held_model_optimizer_bindings=previous_bindings
            )
        assert invalid_value.value is failure
        assert (
            previous_bindings[0].classifier.feature_extractor
            is previous_bindings[2].classifier.feature_extractor
        )
        assert (
            previous_bindings[1].classifier.feature_extractor
            is previous_bindings[2].classifier.feature_extractor
        )
        assert (
            previous_bindings[0].concept_specific_parameter_optimizer_state.parameter_optimizer
            is not optimizer_snapshots[0][1][0]
        )
        assert (
            previous_bindings[1].concept_specific_parameter_optimizer_state.parameter_optimizer
            is optimizer_snapshots[1][1][0]
        )
        assert previous_bindings[3].classifier.feature_extractor is parameter_snapshots[3][0]
        for binding_index, binding in enumerate(previous_bindings):
            assert_parameter_values_and_gradients_unchanged(parameter_snapshots[binding_index][1])
            assert (
                binding.shared_parameter_optimizer_state.parameter_optimizer
                is optimizer_snapshots[binding_index][0][0]
            )
            for previous_optimizer, previous_state_dict in optimizer_snapshots[binding_index]:
                torch.testing.assert_close(
                    previous_optimizer.state_dict(), previous_state_dict, rtol=0, atol=0
                )


def test_reconnection_calls_follow_input_order_and_skip_source(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        previous_bindings = tuple(
            build_held_model_optimizer_binding(model_id) for model_id in (-3, 4, 0, -9)
        )
        events = []
        for binding in previous_bindings:
            monkeypatch.setattr(
                binding.classifier,
                "attach_shared_feature_extractor",
                Mock(
                    side_effect=lambda *, shared_feature_extractor, binding=binding: (
                        events.append((binding.model_id, "attach")),
                        ResidualAdapterClassifier.attach_shared_feature_extractor(
                            binding.classifier, shared_feature_extractor=shared_feature_extractor
                        ),
                    )[1]
                ),
            )
            monkeypatch.setattr(
                binding.concept_specific_parameter_optimizer_state,
                "reset_parameter_optimizer",
                Mock(
                    side_effect=lambda binding=binding: (
                        events.append((binding.model_id, "reset")),
                        ParameterOptimizerState.reset_parameter_optimizer(
                            binding.concept_specific_parameter_optimizer_state
                        ),
                    )[1]
                ),
            )
        reconnect_held_models_to_shared_feature_extractor(
            held_model_optimizer_bindings=previous_bindings
        )
        assert events == [
            (-3, "attach"),
            (-3, "reset"),
            (4, "attach"),
            (4, "reset"),
            (-9, "attach"),
            (-9, "reset"),
        ]


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("initially_shared", [False, True])
def test_reconnection_result_bindings_match_legacy_joint_training(
    class_count, optimizer_variant, initially_shared, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(829)
        training_batches, previous_optimizer, legacy_client = build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(2, 5),
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        if not initially_shared:
            legacy_client.models[-7].attach_backbone(
                SharedFeatureBackbone(input_dim=2, hidden_dims=(5, 4))
            )
            shared_feature_extractor = SharedFeatureExtractor(
                input_feature_count=2, hidden_layer_widths=(5, 4)
            )
            shared_feature_extractor.hidden_layers.load_state_dict(
                legacy_client.models[-7].backbone.net.state_dict()
            )
            training_batches[1].classifier.attach_shared_feature_extractor(
                shared_feature_extractor=shared_feature_extractor
            )
        optimizer_settings = (
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        )
        shared_optimizer_state = ParameterOptimizerState(
            parameters=tuple(training_batches[0].classifier.feature_extractor.parameters()),
            optimizer_settings=optimizer_settings,
        )
        previous_bindings = (
            HeldModelOptimizerBinding(
                model_id=4,
                classifier=training_batches[0].classifier,
                shared_parameter_optimizer_state=shared_optimizer_state,
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(training_batches[0].classifier.residual_adapter.parameters())
                    + tuple(training_batches[0].classifier.classification_layer.parameters()),
                    optimizer_settings=optimizer_settings,
                ),
            ),
            HeldModelOptimizerBinding(
                model_id=-7,
                classifier=training_batches[1].classifier,
                shared_parameter_optimizer_state=shared_optimizer_state
                if initially_shared
                else ParameterOptimizerState(
                    parameters=tuple(training_batches[1].classifier.feature_extractor.parameters()),
                    optimizer_settings=optimizer_settings,
                ),
                concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                    parameters=tuple(training_batches[1].classifier.residual_adapter.parameters())
                    + tuple(training_batches[1].classifier.classification_layer.parameters()),
                    optimizer_settings=optimizer_settings,
                ),
            ),
        )
        for binding in previous_bindings:
            legacy_model = legacy_client.models[binding.model_id]
            for parameter in tuple(binding.classifier.parameters()) + tuple(
                legacy_model.parameters()
            ):
                parameter.grad = torch.ones_like(parameter)
            binding.shared_parameter_optimizer_state.parameter_optimizer.step()
            binding.concept_specific_parameter_optimizer_state.parameter_optimizer.step()
            legacy_model.backbone.optimizer.step()
            legacy_model.head_optimizer.step()
        training_batches = tuple(
            replace(
                training_batch,
                concept_specific_parameter_optimizer=binding.concept_specific_parameter_optimizer_state.parameter_optimizer,
            )
            for training_batch, binding in zip(training_batches, previous_bindings, strict=True)
        )
        parameter_snapshots = tuple(
            snapshot_parameter_values_and_gradients(
                binding.classifier.feature_extractor.parameters()
            )
            for binding in previous_bindings
        )
        optimizer_snapshots = tuple(
            (
                binding.shared_parameter_optimizer_state.parameter_optimizer,
                deepcopy(binding.shared_parameter_optimizer_state.parameter_optimizer.state_dict()),
            )
            for binding in previous_bindings
        )
        previous_concept_optimizer_snapshots = tuple(
            (
                binding.concept_specific_parameter_optimizer_state.parameter_optimizer,
                deepcopy(
                    binding.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
                ),
            )
            for binding in previous_bindings
        )
        _SharedRepresentationFedSDAClientMixin._share_model_backbones(legacy_client)
        result_bindings = reconnect_held_models_to_shared_feature_extractor(
            held_model_optimizer_bindings=tuple(reversed(previous_bindings))
        )
        assert tuple(binding.model_id for binding in result_bindings) == (-7, 4)
        bindings_by_model_id = {binding.model_id: binding for binding in result_bindings}
        assert bindings_by_model_id[4] is previous_bindings[0]
        for binding_index, binding in enumerate(previous_bindings):
            for previous_optimizer, previous_state_dict in (
                optimizer_snapshots[binding_index],
                previous_concept_optimizer_snapshots[binding_index],
            ):
                torch.testing.assert_close(
                    previous_optimizer.state_dict(), previous_state_dict, rtol=0, atol=0
                )
            assert (
                bindings_by_model_id[binding.model_id].shared_parameter_optimizer_state
                is shared_optimizer_state
            )
        participating_training_batches = []
        for model_id, training_batch in zip((4, -7), training_batches, strict=True):
            binding = HeldModelTrainingBinding(
                model_id=model_id,
                classifier=bindings_by_model_id[model_id].classifier,
                concept_specific_parameter_optimizer=bindings_by_model_id[
                    model_id
                ].concept_specific_parameter_optimizer_state.parameter_optimizer,
            )
            participating_training_batches.append(
                replace(
                    training_batch,
                    concept_specific_parameter_optimizer=binding.concept_specific_parameter_optimizer,
                )
            )
        participating_training_batches = tuple(participating_training_batches)
        assert_joint_update_states_equal(
            participating_training_batches=participating_training_batches,
            shared_parameter_optimizer=shared_optimizer_state.parameter_optimizer,
            legacy_client=legacy_client,
        )
        local_training_settings = LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        )
        for update_shared_features in (True, False, True):
            legacy_loss = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            new_loss = perform_joint_model_parameter_update(
                local_training_settings=local_training_settings,
                shared_feature_extractor=bindings_by_model_id[4].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer_state.parameter_optimizer,
                participating_training_batches=participating_training_batches,
                update_shared_features=update_shared_features,
            )
            assert new_loss == legacy_loss
            assert_joint_update_states_equal(
                participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_optimizer_state.parameter_optimizer,
                legacy_client=legacy_client,
            )
            assert (
                training_batches[1].concept_specific_parameter_optimizer
                is previous_concept_optimizer_snapshots[1][0]
            )
            torch.testing.assert_close(
                previous_concept_optimizer_snapshots[1][0].state_dict(),
                previous_concept_optimizer_snapshots[1][1],
                rtol=0,
                atol=0,
            )
            if not initially_shared:
                assert (
                    bindings_by_model_id[-7].shared_parameter_optimizer_state
                    is not previous_bindings[1].shared_parameter_optimizer_state
                )
                assert_parameter_values_and_gradients_unchanged(parameter_snapshots[1])
                torch.testing.assert_close(
                    optimizer_snapshots[1][0].state_dict(),
                    optimizer_snapshots[1][1],
                    rtol=0,
                    atol=0,
                )


def test_optimizer_binding_is_frozen_but_borrows_live_objects():
    with torch.random.fork_rng(devices=[]):
        binding = build_held_model_optimizer_binding(4)
        with pytest.raises(FrozenInstanceError):
            binding.model_id = 5
        assert reconnect_held_models_to_shared_feature_extractor(
            held_model_optimizer_bindings=(binding,)
        ) == (binding,)
