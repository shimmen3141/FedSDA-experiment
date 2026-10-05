"""採用候補の共有値反映・接続・個別resetを検証する。"""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    build_attachment_classifier,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import (
    integrate_adopted_candidate_shared_features,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)


def build_candidate_integration_inputs(*, initially_shared=False, hidden_layer_widths=(5, 4)):
    active = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=hidden_layer_widths)
    candidate = build_attachment_classifier(hidden_layer_widths=hidden_layer_widths)
    if initially_shared:
        candidate.attach_shared_feature_extractor(shared_feature_extractor=active)
    owner = ParameterOptimizerState(
        parameters=tuple(candidate.residual_adapter.parameters())
        + tuple(candidate.classification_layer.parameters()),
        optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
        ),
    )
    for parameter in candidate.parameters():
        parameter.grad = torch.ones_like(parameter)
    owner.parameter_optimizer.step()
    return candidate, owner, active


@pytest.mark.parametrize("hidden_layer_widths", [(), (3,), (5, 4)])
@pytest.mark.parametrize("initially_shared", [False, True])
def test_integration_copies_values_preserves_references_and_resets_only_concept_owner(
    hidden_layer_widths, initially_shared
):
    with torch.random.fork_rng(devices=[]):
        candidate, owner, active = build_candidate_integration_inputs(
            initially_shared=initially_shared, hidden_layer_widths=hidden_layer_widths
        )
        source = candidate.feature_extractor
        source_snapshot = snapshot_parameter_values_and_gradients(source.parameters())
        concept_snapshot = snapshot_parameter_values_and_gradients(
            tuple(candidate.residual_adapter.parameters())
            + tuple(candidate.classification_layer.parameters())
        )
        active_parameters = tuple(active.parameters())
        for parameter in active_parameters:
            parameter.grad = torch.full_like(parameter, 2)
        active_gradients = tuple(parameter.grad for parameter in active_parameters)
        previous_optimizer = owner.parameter_optimizer
        previous_state = deepcopy(previous_optimizer.state_dict())
        rng_state = torch.get_rng_state().clone()
        assert (
            integrate_adopted_candidate_shared_features(
                adopted_candidate_classifier=candidate,
                candidate_concept_specific_parameter_optimizer_state=owner,
                active_shared_feature_extractor=active,
            )
            is None
        )
        assert candidate.feature_extractor is active
        assert all(a is b for a, b in zip(active.parameters(), active_parameters))
        for parameter, source_parameter, gradient in zip(
            active.parameters(), source.parameters(), active_gradients
        ):
            assert torch.equal(parameter, source_parameter)
            assert parameter.grad is gradient
            assert torch.equal(gradient, torch.full_like(parameter, 2))
        if not initially_shared:
            assert_parameter_values_and_gradients_unchanged(source_snapshot)
        assert_parameter_values_and_gradients_unchanged(concept_snapshot)
        assert owner.parameter_optimizer is not previous_optimizer
        assert not owner.parameter_optimizer.state
        torch.testing.assert_close(previous_optimizer.state_dict(), previous_state, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng_state)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "candidate_type",
        "owner_type",
        "target_type",
        "source_type",
        "target_dimensions",
        "source_dimensions",
        "target_layer",
        "source_layer",
        "target_dtype",
        "source_dtype",
        "target_meta",
        "source_meta",
        "owner_wrong_parameters",
        "owner_reversed_parameters",
        "concept_dtype",
        "concept_shared_parameter",
    ],
)
def test_invalid_inputs_are_rejected_before_any_mutation(invalid_case):
    with torch.random.fork_rng(devices=[]):
        candidate, owner, active = build_candidate_integration_inputs()
        arguments = dict(
            adopted_candidate_classifier=candidate,
            candidate_concept_specific_parameter_optimizer_state=owner,
            active_shared_feature_extractor=active,
        )
        if invalid_case == "candidate_type":
            arguments["adopted_candidate_classifier"] = object()
        elif invalid_case == "owner_type":
            arguments["candidate_concept_specific_parameter_optimizer_state"] = object()
        elif invalid_case == "target_type":
            arguments["active_shared_feature_extractor"] = object()
        elif invalid_case == "source_type":
            candidate.feature_extractor = torch.nn.Identity()
        elif invalid_case == "target_dimensions":
            active = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(6, 4))
            arguments["active_shared_feature_extractor"] = active
        elif invalid_case == "source_dimensions":
            candidate.feature_extractor.output_feature_count = 99
        elif invalid_case in ("target_layer", "source_layer"):
            (
                active if invalid_case == "target_layer" else candidate.feature_extractor
            ).hidden_layers[1] = torch.nn.Identity()
        elif invalid_case in ("target_dtype", "source_dtype"):
            (active if invalid_case == "target_dtype" else candidate.feature_extractor).double()
        elif invalid_case in ("target_meta", "source_meta"):
            (active if invalid_case == "target_meta" else candidate.feature_extractor).to("meta")
        elif invalid_case == "owner_wrong_parameters":
            arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                ParameterOptimizerState(
                    parameters=tuple(active.parameters()),
                    optimizer_settings=AdamParameterOptimizerSettings(
                        learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
                    ),
                )
            )
        elif invalid_case == "owner_reversed_parameters":
            owner.parameter_optimizer.param_groups[0]["params"].reverse()
        elif invalid_case == "concept_dtype":
            candidate.residual_adapter.double()
        elif invalid_case == "concept_shared_parameter":
            candidate.classification_layer.bias = active.hidden_layers[2].bias
            arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                ParameterOptimizerState(
                    parameters=tuple(candidate.residual_adapter.parameters())
                    + tuple(candidate.classification_layer.parameters()),
                    optimizer_settings=AdamParameterOptimizerSettings(
                        learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
                    ),
                )
            )
        source = candidate.feature_extractor
        snapshots = snapshot_parameter_values_and_gradients(
            tuple(candidate.parameters()) + tuple(active.parameters())
        )
        previous_optimizer = owner.parameter_optimizer
        previous_state = deepcopy(previous_optimizer.state_dict())
        rng_state = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            integrate_adopted_candidate_shared_features(**arguments)
        assert candidate.feature_extractor is source
        assert owner.parameter_optimizer is previous_optimizer
        torch.testing.assert_close(previous_optimizer.state_dict(), previous_state, rtol=0, atol=0)
        assert_parameter_values_and_gradients_unchanged(snapshots)
        assert torch.equal(torch.get_rng_state(), rng_state)


def test_operation_order_and_unexpected_reset_failure_preserve_completed_changes(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        candidate, owner, active = build_candidate_integration_inputs()
        source = candidate.feature_extractor
        events = []
        copy_values = active.copy_parameter_values_from
        attach = candidate.attach_shared_feature_extractor

        def copy_then_record(**kwargs):
            events.append("copy")
            copy_values(**kwargs)

        def attach_then_record(**kwargs):
            events.append("attach")
            attach(**kwargs)

        def fail_reset():
            events.append("reset")
            raise RuntimeError("injected reset failure")

        monkeypatch.setattr(active, "copy_parameter_values_from", copy_then_record)
        monkeypatch.setattr(candidate, "attach_shared_feature_extractor", attach_then_record)
        monkeypatch.setattr(owner, "reset_parameter_optimizer", fail_reset)
        previous_optimizer = owner.parameter_optimizer
        with pytest.raises(RuntimeError, match="injected"):
            integrate_adopted_candidate_shared_features(
                adopted_candidate_classifier=candidate,
                candidate_concept_specific_parameter_optimizer_state=owner,
                active_shared_feature_extractor=active,
            )
        assert events == ["copy", "attach", "reset"]
        assert candidate.feature_extractor is active
        assert owner.parameter_optimizer is previous_optimizer
        for parameter, source_parameter in zip(active.parameters(), source.parameters()):
            assert torch.equal(parameter, source_parameter)


@pytest.mark.parametrize("invalid_side", ["target", "source"])
def test_direct_value_copy_validates_both_sides_before_load(invalid_side, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        source = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(5, 4))
        active = SharedFeatureExtractor(input_feature_count=2, hidden_layer_widths=(5, 4))
        (active if invalid_side == "target" else source).hidden_layers[1] = torch.nn.Identity()
        load = Mock()
        monkeypatch.setattr(active, "load_state_dict", load)
        snapshots = snapshot_parameter_values_and_gradients(
            tuple(source.parameters()) + tuple(active.parameters())
        )
        with pytest.raises(ValueError):
            active.copy_parameter_values_from(source_feature_extractor=source)
        load.assert_not_called()
        assert_parameter_values_and_gradients_unchanged(snapshots)
