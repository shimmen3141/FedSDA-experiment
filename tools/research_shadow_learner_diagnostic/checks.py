"""影の継続学習・計測・既定の軌跡との一致を検証する。"""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from tools.research_shadow_learner_diagnostic.diagnostic import (
    best_new_model_hypothesis,
    continue_shadow_observation,
)


def test_best_new_model_cost_includes_prefix_switch_and_tie_order():
    args = dict(
        current_model_log_losses=(1.0, 2.0, 3.0, 4.0),
        switching_code_length=5.0,
        new_model_code_lengths_by_change_interval_sample_count={4: 8.0, 2: 5.0},
        new_model_learner_names_by_change_interval_sample_count={
            2: "warm_started_learner",
            4: "reinitialized_learner",
        },
    )
    # k2: (1+2+5)+5=13、k4: 5+8=13。同点は短いk。
    assert best_new_model_hypothesis(args) == (2, "warm_started_learner", 10.0, 13.0)


@pytest.mark.parametrize("processed_alarm", [False, True])
def test_short_window_without_new_hypothesis_does_not_create_shadow(processed_alarm):
    from types import SimpleNamespace
    from unittest.mock import patch
    import federated_learning_experiments.runtime.fedsda_run_client as client_module
    import federated_learning_experiments.runtime.observed_sample_processing as processing
    import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as mdl
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        shadow_learner_diagnostic_policy,
    )

    args = dict(
        current_model_log_losses=(1.0, 2.0),
        held_model_log_losses_by_model_id=(),
        new_model_code_lengths_by_change_interval_sample_count={},
        new_model_learner_names_by_change_interval_sample_count={},
        switching_code_length=5.0,
        change_interval_grid_sample_count=5,
    )
    assert (
        mdl.select_assignment_by_minimum_description_length(**args).selected_hypothesis
        == "current_model"
    )
    assert best_new_model_hypothesis(args) is None

    def select(**kw):
        return mdl.select_assignment_by_minimum_description_length(**args)

    def process(**kw):
        processing.select_alarm_interval_assignment()
        return SimpleNamespace(
            alarm_occurrence_handling=object() if processed_alarm else None
        )

    with (
        patch.object(client_module, "process_observed_sample", process),
        patch.object(processing, "select_alarm_interval_assignment", select),
    ):
        diagnostics = {}
        with shadow_learner_diagnostic_policy(diagnostics=diagnostics, trace=[]):
            client_module.process_observed_sample(
                client_id=0, indexed_observation=SimpleNamespace(sample_index=2)
            )
        assert diagnostics["episodes"] == []


@pytest.mark.parametrize("class_count", [2, 3])
def test_shadow_continues_real_prequential_optimizer_without_mutating_current_or_rng(
    class_count,
):
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )
    from federated_learning_experiments.learning.models.model_architecture_settings import (
        ModelArchitectureSettings,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
        AdamParameterOptimizerSettings,
    )
    from federated_learning_experiments.learning.training.parameter_optimizer_construction import (
        create_parameter_optimizer,
    )
    from federated_learning_experiments.learning.training.prequential_code_length import (
        _accumulate_prequential_code_length,
    )
    from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
        evaluate_classifier_per_sample_log_losses,
    )
    from federated_learning_experiments.learning.models.model_computation_measurement import (
        measure_model_computation,
    )

    torch.manual_seed(7)
    current = ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=class_count,
    )
    shadow, reference = deepcopy(current), deepcopy(current)
    settings = AdamParameterOptimizerSettings(
        learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
    )
    optimizer = create_parameter_optimizer(
        parameters=tuple(
            p
            for n, p in shadow.named_parameters()
            if not n.startswith("feature_extractor.")
        ),
        optimizer_settings=settings,
    )
    x = torch.tensor([[0.1, 0.7], [0.5, -0.2], [0.3, 0.8]])
    y = torch.tensor([[0.0], [1.0], [float(class_count - 1)]])
    current_state = deepcopy(current.state_dict())
    initial_shared = deepcopy(shadow.feature_extractor.state_dict())
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    measured_losses = []
    with measure_model_computation() as meter:
        for i in range(3):
            with torch.no_grad():
                expected_current = float(
                    evaluate_classifier_per_sample_log_losses(
                        classifier=current,
                        input_features=x[i : i + 1],
                        observed_class_labels=y[i : i + 1],
                    ).sum()
                )
            a, b = continue_shadow_observation(
                learner=shadow,
                optimizer=optimizer,
                current_classifier=current,
                input_features=x[i : i + 1],
                observed_class_labels=y[i : i + 1],
            )
            assert a == expected_current
            measured_losses.append(b)
    assert random.getstate() == random_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    expected_code = _accumulate_prequential_code_length(
        learner=reference,
        input_features=x,
        observed_class_labels=y,
        parameter_optimizer_settings=settings,
    )
    assert sum(measured_losses) == pytest.approx(expected_code, abs=1e-10)
    for k, v in shadow.state_dict().items():
        assert torch.equal(v, reference.state_dict()[k])
    for k, v in current_state.items():
        assert torch.equal(v, current.state_dict()[k])
    for k, v in initial_shared.items():
        assert torch.equal(v, shadow.feature_extractor.state_dict()[k])
    assert (
        meter.get_model_computation_counts().concept_specific_parameter_optimizer_step_count
        == 3
    )


@pytest.mark.parametrize("dataset", ["sea4", "sine2"])
def test_full_run_predictions_training_and_accounted_computation_match_product(dataset):
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_shadow_learner_diagnostic.run import execute_diagnostic_job

    job = dict(dataset=dataset, interval=100, seed=1, sample_count=600)
    baseline = _execute_job(
        {
            **job,
            "variant": "product-default",
            "check": False,
            "record_prediction_period_metrics": True,
        }
    )
    measured = execute_diagnostic_job({**job, "horizon_samples": 240})
    diagnostics = measured.pop("shadow_diagnostic")
    for r in [baseline, measured]:
        r.pop("seconds")
    for k, extra in diagnostics["extra_model_computation_counts"].items():
        measured["model_computation_counts"][k] -= extra
    assert measured == baseline
    assert len(diagnostics["episodes"]) > 0
    assert diagnostics["trace_row_count"] > 0
    assert (
        diagnostics["extra_model_computation_counts"][
            "concept_specific_parameter_optimizer_step_count"
        ]
        == diagnostics["trace_row_count"]
    )
    assert all(e["end_reason"] is not None for e in diagnostics["episodes"])
