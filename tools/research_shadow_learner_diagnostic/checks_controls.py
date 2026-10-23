"""無変化入力、乱数順、オンライン継続の独立した状態を実物で確認する。"""

from copy import deepcopy
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "dataset,concept", [("sea4", 0), ("sea4", 2), ("sine2", 0), ("sine2", 1)]
)
def test_fixed_concept_changes_pretraining_and_trace_without_changing_schedule_rng(
    dataset, concept
):
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )
    import federated_learning_experiments.runtime.single_run_execution as execution
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
        RandomConceptScheduleSettings,
    )
    from federated_learning_experiments.data.observed_sample_generation import (
        create_observed_sample_generator,
    )

    a, b = Random(3), Random(3)
    args = dict(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=3,
            client_count=2,
            per_client_sample_count=600,
            server_aggregation_interval_per_client_samples=100,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=0.05,
        ),
    )
    original = execution.generate_random_client_concept_traces(
        **args, python_random_generator=a
    )
    expected = create_observed_sample_generator(
        dataset_name=dataset, numpy_random_generator=np.random.RandomState(3)
    )
    expected_samples = [expected.generate_sample(concept_id=concept) for _ in range(3)]
    actual = create_observed_sample_generator(
        dataset_name=dataset, numpy_random_generator=np.random.RandomState(3)
    )
    with stationary_concept_policy(dataset=dataset, concept_id=concept):
        traces = execution.generate_random_client_concept_traces(
            **args, python_random_generator=b
        )
        assert all(set(t.concept_ids_by_sample_index) == {concept} for t in traces)
        # 事前学習側がconcept0を要求しても固定した概念で生成する。
        assert [
            actual.generate_sample(concept_id=0) for _ in range(3)
        ] == expected_samples
    assert a.getstate() == b.getstate()
    assert len(original) == len(traces)
    assert actual.generate_sample.__func__ == expected.generate_sample.__func__


def test_online_control_copies_optimizer_binding_and_continues_without_mutating_product():
    from tools.research_shadow_learner_diagnostic.controls import (
        initialize_online_current_control,
        observe_online_current_control,
    )
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
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        continue_shadow_observation,
    )

    torch.manual_seed(4)
    current = ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
    )
    optimizer = create_parameter_optimizer(
        parameters=tuple(
            p
            for n, p in current.named_parameters()
            if not n.startswith("feature_extractor.")
        ),
        optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
        ),
    )
    x, y = torch.tensor([[0.2, 0.7]]), torch.tensor([[1.0]])
    continue_shadow_observation(
        learner=current,
        optimizer=optimizer,
        current_classifier=current,
        input_features=x,
        observed_class_labels=y,
    )
    before = deepcopy(current.state_dict())
    opt_before = deepcopy(optimizer.state_dict())
    torch_before = torch.get_rng_state().clone()
    held = SimpleNamespace(
        classifier=current,
        concept_specific_parameter_optimizer_state=SimpleNamespace(
            parameter_optimizer=optimizer
        ),
    )
    state = dict(episode=dict(initial_score=-2.0))
    initialize_online_current_control(
        state=state,
        arguments=dict(
            current_training_model_assignment=SimpleNamespace(
                current_training_model_id=0
            ),
            held_model_training_state_registry=SimpleNamespace(
                get_held_model_training_state=lambda **kw: held
            ),
        ),
    )
    control = state["online_control"]
    assert all(
        p is q
        for p, q in zip(
            control["optimizer"].param_groups[0]["params"],
            [
                p
                for n, p in control["classifier"].named_parameters()
                if not n.startswith("feature_extractor.")
            ],
        )
    )
    reference, reference_opt = deepcopy((current, optimizer))
    _, expected_loss = continue_shadow_observation(
        learner=reference,
        optimizer=reference_opt,
        current_classifier=current,
        input_features=x,
        observed_class_labels=y,
    )
    out = observe_online_current_control(
        state=state,
        current_classifier=current,
        input_features=x,
        observed_class_labels=y,
        current_loss=expected_loss + 0.5,
        shadow_loss=expected_loss - 3.0,
        age=1,
        same_concept=True,
    )
    assert out["control_score"] == 1.0
    assert out["current_control_advantage"] == 0.5
    assert state["episode"]["control_first_crossing_age"] == 1
    for k, v in before.items():
        assert torch.equal(current.state_dict()[k], v)
        assert torch.equal(
            control["classifier"].state_dict()[k], reference.state_dict()[k]
        )
    assert optimizer.state_dict()["param_groups"] == opt_before["param_groups"]
    for p, values in optimizer.state_dict()["state"].items():
        for k, v in values.items():
            assert torch.equal(v, opt_before["state"][p][k])
    assert torch.equal(torch.get_rng_state(), torch_before)
