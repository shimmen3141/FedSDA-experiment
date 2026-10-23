"""E2の因果順序・再生・全体runの対照を明示実行する。"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.research_provisional_candidate_expert.checks_mixing_past_posteriors import (
    controller,
    observe,
)
from tools.research_provisional_candidate_expert.class_conditioned_prediction import (
    choose_class_conditioned_prediction,
    class_conditioned_prediction_policy,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_and_local_prediction_mixture import (
    SynchronizedAndLocalPredictionMixture,
)


@pytest.mark.parametrize("class_count", [2, 3])
def test_label_free_route_and_selected_expert(class_count):
    controllers = [controller() for _ in range(class_count)]
    for _ in range(15):
        observe(controllers[0], {0: 0.0, 1: 1.0})
        observe(controllers[1], {0: 1.0, 1: 0.0})
    if class_count == 2:
        probabilities = {0: torch.tensor([[0.1]]), 1: torch.tensor([[0.9]])}
        baselines = [torch.tensor([[0.4]]), torch.tensor([[0.6]])]
    else:
        probabilities = {
            0: torch.tensor([[0.9, 0.05, 0.05]]),
            1: torch.tensor([[0.05, 0.9, 0.05]]),
        }
        baselines = [torch.tensor([[0.6, 0.3, 0.1]]), torch.tensor([[0.3, 0.6, 0.1]])]
    for route, baseline in enumerate(baselines):
        before = [deepcopy(c.__dict__) for c in controllers]
        result = choose_class_conditioned_prediction(
            baseline_probabilities=baseline,
            probabilities_by_model_id=[probabilities] * 2,
            source_weights={0: 0.5, 1: 0.5},
            class_controllers=controllers,
            source_mixture_weights=(0.25, 0.75),
            expert_mixture=SynchronizedAndLocalPredictionMixture(share_rate=0.1),
            class_count=class_count,
        )
        assert result[0] == route
        assert max(result[1], key=result[1].get) == route
        assert torch.allclose(result[3], 0.5 * baseline + 0.5 * result[2])
        assert sum(result[4].values()) == pytest.approx(1.0)
        assert [c.__dict__ for c in controllers] == before


def test_single_model_effective_weight_is_normalized_before_product_validation():
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )

    mixture = SynchronizedAndLocalPredictionMixture(share_rate=0.1)
    # 二者の演算の丸めで総和が1をわずかに超える境界を再現する。
    mixture._synchronized_weight = 0.5000000000000001
    mixture._local_weight = 0.5000000000000001
    assert sum(mixture.snapshot_weights()) > 1.0
    result = choose_class_conditioned_prediction(
        baseline_probabilities=torch.tensor([[0.7]]),
        probabilities_by_model_id=[{3: torch.tensor([[0.7]])}] * 2,
        source_weights={3: 1.0},
        class_controllers=[controller(), controller()],
        source_mixture_weights=(0.5, 0.5),
        expert_mixture=mixture,
        class_count=2,
    )
    assert result[4] == {3: 1.0}
    assert (
        FixedShareWeightController.select_maximum_weight_model_id(
            prediction_weights_by_model_id=result[4]
        )
        == 3
    )


def test_fifo_replay_uses_prediction_time_routes_and_clears_absent_class():
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    global_controller = controller()
    classes = [controller(), controller(), controller()]
    for c in classes:
        observe(c)
    sequence = ({0: 0.1, 1: 0.9}, {0: 0.8, 1: 0.2}, {0: 0.3, 1: 0.7})
    owners = SimpleNamespace(
        fixed_share_weight_controller=global_controller,
        pending_sample_observation_store=SimpleNamespace(
            snapshot_pending_sample_observations=lambda: tuple(
                SimpleNamespace(sample_index=i) for i in range(3)
            )
        ),
    )
    client = SimpleNamespace(owners=owners)

    def recalibrate(self):
        self.owners.fixed_share_weight_controller.replay_observed_losses_after_aggregation(
            observed_loss_sequence=sequence
        )

    expected = [controller() for _ in classes]
    expected[0].replay_observed_losses_after_aggregation(
        observed_loss_sequence=(sequence[0], sequence[2])
    )
    expected[1].replay_observed_losses_after_aggregation(
        observed_loss_sequence=(sequence[1],)
    )
    original = FixedShareWeightController.replay_observed_losses_after_aggregation
    with patch.object(
        FedsdaRunClient, "recalibrate_prediction_state_after_aggregation", recalibrate
    ):
        diagnostics = {}
        with class_conditioned_prediction_policy(diagnostics=diagnostics) as states:
            states[global_controller] = dict(
                controllers=classes, routes={0: 0, 1: 1, 2: 0}
            )
            FedsdaRunClient.recalibrate_prediction_state_after_aggregation(client)
            for a, b in zip(classes, expected, strict=True):
                assert a.weights_by_model_id == b.weights_by_model_id
                assert (
                    a.cumulative_observed_loss_variance
                    == b.cumulative_observed_loss_variance
                )
            assert diagnostics["replay_updates"] == 3
    assert (
        FixedShareWeightController.replay_observed_losses_after_aggregation is original
    )


@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_active_path_full_run_check_and_training_invariance(dataset):
    from tools.research_provisional_candidate_expert.run import _execute_job

    job = dict(
        dataset=dataset,
        interval=100,
        seed=0,
        sample_count=600,
        check=False,
        record_prediction_period_metrics=True,
    )
    baseline = _execute_job({**job, "variant": "product-default"})
    check = _execute_job(
        {**job, "variant": "product-default-class-conditioned-prediction-check"}
    )
    active = _execute_job(
        {**job, "variant": "product-default-class-conditioned-prediction"}
    )
    check_diagnostics = check.pop("class_conditioned_prediction")
    active_diagnostics = active.pop("class_conditioned_prediction")
    for item in [baseline, check, active]:
        item.pop("seconds")
        item.pop("variant")
    assert check == baseline
    prediction_keys = {
        "accuracy",
        "record_accuracy",
        "concept_accuracies",
        "prediction_period_accuracies",
        "any_correct_rate",
        "prediction_record_digest",
    }
    assert {k: v for k, v in active.items() if k not in prediction_keys} == {
        k: v for k, v in baseline.items() if k not in prediction_keys
    }
    assert check_diagnostics == active_diagnostics
    assert active_diagnostics["predictions"] == 6000
    assert active_diagnostics["conditional_updates"] == 6000
    assert active_diagnostics["changed_predictions"] > 0
    assert active_diagnostics["replay_updates"] > 0
    assert len(active_diagnostics["class_route_counts"]) >= 2
