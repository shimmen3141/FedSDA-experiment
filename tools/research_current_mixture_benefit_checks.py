"""混合利得の計測を明示実行で検査する。通常のCIへ研究分岐を足さない。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.research_current_mixture_benefit import (
    MixtureBenefitObserver,
    compare_probabilities,
    probability_statistics,
    verify_reference,
)


@pytest.mark.parametrize(
    "current,candidate,mixture,expected",
    [(0.4, 0.6, 0.9, (True, False)), (0.9, 0.6, 0.3, (False, True))],
)
def test_single_model_advantage_and_mixture_benefit_can_disagree(
    current, candidate, mixture, expected
):
    data = compare_probabilities(
        *[torch.tensor([[v]]) for v in (current, candidate, mixture)],
        label=1,
        classes=2,
    )
    assert (data["gains"]["current"] > 0, data["gains"]["add_0.1"] > 0) == expected


def test_window_log_loss_complementarity_is_not_bounded_loss_identity():
    data = [
        compare_probabilities(
            torch.tensor([[0.5]]),
            torch.tensor([[p]]),
            torch.tensor([[0.5]]),
            label=1,
            classes=2,
        )
        for p in (0.99, 0.1)
    ]
    assert sum(d["gains"]["mixture_alone"] for d in data) < 0
    assert sum(d["gains"]["add_0.1"] for d in data) > 0
    for d in data:
        s = d["statistics"]
        assert s["mixture"]["bounded_loss"] - s["add_0.1"][
            "bounded_loss"
        ] == pytest.approx(
            0.1 * (s["mixture"]["bounded_loss"] - s["candidate"]["bounded_loss"]),
            abs=1e-7,
        )


@pytest.mark.parametrize("classes,label", [(2, 0), (2, 1), (3, 2)])
def test_probability_log_loss_exactly_matches_classifier_helper(classes, label):
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )
    from federated_learning_experiments.learning.models.model_architecture_settings import (
        ModelArchitectureSettings,
    )
    from federated_learning_experiments.learning.prediction.class_probability_calculations import (
        convert_model_outputs_to_prediction_probabilities,
    )
    from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
        evaluate_classifier_per_sample_log_losses,
    )

    torch.manual_seed(4)
    model = ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=classes,
    )
    x = torch.tensor([[0.2, 0.8]])
    with torch.no_grad():
        probabilities = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id={0: model(x)}, class_count=classes
        )[0]
        expected = float(
            evaluate_classifier_per_sample_log_losses(
                classifier=model,
                input_features=x,
                observed_class_labels=torch.tensor([[float(label)]]),
            ).item()
        )
    assert (
        probability_statistics(probabilities, label=label, classes=classes)["log_loss"]
        == expected
    )
    assert torch.equal(1.0 * probabilities + 0.0 * probabilities, probabilities)


def test_capture_is_before_training_and_uses_pre_label_mixture_weights():
    from tools.research_reconnected_candidate_holdout_checks import (
        setup_control,
        advance,
    )
    import federated_learning_experiments.runtime.fedsda_run_client as client
    import federated_learning_experiments.runtime.observed_sample_processing as processing

    _, a, state, current, _ = setup_control(gate=False, window=1)

    def adopt(**kw):
        kw["state"]["episode"]["adopted_model_id"] = -1

    observer = MixtureBenefitObserver(window=1, gate=False, adopter=adopt)
    observer.start(state=state, arguments=a)
    advance(a, 11)
    branches = [0.7, 0.3]
    mixture = SimpleNamespace(snapshot_weights=lambda: tuple(branches))
    calls = []

    def prediction(**kwargs):
        branches[:] = [0.2, 0.8]
        return SimpleNamespace(
            combined_prediction_probabilities=torch.tensor([[0.8]]),
            prediction_weights_by_model_id={0: 1.0},
        )

    def process(**kwargs):
        processing.predict_observed_sample_and_update_prediction_weights(
            synchronized_and_local_prediction_mixture=mixture
        )
        calls.append(len(observer.records[0]["mixture_samples"]))
        with torch.no_grad():
            for p in current.parameters():
                p.add_(0.1)
        return SimpleNamespace(alarm_occurrence_handling=None)

    candidate = observer.active[0]["state"]["learner"]
    before_hooks = (len(current._forward_hooks), len(candidate._forward_hooks))
    rng = torch.get_rng_state().clone()
    with (
        patch.object(
            processing,
            "predict_observed_sample_and_update_prediction_weights",
            prediction,
        ),
        patch.object(client, "process_observed_sample", process),
        observer.activated(),
    ):
        client.process_observed_sample(**a)
    record = observer.records[0]
    assert record["end_reason"] == "adopted" and calls == [1]
    assert record["mixture_samples"][0]["branch_weights"] == [0.7, 0.3]
    assert record["mixture_samples"][0]["probabilities"]["mixture"] == pytest.approx(
        [0.8]
    )
    assert (len(current._forward_hooks), len(candidate._forward_hooks)) == before_hooks
    assert torch.equal(rng, torch.get_rng_state())
    assert observer.counts["shared_part_inference_example_count"] == 2
    assert not any(
        v for k, v in observer.counts.items() if "training" in k or "optimizer" in k
    )


def test_reference_detects_prediction_or_adoption_change():
    ref = dict(
        measured=dict(accuracy=0.5, seconds=1, model_computation_counts={"mac": 10}),
        holdouts=[dict(score=2)],
        s1_diagnostic={"events": 1},
        validation_counts={"forward": 2},
    )
    from copy import deepcopy

    arm = deepcopy(ref)
    arm["measured"]["seconds"] = 2
    arm["holdouts"][0]["mixture_samples"] = []
    verify_reference(arm, ref)
    arm["measured"]["accuracy"] = 0.6
    with pytest.raises(ValueError, match="measured"):
        verify_reference(arm, ref)
    arm = deepcopy(ref)
    arm["holdouts"][0]["score"] = 3
    with pytest.raises(ValueError, match="holdouts"):
        verify_reference(arm, ref)


def test_seed_balancing_zero_scores_and_no_candidate_are_distinct():
    from tools.research_current_mixture_benefit_summary import summarize_groups

    def record(score):
        data = compare_probabilities(
            torch.tensor([[0.5]]),
            torch.tensor([[0.5]]),
            torch.tensor([[0.5]]),
            label=1,
            classes=2,
        )
        data["gains"].update(current=score, add_0_ignored=0)
        data["gains"]["add_0.1"] = score
        return dict(end_reason="adopted", mixture_samples=[data])

    rows = [
        dict(
            job=dict(dataset="sea4", stream="drift", interval=100, seed=seed),
            wait=dict(holdouts=records),
        )
        for seed, records in (
            (0, [record(1)] * 3),
            (1, [record(-1)]),
            (2, []),
            (3, [record(0)]),
        )
    ]
    g = summarize_groups(rows)[0]
    assert g["eligible_seed_count"] == 3 and g["all_seed_count"] == 4
    assert g["seed_balanced"]["quadrant_fractions"][
        "current_positive:add_positive"
    ] == pytest.approx(1 / 3)
    assert g["seed_balanced"]["candidate_mean_cumulative_gains"]["current"] == 0
    assert g["strict_zero_window_scores"]["current"] == 1
    assert g["strict_zero_window_scores"]["add_0.1"] == 1
    assert g["seeds"][2]["sample_mean_gains"] is None
