"""共有部接続の計測が実験経路を変えないことと、固定コピーの対照を検証する。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.research_shadow_feature_reconnection import (
    ReconnectionProbe,
    execute_probe_job,
)


@pytest.mark.parametrize("change_shared", [False, True])
def test_probe_uses_copies_and_preserves_rng_parameters_and_future_order(change_shared):
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )
    from federated_learning_experiments.learning.models.model_architecture_settings import (
        ModelArchitectureSettings,
    )

    torch.manual_seed(7)
    learner = ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
    )
    current = deepcopy(learner)
    if change_shared:
        with torch.no_grad():
            for p in current.feature_extractor.parameters():
                p.add_(2.0)
    before = deepcopy(learner.state_dict())
    current_before = deepcopy(current.state_dict())
    sample = SimpleNamespace(
        input_features=torch.tensor([[0.3, 0.8]]),
        observed_class_labels=torch.tensor([[1.0]]),
    )
    observation = SimpleNamespace(
        sample_index=10, observed_concept_id=0, training_sample=sample
    )
    args = dict(
        client_id=0,
        indexed_observation=observation,
        current_training_model_assignment=SimpleNamespace(current_training_model_id=1),
        held_model_training_state_registry=SimpleNamespace(
            get_held_model_training_state=lambda **kwargs: SimpleNamespace(
                classifier=current
            )
        ),
    )
    state = dict(
        learner=learner, observations=[observation], episode=dict(episode_id=3)
    )
    probe = ReconnectionProbe(1)
    rng = torch.get_rng_state().clone()

    def adopt(**kwargs):
        kwargs["state"]["episode"]["adopted_model_id"] = -1

    probe.adopt(adopt, state=state, arguments=args)
    record = probe.records[0]
    assert (record["shared_squared_parameter_difference"] > 0) is change_shared
    if change_shared:
        assert record["frozen_losses"] != record["connected_losses"]
    else:
        assert record["frozen_losses"] == record["connected_losses"]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, learner.state_dict()[k]) for k, v in before.items())
    assert all(
        torch.equal(v, current.state_dict()[k]) for k, v in current_before.items()
    )
    args["indexed_observation"] = SimpleNamespace(
        sample_index=11, observed_concept_id=1, training_sample=sample
    )
    probe.observe(args)
    assert len(record["future"]) == 1 and record["end_reason"] == "horizon"
    assert not probe.active
    assert not any(
        v for k, v in probe.counts.items() if "training" in k or "optimizer_step" in k
    )


def test_real_adoption_run_matches_uninstrumented_trajectory_and_counts():
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    job = dict(
        dataset="sea4",
        stream="drift",
        fixed_concept=None,
        interval=100,
        seed=1,
        sample_count=600,
        horizon_samples=240,
        task="adoption",
        disabled=False,
    )
    original = execute_job(job)
    measured = execute_probe_job(job, future_samples=3)
    for field in ("baseline", "measured"):
        assert {k: v for k, v in original[field].items() if k != "seconds"} == {
            k: v for k, v in measured[field].items() if k != "seconds"
        }
    assert original["shadow_diagnostic"] == measured["shadow_diagnostic"]
    probe = measured["feature_reconnection_probe"]
    assert probe["records"] and any(e["future"] for e in probe["records"])
    assert sum(
        e["end_reason"] == "shadow_adopted"
        for e in measured["shadow_diagnostic"]["episodes"]
    ) == len(probe["records"])
    assert all(
        probe["probe_inclusive_model_computation_counts"][k]
        - probe["model_computation_counts"].get(k, 0)
        == v
        for k, v in measured["measured"]["model_computation_counts"].items()
    )
