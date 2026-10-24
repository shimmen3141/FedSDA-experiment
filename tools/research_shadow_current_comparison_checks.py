"""現行コピーの時点・同一標本・非介入・MAC分離を検証する研究専用test。"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from tools.research_shadow_current_comparison import (
    CurrentComparisonProbe,
    comparison_values,
    execute_comparison_job,
)


@pytest.mark.parametrize(
    "future_samples,boundary",
    [(0, "none"), (1, "horizon"), (1, "replacement"), (1, "run_end")],
)
def test_current_snapshot_is_before_adoption_and_frozen(future_samples, boundary):
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )
    from federated_learning_experiments.learning.models.model_architecture_settings import (
        ModelArchitectureSettings,
    )
    from federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation import (
        evaluate_classifier_per_sample_log_losses,
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
    with torch.no_grad():
        for p in current.feature_extractor.parameters():
            p.add_(2.0)
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
            get_held_model_training_state=lambda **kw: SimpleNamespace(
                classifier=current
            )
        ),
    )
    state = dict(
        learner=learner, observations=[observation], episode=dict(episode_id=0)
    )
    expected = evaluate_classifier_per_sample_log_losses(
        classifier=current,
        input_features=sample.input_features,
        observed_class_labels=sample.observed_class_labels,
    ).tolist()
    before = deepcopy(learner.state_dict())
    current_before = deepcopy(current.state_dict())
    rng = torch.get_rng_state().clone()
    probe = CurrentComparisonProbe(future_samples)
    calls = []

    def adopt(**kw):
        calls.append(1)
        kw["state"]["episode"]["adopted_model_id"] = -1

    probe.adopt(adopt, state=state, arguments=args)
    r = probe.records[0]
    assert (
        calls == [1] and r["current_losses"] == expected and r["current_model_id"] == 1
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, learner.state_dict()[k]) for k, v in before.items())
    assert all(
        torch.equal(v, current.state_dict()[k]) for k, v in current_before.items()
    )
    if boundary == "run_end":
        with probe.activated():
            pass
        assert r["end_reason"] == "run_end"
        assert not probe.active and not probe.current_snapshots
        return
    if boundary == "replacement":
        previous_snapshot = probe.current_snapshots[0]
        args["indexed_observation"] = SimpleNamespace(
            sample_index=20, observed_concept_id=1, training_sample=sample
        )
        state["episode"] = dict(episode_id=1)
        probe.adopt(adopt, state=state, arguments=args)
        assert r["end_reason"] == "next_adoption"
        assert probe.current_snapshots[0] is not previous_snapshot
        assert len(probe.current_snapshots) == 1
        r = probe.records[-1]
    if future_samples:
        with torch.no_grad():
            for p in current.parameters():
                p.add_(3.0)
        args["indexed_observation"] = SimpleNamespace(
            sample_index=r["sample_index"] + 1,
            observed_concept_id=1,
            training_sample=sample,
        )
        probe.observe(args)
        assert r["future"][0]["current_loss"] == expected[0]
        assert not probe.active and not probe.current_snapshots
        counts_before = dict(probe.counts)
        args["indexed_observation"] = SimpleNamespace(
            sample_index=r["sample_index"] + 2,
            observed_concept_id=1,
            training_sample=sample,
        )
        probe.observe(args)
        assert dict(probe.counts) == counts_before and len(r["future"]) == 1
    else:
        assert r["end_reason"] == "disabled" and not probe.current_snapshots
    assert not any(
        v for k, v in probe.counts.items() if "training" in k or "optimizer_step" in k
    )
    expected_inferences = 3 * (
        len(probe.records) + sum(len(r["future"]) for r in probe.records)
    )
    assert probe.counts["shared_part_inference_example_count"] == expected_inferences
    assert (
        probe.counts["concept_specific_part_inference_example_count"]
        == expected_inferences
    )


@pytest.mark.parametrize(
    "invalid",
    ["nonfinite", "length", "recent_zero", "missing_reference", "duplicate_reference"],
)
def test_analysis_rejects_invalid_pairs_and_reference(invalid, monkeypatch):
    from tools.research_shadow_current_comparison import analyze_comparison
    from tools.research_shadow_learner_diagnostic import run_followup

    monkeypatch.setattr(run_followup, "analyze", lambda output: None)
    output = Path(__file__).resolve().parents[1] / "results" / "_test_tmp" / uuid4().hex
    output.mkdir(parents=True)
    job = dict(dataset="sea4", stream="drift", interval=100, seed=1)
    record = dict(
        client_id=0,
        sample_index=10,
        episode_id=0,
        frozen_losses=[0.1],
        connected_losses=[0.1],
        current_losses=[0.1],
        recent_sample_count=1,
        future=[],
        end_reason="disabled",
        shared_squared_parameter_difference=0,
    )
    row = dict(
        job=job,
        baseline={},
        measured={},
        shadow_diagnostic=dict(
            episodes=[
                dict(
                    client_id=0,
                    adoption_sample_index=10,
                    episode_id=0,
                    adopted_model_id=-1,
                )
            ]
        ),
        feature_reconnection_probe=dict(records=[record], future_samples=0),
    )
    ref_path = None
    error = ValueError
    if invalid in ("missing_reference", "duplicate_reference"):
        ref_path = output / "reference"
        ref_path.mkdir()
        (ref_path / "results.json").write_text(
            json.dumps([] if invalid == "missing_reference" else [row, row]),
            encoding="utf-8",
        )
        if invalid == "missing_reference":
            error = KeyError
    elif invalid == "nonfinite":
        record["connected_losses"] = [float("nan")]
    elif invalid == "length":
        record["connected_losses"] = [0.1, 0.2]
    else:
        record["recent_sample_count"] = 0
    (output / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    (output / "research-source.json").write_text(
        json.dumps(
            dict(
                jobs=[job],
                sha256={},
                feature_reconnection_probe=dict(
                    reference_result=str(ref_path) if ref_path else None
                ),
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(error):
        analyze_comparison(output)
    assert not (output / "summary_current_comparison.json").exists()


def test_comparison_signs_and_advantage_loss_use_same_recent_samples():
    record = dict(
        recent_sample_count=2,
        current_losses=[100, 2, 2],
        frozen_losses=[0, 1, 1],
        connected_losses=[0, 3, 3],
        future=[dict(current_loss=2, frozen_loss=1, connected_loss=1.5)],
    )
    v = comparison_values(record)
    assert v["recent_frozen_minus_current"] == -1
    assert v["recent_connected_minus_current"] == 1
    assert v["recent_advantage_lost"]
    assert v["future_connected_minus_current"] == -0.5
    record["future"] = []
    assert comparison_values(record)["future_connected_minus_current"] is None


def test_real_comparison_preserves_all_uninstrumented_fields_and_counts():
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
    old = execute_job(job)
    new = execute_comparison_job(job, future_samples=3)
    for part in ("baseline", "measured"):
        assert {k: v for k, v in old[part].items() if k != "seconds"} == {
            k: v for k, v in new[part].items() if k != "seconds"
        }
    assert old["shadow_diagnostic"] == new["shadow_diagnostic"]
    probe = new["feature_reconnection_probe"]
    assert probe["records"] and all(
        r["current_losses"] and r["future"] for r in probe["records"]
    )
    assert all("current_loss" in t for r in probe["records"] for t in r["future"])
    assert all(
        probe["probe_inclusive_model_computation_counts"][k]
        == v + probe["model_computation_counts"].get(k, 0)
        for k, v in new["measured"]["model_computation_counts"].items()
    )
