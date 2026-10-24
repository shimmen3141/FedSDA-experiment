"""新標本の境界、採用条件、中止、共有部追随、既存S1との一致を確認する。"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.research_reconnected_candidate_holdout import (
    ReconnectedHoldout,
    concept_specific_digest,
    run_holdout_arm,
)


def setup_control(*, gate=True, window=3):
    from federated_learning_experiments.learning.models.model_architecture_classifier import (
        ModelArchitectureClassifier,
    )
    from federated_learning_experiments.learning.models.model_architecture_settings import (
        ModelArchitectureSettings,
    )

    torch.manual_seed(7)
    current = ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(4,),
        class_count=2,
    )
    learner = deepcopy(current)
    sample = SimpleNamespace(
        input_features=torch.tensor([[0.3, 0.8]]),
        observed_class_labels=torch.tensor([[1.0]]),
    )
    args = dict(
        client_id=0,
        indexed_observation=SimpleNamespace(
            sample_index=10, observed_concept_id=0, training_sample=sample
        ),
        current_training_model_assignment=SimpleNamespace(current_training_model_id=0),
        held_model_training_state_registry=SimpleNamespace(
            get_held_model_training_state=lambda **kw: SimpleNamespace(
                classifier=current
            )
        ),
    )
    state = dict(
        learner=learner,
        optimizer=None,
        observations=[args["indexed_observation"]],
        episode=dict(episode_id=0, last_score=1.0, end_reason=None),
    )
    adoptions = []

    def adopt(**kw):
        adoptions.append(kw)
        kw["state"]["episode"]["adopted_model_id"] = -1

    control = ReconnectedHoldout(window=window, gate=gate, adopter=adopt)
    return control, args, state, current, adoptions


def advance(args, index):
    old = args["indexed_observation"]
    args["indexed_observation"] = SimpleNamespace(
        sample_index=index,
        observed_concept_id=old.observed_concept_id,
        training_sample=old.training_sample,
    )


@pytest.mark.parametrize(
    "gate,delta,expected",
    [
        (True, 0.1, "adopted"),
        (True, 0, "rejected"),
        (True, -0.1, "rejected"),
        (False, -0.1, "adopted"),
    ],
)
def test_fixed_window_never_uses_start_or_adopts_early(
    gate, delta, expected, monkeypatch
):
    import federated_learning_experiments.learning.prediction.classifier_log_loss_evaluation as evaluation

    c, a, s, current, adoptions = setup_control(gate=gate)
    c.start(state=s, arguments=a)
    assert c.records[0]["score"] == 0 and not adoptions
    with pytest.raises(ValueError, match="新標本"):
        c.observe(a)
    monkeypatch.setattr(
        evaluation,
        "evaluate_classifier_per_sample_log_losses",
        lambda **kw: torch.tensor(
            [0.2 if kw["classifier"] is current else 0.2 - delta]
        ),
    )
    before = concept_specific_digest(c.active[0]["state"]["learner"])
    for index in (11, 12, 13):
        advance(a, index)
        c.observe(a)
        c.after_sample(a, SimpleNamespace(alarm_occurrence_handling=None))
        if index < 13:
            assert not adoptions and c.records[0]["end_reason"] is None
    assert c.records[0]["end_reason"] == expected
    assert bool(adoptions) == (expected == "adopted") and not c.active
    if adoptions:
        assert concept_specific_digest(adoptions[0]["state"]["learner"]) == before
        assert len(adoptions[0]["state"]["observations"]) == 1
    assert [t["sample_index"] for t in c.records[0]["samples"]] == [11, 12, 13]


@pytest.mark.parametrize(
    "reason", ["alarm", "assignment_changed", "replaced", "run_end"]
)
def test_interruption_is_not_rejection_or_adoption(reason):
    c, a, s, current, adoptions = setup_control()
    c.start(state=s, arguments=a)
    advance(a, 11)
    if reason == "alarm":
        c.after_sample(a, SimpleNamespace(alarm_occurrence_handling=object()))
    elif reason == "assignment_changed":
        a["current_training_model_assignment"].current_training_model_id = 1
        c.observe(a)
    elif reason == "replaced":
        c.start(state=s, arguments=a)
    else:
        with c.activated():
            pass
    assert c.records[0]["end_reason"] == reason and not adoptions
    if reason != "replaced":
        assert not c.active


def test_prediction_evaluation_training_adoption_order_and_live_shared_alias():
    import federated_learning_experiments.runtime.fedsda_run_client as client
    import federated_learning_experiments.runtime.observed_sample_processing as processing

    c, a, s, current, adoptions = setup_control(gate=False, window=1)
    c.start(state=s, arguments=a)
    candidate = c.active[0]["state"]["learner"]
    head = concept_specific_digest(candidate)
    events = []
    observe = c.observe

    def tracked_observe(args):
        events.append("evaluate")
        observe(args)

    c.observe = tracked_observe

    def predict(**kw):
        events.append("predict")
        return object()

    def process(**kw):
        processing.predict_observed_sample_and_update_prediction_weights(**kw)
        events.append("train")
        with torch.no_grad():
            for p in current.feature_extractor.parameters():
                p.add_(0.1)
        assert not adoptions
        return SimpleNamespace(alarm_occurrence_handling=None)

    advance(a, 11)
    with (
        patch.object(client, "process_observed_sample", process),
        patch.object(
            processing, "predict_observed_sample_and_update_prediction_weights", predict
        ),
        c.activated(),
    ):
        client.process_observed_sample(**a)
    assert events == ["predict", "evaluate", "train"] and len(adoptions) == 1
    assert candidate.feature_extractor is current.feature_extractor
    assert concept_specific_digest(candidate) == head
    assert not any(
        v for k, v in c.counts.items() if "training" in k or "optimizer_step" in k
    )
    assert c.counts["shared_part_inference_example_count"] == 2


def test_handoff_is_not_falsely_recorded_as_real_adoption():
    import federated_learning_experiments.runtime.fedsda_run_client as client

    c, a, s, current, adoptions = setup_control()

    def process(**kw):
        c.start(state=s, arguments=kw)
        s["episode"]["end_reason"] = "shadow_adopted"
        return SimpleNamespace(alarm_occurrence_handling=None)

    with patch.object(client, "process_observed_sample", process), c.activated():
        client.process_observed_sample(**a)
    assert s["episode"]["end_reason"] == "reconnected_validation_started"
    assert c.records[0]["end_reason"] == "run_end" and not adoptions


def test_2500_sample_zero_wait_is_exactly_original_s1():
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    job = dict(
        dataset="sea4",
        stream="fixed2",
        fixed_concept=2,
        interval=100,
        seed=1,
        sample_count=2500,
        horizon_samples=240,
        task="adoption",
        disabled=False,
    )
    old = execute_job(job)
    new = run_holdout_arm(job, window=0, gate=True)
    assert {k: v for k, v in old["measured"].items() if k != "seconds"} == {
        k: v for k, v in new["measured"].items() if k != "seconds"
    }
    assert old["shadow_diagnostic"] == new["s1_diagnostic"]
    assert not new["holdouts"] and not new["validation_counts"]
