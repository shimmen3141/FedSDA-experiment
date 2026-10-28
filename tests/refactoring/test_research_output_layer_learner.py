"""S3の更新範囲、符号長、研究用patchの復元を検査する。"""

from copy import deepcopy

import pytest
import torch

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    measure_model_computation,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.prequential_code_length import (
    _accumulate_prequential_code_length,
)
from tools.research_output_layer_learner.policy import (
    learner_policy,
    output_layer_training,
)


def model(classes):
    torch.manual_seed(20)
    return ModelArchitectureClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=3,
        hidden_layer_widths=(6,),
        class_count=classes,
    )


@pytest.mark.parametrize("classes", [2, 4])
def test_only_head_changes_and_normal_training_resumes(classes):
    classifier = model(classes)
    before = deepcopy(classifier.state_dict())
    settings = AdamParameterOptimizerSettings(
        learning_rate=0.05, weight_decay=0.1, adam_variant="standard"
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [0.0]])
    rng = torch.get_rng_state().clone()
    with output_layer_training(), measure_model_computation() as meter:
        length = _accumulate_prequential_code_length(
            learner=classifier,
            input_features=x,
            observed_class_labels=y,
            parameter_optimizer_settings=settings,
        )
    assert length > 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(v, classifier.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("classification_layer.")
    )
    assert any(
        not torch.equal(v, classifier.state_dict()[k])
        for k, v in before.items()
        if k.startswith("classification_layer.")
    )
    assert all(p.requires_grad for p in classifier.parameters())
    counts = meter.get_model_computation_counts()
    assert counts.shared_part_estimated_backward_multiply_accumulate_count == 0
    assert counts.concept_specific_part_estimated_backward_multiply_accumulate_count == 3 * 6 * (
        1 if classes == 2 else classes
    )
    _accumulate_prequential_code_length(
        learner=classifier,
        input_features=x,
        observed_class_labels=y,
        parameter_optimizer_settings=settings,
    )
    assert any(
        not torch.equal(v, classifier.state_dict()[k])
        for k, v in before.items()
        if k.startswith("model_owned_feature_transform.")
    )


def test_exception_restores_patch_and_default_is_noop():
    original = ModelArchitectureClassifier.forward_from_shared_features
    with pytest.raises(RuntimeError), output_layer_training():
        raise RuntimeError("復元確認")
    assert ModelArchitectureClassifier.forward_from_shared_features is original
    events = []
    with learner_policy(variant="product-default", events=events):
        assert ModelArchitectureClassifier.forward_from_shared_features is original
    assert events == []


def test_warm_score_is_predict_before_update_on_disposable_copy():
    import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection

    classifier = model(2)
    before = deepcopy(classifier.state_dict())
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[0.0]])
    with torch.no_grad():
        expected = -torch.log(1 - classifier(x)).item()
    events = []
    with learner_policy(variant="output-layer-matched", events=events):
        value = selection.compute_warm_started_learner_prequential_code_length(
            template_classifier=classifier,
            input_features=x,
            observed_class_labels=y,
            parameter_optimizer_settings=AdamParameterOptimizerSettings(
                learning_rate=0.05, weight_decay=0.1, adam_variant="standard"
            ),
        )
    assert value == pytest.approx(expected, abs=1e-6)
    assert all(torch.equal(v, classifier.state_dict()[k]) for k, v in before.items())
    assert events == [dict(event="warm_score", samples=1)]


@pytest.mark.parametrize("classes", [2, 4])
def test_real_candidate_epochs_keep_adapter_and_shared_features(classes):
    from federated_learning_experiments.learning.training.candidate_epoch_training import (
        train_candidate_classifier_epochs,
    )
    from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
        CandidateEpochTrainingSettings,
    )
    from federated_learning_experiments.runtime.candidate_classifier_construction import (
        create_independent_candidate_training_state,
    )

    reference = model(classes)
    before = deepcopy(reference.state_dict())
    state = create_independent_candidate_training_state(
        architecture_reference_classifier=reference,
        initial_candidate_parameter_snapshot=dict(before),
        parameter_optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=0.05, weight_decay=0.1, adam_variant="standard"
        ),
    )
    with output_layer_training():
        result = train_candidate_classifier_epochs(
            candidate_classifier=state.candidate_classifier,
            candidate_shared_parameter_optimizer_state=state.candidate_shared_parameter_optimizer_state,
            candidate_concept_specific_parameter_optimizer_state=state.candidate_concept_specific_parameter_optimizer_state,
            input_features=torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]),
            observed_class_labels=torch.tensor([[0.0], [1.0], [0.0]]),
            candidate_epoch_training_settings=CandidateEpochTrainingSettings(
                candidate_training_strategy="fixed_epoch_training",
                maximum_epoch_count=2,
                maximum_batch_sample_count=2,
                validation_sample_fraction=0.2,
                consecutive_non_improving_epoch_limit=2,
                minimum_validation_loss_decrease=0.0,
                candidate_shared_feature_update_policy="keep_shared_features_fixed",
            ),
        )
    assert result.candidate_trained_sample_count == 6
    assert all(
        torch.equal(v, state.candidate_classifier.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("classification_layer.")
    )
    assert any(
        not torch.equal(v, state.candidate_classifier.state_dict()[k])
        for k, v in before.items()
        if k.startswith("classification_layer.")
    )
    optimizer = state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer
    assert all(
        p not in optimizer.state
        for p in state.candidate_classifier.model_owned_feature_transform.parameters()
    )
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in before.items())


def test_analysis_accepts_windows_metadata_and_rejects_missing_seed():
    import json
    from pathlib import Path
    from uuid import uuid4

    from tools.research_output_layer_learner.analyze import analyze
    from tools.research_output_layer_learner.run import source_hashes

    directory = Path("results/_test_tmp") / str(uuid4())
    directory.mkdir(parents=True)
    rows = [
        dict(
            dataset="sea4",
            fixed_concept=None,
            interval=100,
            sample_count=2500,
            seed=seed,
            research_variant=variant,
            accuracy=0.8 if variant == "product-default" else 0.81,
            uploaded_bytes=100,
            downloaded_bytes=200,
            seconds=2.0,
            model_computation_counts={
                "shared_part_inference_forward_multiply_accumulate_count": 300
            },
            research_events=[],
            concept_accuracies={"0": 0.8},
        )
        for seed in range(4)
        for variant in ("product-default", "output-layer-matched")
    ]
    hashes = {k.replace("/", "\\"): v for k, v in source_hashes().items()}
    (directory / "metadata.json").write_text(
        json.dumps(dict(jobs=rows, source_hashes=hashes)), encoding="utf-8"
    )
    (directory / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (directory / "complete.json").write_text(
        json.dumps(dict(run_count=len(rows), source_hashes_verified=True)),
        encoding="utf-8",
    )
    summary = analyze(inputs=[directory], baselines=[], output=directory / "analysis")
    assert summary["summaries"][0]["accuracy_delta_pp"] == pytest.approx(1.0)
    assert summary["summaries"][0]["mac"] == 300
    rows = [r for r in rows if r["seed"] != 3]
    (directory / "metadata.json").write_text(
        json.dumps(dict(jobs=rows, source_hashes=hashes)), encoding="utf-8"
    )
    (directory / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (directory / "complete.json").write_text(
        json.dumps(dict(run_count=len(rows), source_hashes_verified=True)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="4 seeds"):
        analyze(inputs=[directory], baselines=[], output=directory / "analysis")


@pytest.mark.parametrize("classes", [2, 4])
def test_forced_warm_selection_trains_and_adopts_real_head_candidate(classes, monkeypatch):
    """自然発生しない枝も、採用までの実owner接続で検査する。性能評価とは別。"""
    from dataclasses import replace

    from test_alarm_change_interval_resolution import (
        build_alarm_change_interval_resolution_oracle,
    )

    import federated_learning_experiments.runtime.alarm_buffer_response as response
    from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
        MinimumDescriptionLengthAssignmentSelection,
    )

    arguments, _, _, _ = build_alarm_change_interval_resolution_oracle(
        class_count=classes,
        alarm_interval_resolution_case="no_model_fits",
        monkeypatch=monkeypatch,
    )
    arguments["candidate_epoch_training_settings"] = replace(
        arguments["candidate_epoch_training_settings"],
        candidate_training_strategy="fixed_epoch_training",
        maximum_epoch_count=2,
        candidate_shared_feature_update_policy="keep_shared_features_fixed",
    )
    arguments["alarm_interval_assignment_selection"] = MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="new_model",
        change_interval_sample_count=len(arguments["change_interval_training_samples"]),
        destination_model_id=None,
        description_length=0.0,
        new_model_learner_name="warm_started_learner",
    )
    registry = arguments["held_model_training_state_registry"]
    current = registry.get_held_model_training_state(
        model_id=arguments["current_training_model_assignment"].current_training_model_id
    ).classifier
    before = deepcopy(current.state_dict())
    original_forward = ModelArchitectureClassifier.forward_from_shared_features
    events = []
    with learner_policy(variant="output-layer-matched", events=events):
        result = response.resolve_alarm_change_interval(**arguments)
    assert result.resolution_outcome == "alarm_interval_candidate_adopted_as_new_model"
    assert ModelArchitectureClassifier.forward_from_shared_features is original_forward
    candidate = registry.get_held_model_training_state(model_id=result.assigned_model_id).classifier
    assert all(
        torch.equal(v, candidate.state_dict()[k])
        for k, v in before.items()
        if k.startswith("model_owned_feature_transform.")
    )
    assert any(
        not torch.equal(v, candidate.state_dict()[k])
        for k, v in before.items()
        if k.startswith("classification_layer.")
    )
    assert all(torch.equal(v, current.state_dict()[k]) for k, v in before.items())
    matched = [e for e in events if e["event"] == "matched_candidate"]
    assert (
        len(matched) == 1 and matched[0]["frozen_unchanged"] and matched[0]["trained_samples"] > 0
    )
