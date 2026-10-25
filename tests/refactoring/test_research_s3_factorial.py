"""符号長と候補学習の独立な切替えを、実モデル・旧対照・RNGに照合する。"""

import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from test_alarm_change_interval_resolution import build_alarm_change_interval_resolution_oracle
from test_joint_model_parameter_update import assert_nested_state_equal
from test_research_output_layer_learner import model

import federated_learning_experiments.runtime.alarm_buffer_response as response
import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
)
from tools.research_output_layer_learner.policy import learner_policy, output_layer_training
from tools.research_s3_factorial.analyze import contrasts
from tools.research_s3_factorial.policy import FACTORS, factorial_policy, input_signature


@pytest.mark.parametrize("arm", list(FACTORS))
@pytest.mark.parametrize("classes", [2, 4])
def test_score_depends_only_on_score_factor_and_does_not_change_template(arm, classes):
    classifier = model(classes)
    before = {k: v.clone() for k, v in classifier.state_dict().items()}
    kw = dict(
        template_classifier=classifier,
        input_features=torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]),
        observed_class_labels=torch.tensor([[0.0], [1.0], [0.0]]),
        parameter_optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=0.05, weight_decay=0.1, adam_variant="standard"
        ),
    )
    native = selection.compute_warm_started_learner_prequential_code_length
    score_head, _ = FACTORS[arm]
    with output_layer_training() if score_head else nullcontext():
        expected = native(**kw)
    rng = torch.get_rng_state().clone()
    events = []
    with factorial_policy(arm=arm, events=events):
        actual = selection.compute_warm_started_learner_prequential_code_length(**kw)
    assert actual == expected
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, classifier.state_dict()[k]) for k, v in before.items())
    assert selection.compute_warm_started_learner_prequential_code_length is native
    assert events == [dict(event="warm_score", samples=3)]


def resolve_candidate(classes, learner, policy, monkeypatch):
    torch.manual_seed(827)
    kw, _, _, _ = build_alarm_change_interval_resolution_oracle(
        class_count=classes, alarm_interval_resolution_case="no_model_fits", monkeypatch=monkeypatch
    )
    kw["candidate_model_training_and_acceptance_settings"] = replace(
        kw["candidate_model_training_and_acceptance_settings"],
        candidate_model_acceptance_policy="current_model_first_reuse_then_candidate_adoption_without_validation",
    )
    kw["candidate_epoch_training_settings"] = replace(
        kw["candidate_epoch_training_settings"],
        candidate_training_strategy="fixed_epoch_training",
        maximum_epoch_count=2,
        candidate_shared_feature_update_policy="keep_shared_features_fixed",
    )
    kw["alarm_interval_assignment_selection"] = MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="new_model",
        change_interval_sample_count=len(kw["change_interval_training_samples"]),
        destination_model_id=None,
        description_length=0.0,
        new_model_learner_name=learner,
    )
    events = []
    with policy(events):
        result = response.resolve_alarm_change_interval(**kw)
    held = kw["held_model_training_state_registry"].get_held_model_training_state(
        model_id=result.assigned_model_id
    )
    return (
        held.classifier.state_dict(),
        held.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
        torch.get_rng_state().clone(),
        events,
    )


@pytest.mark.parametrize("arm", list(FACTORS))
@pytest.mark.parametrize("classes", [2, 4])
def test_actual_warm_candidate_and_optimizer_match_existing_training_scope(
    arm, classes, monkeypatch
):
    _, train_head = FACTORS[arm]
    existing = "output-layer-matched" if train_head else "warm-matched"
    expected = resolve_candidate(
        classes,
        "warm_started_learner",
        lambda events: learner_policy(variant=existing, events=events),
        monkeypatch,
    )
    actual = resolve_candidate(
        classes,
        "warm_started_learner",
        lambda events: factorial_policy(arm=arm, events=events),
        monkeypatch,
    )
    assert_nested_state_equal(actual[0], expected[0])
    assert_nested_state_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert actual[3] == expected[3]
    matched = [e for e in actual[3] if e["event"] == "matched_candidate"]
    assert len(matched) == 1 and matched[0]["head_only"] == train_head


@pytest.mark.parametrize("arm", ["head-score-full-train", "full-score-head-train"])
def test_reinitialized_candidate_is_not_changed(arm, monkeypatch):
    expected = resolve_candidate(
        4,
        "reinitialized_learner",
        lambda events: learner_policy(variant="warm-matched", events=events),
        monkeypatch,
    )
    actual = resolve_candidate(
        4,
        "reinitialized_learner",
        lambda events: factorial_policy(arm=arm, events=events),
        monkeypatch,
    )
    assert_nested_state_equal(actual[0], expected[0])
    assert_nested_state_equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


def test_exception_restores_both_scoring_and_training_contexts():
    score = selection.compute_warm_started_learner_prequential_code_length
    forward = ModelArchitectureClassifier.forward_from_shared_features
    resolve = response.resolve_alarm_change_interval
    with pytest.raises(RuntimeError), factorial_policy(arm="full-score-head-train", events=[]):
        raise RuntimeError("復元確認")
    assert selection.compute_warm_started_learner_prequential_code_length is score
    assert ModelArchitectureClassifier.forward_from_shared_features is forward
    assert response.resolve_alarm_change_interval is resolve


def test_input_signature_matches_previous_audit_format_and_rejects_wrong_coverage():
    rows = {(c, s): (f"hash-{c}-{s}", s, c % 4) for c in range(10) for s in range(2)}
    previous_payload = [(key, v[0], v[1], v[2]) for key, v in sorted(rows.items())]
    expected = hashlib.sha256(json.dumps(previous_payload).encode()).hexdigest()
    assert input_signature(rows, 2) == expected
    changed = dict(rows)
    del changed[9, 1]
    changed[10, 1] = rows[9, 1]
    with pytest.raises(ValueError):
        input_signature(changed, 2)


def test_factorial_effects_separate_default_initializer_and_interaction():
    x = contrasts(
        {
            "product-default": 0.80,
            "warm-matched": 0.79,
            "head-score-full-train": 0.81,
            "full-score-head-train": 0.82,
            "output-layer-matched": 0.825,
        }
    )
    assert x["score_effect_full_training_pp"] == pytest.approx(2)
    assert x["training_effect_full_score_pp"] == pytest.approx(3)
    assert x["interaction_pp"] == pytest.approx(-1.5)
    assert x["initialization_effect_pp"] == pytest.approx(-1)
    assert x["s3_minus_default_pp"] == pytest.approx(2.5)


def synthetic_artifacts(monkeypatch):
    from pathlib import Path
    from uuid import uuid4

    import tools.research_s3_factorial.analyze as analysis
    import tools.research_s3_factorial.run as runner

    directory = Path("results/_test_tmp") / str(uuid4())
    directory.mkdir(parents=True)
    reference, audit, output = directory / "reference", directory / "audit", directory / "output"
    for p in (reference, audit, output):
        p.mkdir()
    refs = {}
    for seed in range(4):
        for arm, value in (
            ("product-default", 0.80),
            ("warm-matched", 0.79),
            ("output-layer-matched", 0.825),
        ):
            refs["mnist4", None, 500, 2500, seed, arm] = dict(
                dataset="mnist4",
                interval=500,
                sample_count=2500,
                seed=seed,
                accuracy=value,
                variant="product-default",
                check=False,
                fixed_concept=None,
                research_variant=arm,
                research_events=[],
                uploaded_bytes=100,
                downloaded_bytes=200,
                seconds=1.0,
                model_computation_counts={
                    "shared_part_inference_forward_multiply_accumulate_count": 300
                },
                concept_accuracies={"0": value},
                prediction_period_accuracies={
                    "stable_period_accuracy": value,
                    "recovery_period_accuracy": value,
                },
            )
    (reference / "results.json").write_text(json.dumps(list(refs.values())), encoding="utf-8")
    proof = audit / "audit_proof.json"
    signatures = {str(seed): f"signature-{seed}" for seed in range(4)}
    parent_rows = [
        dict(
            job=dict(
                dataset="mnist4",
                fixed_concept=None,
                interval=500,
                sample_count=2500,
                seed=m["seed"],
                research_variant=m["research_variant"],
            ),
            metrics=m,
            diagnostic_file=f"s{m['seed']}_{m['research_variant']}.json",
            diagnostic_sha256=f"digest-{i}",
        )
        for i, m in enumerate(refs.values())
    ]
    (audit / "metadata.json").write_text(
        json.dumps(dict(reference_results_sha256=runner.sha(reference / "results.json"))),
        encoding="utf-8",
    )
    (audit / "results.json").write_text(json.dumps(parent_rows), encoding="utf-8")
    proof.write_text(
        json.dumps(
            dict(
                input_signatures=signatures,
                predictions_verified=300000,
                raw_file_sha256={r["diagnostic_file"]: r["diagnostic_sha256"] for r in parent_rows},
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "reference_rows", lambda _: refs)
    monkeypatch.setattr(analysis, "validate_references", lambda *args: (refs, signatures))
    jobs, rows = [], []
    for seed in range(4):
        for arm, value in (("head-score-full-train", 0.81), ("full-score-head-train", 0.82)):
            job = dict(
                dataset="mnist4",
                fixed_concept=None,
                interval=500,
                sample_count=2500,
                seed=seed,
                arm=arm,
            )
            metrics = dict(
                refs["mnist4", None, 500, 2500, seed, "warm-matched"],
                research_variant=arm,
                accuracy=value,
            )
            jobs.append(job)
            rows.append(
                dict(
                    job=job,
                    metrics=metrics,
                    input_signature=signatures[str(seed)],
                    factors=list(FACTORS[arm]),
                )
            )
    meta = dict(
        jobs=jobs,
        seeds=[0, 1, 2, 3],
        sample_count=2500,
        source_hashes=runner.source_hashes(),
        reference=str(reference),
        reference_sha256=runner.sha(reference / "results.json"),
        input_proof=str(proof),
        input_proof_sha256=runner.sha(proof),
    )
    (output / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    (output / "complete.json").write_text(
        json.dumps(dict(run_count=8, source_hashes_verified=True)), encoding="utf-8"
    )
    return output, reference, proof, meta, rows, analysis, runner


def test_analysis_assembles_all_cells_and_separates_baselines(monkeypatch):
    output, _, _, _, _, analysis, _ = synthetic_artifacts(monkeypatch)
    result = analysis.analyze(output)
    assert result["new_runs"] == 8 and result["reused_runs"] == 12
    assert len(result["cells"]) == 20 and len(result["contrasts"]) == 4
    assert result["effect_summary"]["interaction_pp"]["mean"] == pytest.approx(-1.5)


@pytest.mark.parametrize("corruption", ["missing_cell", "wrong_input", "missing_hash"])
def test_analysis_rejects_incomplete_grid_input_or_source_hash(corruption, monkeypatch):
    output, _, _, meta, rows, analysis, _ = synthetic_artifacts(monkeypatch)
    if corruption == "missing_cell":
        rows.pop()
        meta["jobs"].pop()
        (output / "complete.json").write_text(
            json.dumps(dict(run_count=7, source_hashes_verified=True)), encoding="utf-8"
        )
    elif corruption == "wrong_input":
        rows[0]["input_signature"] = "wrong"
    else:
        meta["source_hashes"].pop(next(iter(meta["source_hashes"])))
    (output / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.load_cells(output)


@pytest.mark.parametrize("corruption", ["reference_link", "raw_hash"])
def test_input_proof_is_bound_to_the_same_reference(corruption, monkeypatch):
    _, reference, proof, _, _, _, runner = synthetic_artifacts(monkeypatch)
    runner.validate_references(reference, proof, [0, 1, 2, 3], 2500)
    if corruption == "reference_link":
        path = proof.parent / "metadata.json"
        path.write_text(json.dumps(dict(reference_results_sha256="wrong")), encoding="utf-8")
    else:
        value = json.loads(proof.read_text(encoding="utf-8"))
        value["raw_file_sha256"][next(iter(value["raw_file_sha256"]))] = "wrong"
        proof.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.validate_references(reference, proof, [0, 1, 2, 3], 2500)
