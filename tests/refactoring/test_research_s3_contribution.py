"""S3寄与診断の同時除去・実採用接続・patch復元を検査する。"""

from dataclasses import replace

import pytest
import torch
from test_alarm_change_interval_resolution import build_alarm_change_interval_resolution_oracle

import federated_learning_experiments.runtime.alarm_buffer_response as response
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
)
from tools.research_adopted_candidate_removal import combine_branches, without_candidate
from tools.research_output_layer_learner.policy import learner_policy
from tools.research_s3_contribution.diagnostic import ContributionTracker, remove_models


def test_joint_removal_renormalizes_once_and_is_not_sum_of_single_removals():
    p = {0: torch.tensor([[0.1]]), 1: torch.tensor([[0.7]]), 2: torch.tensor([[0.9]])}
    w = {0: 0.2, 1: 0.3, 2: 0.5}
    joint = remove_models(p, p, w, (0.7, 0.3), 2, [1, 2])
    assert torch.allclose(joint, p[0])
    actual = combine_branches(p, p, w, (0.7, 0.3), 2)
    single_sum = sum((actual - without_candidate(p, p, w, (0.7, 0.3), 2, m)).item() for m in (1, 2))
    assert (actual - joint).item() != pytest.approx(single_sum)
    assert remove_models(p, p, w, (0.7, 0.3), 2, [0, 1, 2]) is None


def test_multiclass_zero_remaining_mass_and_bad_ids():
    p = {0: torch.tensor([[0.2, 0.3, 0.5]]), 1: torch.tensor([[0.5, 0.3, 0.2]])}
    assert remove_models(p, p, {0: 0.0, 1: 1.0}, (0.5, 0.5), 3, [1]) is None
    assert torch.allclose(remove_models(p, p, {0: 0.4, 1: 0.6}, (0.8, 0.2), 3, [0]), p[1])
    with pytest.raises(ValueError):
        remove_models(p, p, {0: 0.4, 1: 0.6}, (0.8, 0.2), 3, [2])


@pytest.mark.parametrize("classes", [2, 4])
def test_real_s3_adoption_is_connected_to_lifecycle_tracker(classes, monkeypatch):
    arguments, _, _, _ = build_alarm_change_interval_resolution_oracle(
        class_count=classes, alarm_interval_resolution_case="no_model_fits", monkeypatch=monkeypatch
    )
    arguments["candidate_model_training_and_acceptance_settings"] = replace(
        arguments["candidate_model_training_and_acceptance_settings"],
        candidate_model_acceptance_policy="current_model_first_reuse_then_candidate_adoption_without_validation",
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
    tracker = ContributionTracker("output-layer-matched")
    tracker.frames.append(dict(client_id=0))
    original = response.resolve_alarm_change_interval
    with tracker.activated(), learner_policy(variant="output-layer-matched", events=[]):
        result = response.resolve_alarm_change_interval(**arguments)
    assert response.resolve_alarm_change_interval is original
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record["temporary_model_id"] == result.assigned_model_id
    assert record["learner"] == "warm_started_learner" and record["is_head_only"]
    assert record["trained_samples"] > 0 and record["update_steps"] > 0
    assert record["sample_index"] == arguments["proposal_sample_index"]
    assert record["initial_classifier_hash"] is not None
    assert tracker.direct_tokens(0, result.assigned_model_id) == [0]
    assert tracker.direct_tokens(1, result.assigned_model_id) == []
    tracker.registered(0, result.assigned_model_id, 100)
    assert tracker.direct_tokens(1, 100) == [0]
    tracker.consolidated(((100, 101),), {100: 100, 101: 100})
    assert tracker.direct_tokens(0, 100) == []
    assert record["first_merge"] is not None


def test_exception_restores_observation_wrappers():
    import federated_learning_experiments.runtime.observed_sample_processing as processing

    before = processing.select_alarm_interval_assignment
    with pytest.raises(RuntimeError), ContributionTracker("product-default").activated():
        raise RuntimeError("復元確認")
    assert processing.select_alarm_interval_assignment is before


def test_analysis_paired_inputs_and_mdl_roundtrip():
    import json
    from dataclasses import asdict

    from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
        select_assignment_by_minimum_description_length,
    )
    from tools.research_s3_contribution.analyze import compare_traces, summarize, validate_decision

    args = dict(
        current_model_log_losses=(1.0, 1.0),
        held_model_log_losses_by_model_id=(),
        new_model_code_lengths_by_change_interval_sample_count={1: 0.1, 2: 0.2},
        new_model_learner_names_by_change_interval_sample_count={
            1: "warm_started_learner",
            2: "warm_started_learner",
        },
        switching_code_length=0.1,
        change_interval_grid_sample_count=1,
    )
    record = json.loads(
        json.dumps(
            dict(
                mdl_inputs=args,
                selection=asdict(select_assignment_by_minimum_description_length(**args)),
            )
        )
    )
    validate_decision(record)
    record["selection"]["description_length"] += 1
    with pytest.raises(ValueError):
        validate_decision(record)
    p = dict(
        client_id=0,
        sample_index=0,
        concept=0,
        label=1,
        input_hash="same",
        probabilities=[0.8],
        correct=True,
    )
    b = dict(p, probabilities=[0.2], correct=False)
    comparison = compare_traces([b], [p], 0)
    assert comparison["strata"]["all"]["net"] == 1
    assert comparison["net_correct_before_first_head_adoption"] == 1
    with pytest.raises(ValueError):
        compare_traces([b], [dict(p, input_hash="different")])
    assert (
        summarize([dict(status="undefined", correctness_benefit=None)])["mean_log_loss_benefit"]
        is None
    )


def test_coverage_rejects_equal_count_substituted_keys_and_pairs_by_key():
    from tools.research_s3_contribution.analyze import compare_traces, validate_prediction_coverage

    ps = [
        dict(
            client_id=c,
            sample_index=s,
            concept=0,
            label=0,
            input_hash=f"{c}-{s}",
            probabilities=[0.8],
            correct=True,
        )
        for c in range(10)
        for s in range(2)
    ]
    validate_prediction_coverage(ps, 2)
    altered = [*ps[:-1], dict(ps[-1], client_id=10)]
    with pytest.raises(ValueError, match="被覆"):
        validate_prediction_coverage(altered, 2)
    with pytest.raises(ValueError):
        validate_prediction_coverage([*ps[:-1], ps[0]], 2)
    result = compare_traces(ps, list(reversed(ps)))
    assert not result["same_processing_order"]
    assert result["strata"]["all"]["net"] == 0
    assert result["first_probability_difference"] is None


def test_analysis_imports_native_mdl_in_fresh_cli_without_pytest_path():
    import os
    import subprocess
    import sys

    from tools.research_s3_contribution.run import ROOT

    environment = dict(os.environ, PYTHONPATH="")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tools.research_s3_contribution.analyze; from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import select_assignment_by_minimum_description_length; print('ok')",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "ok"
