"""候補検証を外した後も残すgolden caseと、run終端の乱数状態を守る。"""

import json
from dataclasses import fields

import behavior_golden_cases as golden
import pytest
from behavior_random_state_cases import (
    RANDOM_STATE_CASE_NAMES,
    RANDOM_STATE_FIXTURE_PATH,
    capture_run_end_random_states,
)

from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientScalarSettings,
    FedsdaRunClientSettings,
)


def test_candidate_training_keeps_early_stopping_without_acceptance_settings():
    assert "candidate_model_training_and_acceptance_settings" not in {
        field.name for field in fields(FedsdaRunClientSettings)
    }
    assert "minimum_candidate_mean_loss_improvement" not in {
        field.name for field in fields(FedsdaRunClientScalarSettings)
    }
    settings = golden.build_case_settings("fedsda", 14).run_participant_settings.run_client_settings
    assert settings.candidate_epoch_training_settings.minimum_validation_loss_decrease == 0.0001
    assert settings.scalar_settings.maximum_tolerated_mean_loss_increase == 0.1


@pytest.mark.parametrize("baseline_version", range(1, 15))
def test_excluded_cases_match_saved_policy_and_scoping_report(baseline_version):
    excluded = set(
        golden.EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION[baseline_version]
    )
    common = {
        "fedsda",
        "fedsda-without-prediction-weight-recalibration",
        "fedsda-loss-distance-clustering",
        "fedsda-overall-loss-esr-detection",
        "fedsda-adahedge-prediction-weights",
        "fedsda-single-model-prediction",
        "fedsda-classwise-adwin-detection",
        "fedsda-overall-loss-adwin-detection",
        "fedsda-without-concept-specific-adapter",
        "fedsda-independent-models",
    }
    expected = {"fedsda-sine2-events", "fedsda-mnist2-decisions"}
    if baseline_version >= 2:
        expected.add("fedsda-sine2-reference-reuse")
    if baseline_version >= 4:
        expected.add("fedsda-sine2-alternative-reference-reuse")
    if baseline_version <= 6:
        expected |= common
    assert excluded == expected
    paths = list(golden.golden_directory(baseline_version).glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        from_saved_settings = set()
        for name, case in document["cases"].items():
            settings = case["run_settings"]["run_participant_settings"].get(
                "run_client_settings", {}
            )
            policy = settings.get("candidate_model_training_and_acceptance_settings", {}).get(
                "candidate_model_acceptance_policy"
            )
            if policy == "current_model_first_reuse_then_two_segment_candidate_validation":
                from_saved_settings.add(name)
        assert excluded == from_saved_settings, path


def test_comparison_excludes_only_named_paths_without_changing_inputs(monkeypatch):
    original = {"run_metrics": {"removed": 1, "kept": 2}}
    assert golden.without_removed_candidate_validation_items(original) == original
    monkeypatch.setattr(
        golden, "REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS", (("run_metrics", "removed"),)
    )
    assert golden.without_removed_candidate_validation_items(original) == {
        "run_metrics": {"kept": 2}
    }
    assert original == {"run_metrics": {"removed": 1, "kept": 2}}
    with pytest.raises(AssertionError, match="kept"):
        golden.assert_case_matches(
            "case",
            golden.without_removed_candidate_validation_items(original),
            {"run_metrics": {"kept": 3}},
        )


def test_random_state_fixture_covers_exactly_the_remaining_fedsda_cases():
    document = json.loads(RANDOM_STATE_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert document["source_commit"] == "6adb3ee77ae57af4bcef612916fc6389d075ab80"
    assert document["baseline_version"] == 14
    assert len(RANDOM_STATE_CASE_NAMES) == 14
    assert set(document["cases"]) == set(RANDOM_STATE_CASE_NAMES)
    assert document["mnist_sha256"] == golden.mnist_sha256()
    for state in document["cases"].values():
        assert set(state) == {"python_random", "numpy_random_state", "torch_cpu"}


@pytest.mark.slow
@pytest.mark.parametrize("case_name", RANDOM_STATE_CASE_NAMES)
def test_run_end_random_states_match_before_candidate_training_move(case_name):
    document = json.loads(RANDOM_STATE_FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = capture_run_end_random_states(case_name)
    assert actual == document["cases"][case_name], case_name
