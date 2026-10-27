"""候補検証を外した後も残すgolden caseと、run終端の乱数状態を守る。"""

import json

import behavior_golden_cases as golden
import pytest
from behavior_random_state_cases import (
    RANDOM_STATE_CASE_NAMES,
    RANDOM_STATE_FIXTURE_PATH,
    capture_run_end_random_states,
)


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
    from_builder = set()
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[baseline_version]:
        participant = golden.build_case_settings(name, baseline_version).run_participant_settings
        client = getattr(participant, "run_client_settings", None)
        acceptance = getattr(client, "candidate_model_training_and_acceptance_settings", None)
        if (
            getattr(acceptance, "candidate_model_acceptance_policy", None)
            == "current_model_first_reuse_then_two_segment_candidate_validation"
        ):
            from_builder.add(name)
    assert excluded == from_builder


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
