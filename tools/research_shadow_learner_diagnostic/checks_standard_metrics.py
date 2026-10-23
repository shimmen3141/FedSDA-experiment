"""対象無しと0%を区別し、単位換算前の標準値と対照照合を確かめる。"""

from copy import deepcopy
import pytest

from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
    describe,
    extract_pair,
    total_mac,
)


def make_pair(changes=0):
    baseline = dict(
        accuracy=0.8,
        uploaded_bytes=1000000,
        downloaded_bytes=2000000,
        seconds=10.0,
        concept_accuracies={"0": 0.8},
        prediction_period_accuracies=dict(
            stable_period_accuracy=0.8, recovery_period_accuracy=0.0
        ),
        communication_volume=dict(
            uploaded_byte_count=1000000, downloaded_byte_count=2000000
        ),
        model_computation_counts=dict(
            test_forward_multiply_accumulate_count=100,
            test_estimated_backward_multiply_accumulate_count=200,
            example_count=2,
        ),
        preparation_model_computation_counts=dict(
            test_forward_multiply_accumulate_count=30
        ),
        training_model_switch_detection_metrics=dict(concept_change_count=changes),
    )
    measured = deepcopy(baseline)
    measured["seconds"] = 12.0
    measured["model_computation_counts"]["test_forward_multiply_accumulate_count"] += 30
    return dict(
        job=dict(
            dataset="sea4", stream="fixed0", interval=100, seed=1, sample_count=5000
        ),
        baseline=baseline,
        measured=measured,
        baseline_parity=True,
        shadow_diagnostic=dict(
            extra_model_computation_counts=dict(
                test_forward_multiply_accumulate_count=30
            ),
            episodes=[],
        ),
    )


def test_seed_summary_uses_sample_sd_and_does_not_treat_unavailable_as_zero():
    out = describe([None, 1.0, 3.0])
    assert out["n"] == 2 and out["unavailable"] == 1
    assert out["mean"] == 2.0 and out["sd"] == pytest.approx(2**0.5)
    assert describe([None])["mean"] is None
    assert describe([0.0])["mean"] == 0.0


def test_saved_standard_values_extra_mac_and_raw_no_recovery_are_preserved():
    row = make_pair()
    out = extract_pair(row)
    assert out["baseline_recovery_period_accuracy"] is None
    assert (
        '"recovery_period_accuracy": 0.0'
        in out["baseline_prediction_period_accuracies_json"]
    )
    assert out["baseline_uploaded_bytes"] == 1000000
    assert out["baseline_mac"] == 300 and out["measured_mac"] == 330
    assert out["baseline_preparation_mac"] == 30
    assert out["extra_mac_fraction"] == 0.1
    assert out["accuracy_delta_pp"] == 0.0
    assert out["baseline_seconds"] == 10 and out["measured_seconds"] == 12
    assert total_mac(row["baseline"]["model_computation_counts"]) == 300


def test_observed_zero_recovery_accuracy_is_not_marked_unavailable():
    out = extract_pair(make_pair(changes=2))
    assert out["baseline_recovery_period_accuracy"] == 0.0


@pytest.mark.parametrize("bad_field", ["accuracy", "model_computation_counts"])
def test_changed_baseline_or_unaccounted_computation_is_rejected(bad_field):
    row = make_pair()
    if bad_field == "accuracy":
        row["measured"][bad_field] = 0.9
    else:
        row["measured"][bad_field]["example_count"] += 1
    with pytest.raises(ValueError, match="不一致"):
        extract_pair(row)
