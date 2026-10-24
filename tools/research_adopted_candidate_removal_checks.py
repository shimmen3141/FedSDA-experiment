"""除去の枝別再正規化・ID範囲・統合時停止・seed集計を明示実行で検査する。"""

from types import SimpleNamespace
import pytest
import torch

from tools.research_adopted_candidate_removal import (
    RemovalTracker,
    combine_branches,
    without_candidate,
    summarize_rows,
)
from tools.research_shadow_model_lifecycle_checks import add


@pytest.mark.parametrize("label", [0, 1])
def test_removal_keeps_outer_branch_weights_and_changes_correctness(label):
    from tools.research_current_mixture_benefit import probability_statistics

    sync = {0: torch.tensor([[0.2]]), 1: torch.tensor([[0.8]])}
    local = {0: torch.tensor([[0.9]]), 1: torch.tensor([[0.1]])}
    weights = {0: 0.4, 1: 0.6}
    branches = (0.8, 0.2)
    actual = combine_branches(sync, local, weights, branches, 2)
    without = without_candidate(sync, local, weights, branches, 2, 1)
    assert actual.item() == pytest.approx(0.532)
    assert without.item() == pytest.approx(0.34)
    benefit = (
        probability_statistics(without, label=label, classes=2)["log_loss"]
        - probability_statistics(actual, label=label, classes=2)["log_loss"]
    )
    assert (benefit > 0) == bool(label)
    assert probability_statistics(actual, label=label, classes=2)["correct"] == bool(
        label
    )
    assert probability_statistics(without, label=label, classes=2)["correct"] != bool(
        label
    )


def test_multiclass_and_undefined_mass_are_not_zero_benefit():
    p = {0: torch.tensor([[0.2, 0.3, 0.5]]), 1: torch.tensor([[0.5, 0.4, 0.1]])}
    removed = without_candidate(p, p, {0: 0.4, 1: 0.6}, (0.2, 0.8), 3, 1)
    # 外側の0.2*p+0.8*pにはfloat32の丸めが入る。
    assert torch.allclose(removed, p[0], atol=1e-7, rtol=0)
    assert without_candidate({0: p[0]}, {0: p[0]}, {0: 1.0}, (0.5, 0.5), 3, 0) is None
    assert without_candidate(p, p, {0: 1e-13, 1: 1 - 1e-13}, (0.5, 0.5), 3, 1) is None
    with pytest.raises(ValueError):
        without_candidate(p, {0: p[0]}, {0: 1.0}, (0.5, 0.5), 3, 0)


def test_negative_ids_are_scoped_and_both_merge_sides_stop_measurement():
    tracker = RemovalTracker()
    add(tracker, 0, -1)
    add(tracker, 1, -1)
    assert tracker.direct_tokens(0, -1) == [0]
    assert tracker.direct_tokens(1, -1) == [1]
    tracker.registered(0, -1, 0)
    tracker.registered(1, -1, 1)
    assert tracker.direct_tokens(9, 0) == [0]
    tracker.consolidated(((0, 1),), {0: 0, 1: 0})
    assert not tracker.direct_tokens(0, 0)
    assert all(r["first_merge"] is not None for r in tracker.records)


def test_saved_measurement_uses_both_actual_branches():
    tracker = RemovalTracker()
    add(tracker, 0, -1)
    p = {0: torch.tensor([[0.2]]), -1: torch.tensor([[0.8]])}
    local = {0: torch.tensor([[0.9]]), -1: torch.tensor([[0.1]])}
    weights = {0: 0.4, -1: 0.6}
    branches = (0.8, 0.2)
    tracker.captured = (local, weights, branches)
    tracker.frames.append(
        dict(
            held_model_training_state_registry=SimpleNamespace(
                get_held_model_training_state=lambda **kw: SimpleNamespace(
                    classifier=SimpleNamespace(class_count=2)
                )
            ),
            indexed_observation=SimpleNamespace(
                training_sample=SimpleNamespace(
                    observed_class_labels=torch.tensor([[1.0]])
                ),
                observed_concept_id=2,
            ),
        )
    )
    result = SimpleNamespace(
        prediction_weights_by_model_id=weights,
        prediction_probabilities_by_model_id=p,
        combined_prediction_probabilities=combine_branches(
            p, local, weights, branches, 2
        ),
        sample_prediction_record=SimpleNamespace(maximum_weight_model_id=-1),
    )
    tracker.predicted(0, 11, result, branches)
    sample = tracker.records[0]["removal_samples"][0]
    assert sample["correctness_benefit"] == 1
    assert sample["without_probabilities"] == pytest.approx([0.34])
    assert sample["adoption_age"] == 1 and sample["scope"] == "temporary"
    with pytest.raises(ValueError, match="再結合"):
        result.combined_prediction_probabilities = torch.tensor([[0.8]])
        tracker.predicted(0, 12, result, branches)


def test_seed_mean_and_missing_observations_do_not_pool_candidates():
    def sample(benefit):
        return dict(
            status="defined",
            correctness_benefit=benefit,
            log_loss_benefit=float(benefit),
            scope="temporary",
            is_origin_client=True,
            adoption_age=1,
        )

    rows = []
    for seed, values in ((0, [1, 1, 1]), (1, [-1]), (2, [])):
        rows.append(
            dict(
                job=dict(dataset="sea4", stream="drift", interval=100, seed=seed),
                model_lifecycle=[dict(removal_samples=[sample(v)]) for v in values],
            )
        )
    g = summarize_rows(rows)[0]
    assert g["eligible_seed_count"] == 2 and g["all_seed_count"] == 3
    assert g["seed_balanced_sample_accuracy_delta_pp"] == 0
    assert g["seed_balanced_candidate_mean_log_loss_benefit"] == 0
    assert g["seeds"][2]["delta_accuracy_pp"] is None
