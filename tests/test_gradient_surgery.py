import torch

from federated_drift_experiment.gradient_surgery import (
    compare_gradient_updates,
    project_conflicting_gradients,
    summarize_gradient_conflicts,
)


def test_conflicting_gradient_pair_is_measured_and_projected():
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([-1.0, 1.0])

    summary = summarize_gradient_conflicts((first, second))
    projected = project_conflicting_gradients((first, second))

    assert summary.pair_count == 1
    assert summary.conflict_count == 1
    assert summary.cosine_sum < 0.0
    assert torch.dot(projected[0], second) >= -1e-7
    assert torch.dot(projected[1], first) >= -1e-7


def test_nonconflicting_gradients_are_unchanged():
    vectors = (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))

    projected = project_conflicting_gradients(vectors)

    assert all(torch.equal(before, after) for before, after in zip(vectors, projected))


def test_projected_gradients_and_applied_update_can_be_compared():
    vectors = (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]))

    projected = project_conflicting_gradients(vectors)
    applied_summary = summarize_gradient_conflicts(projected)
    reference = sum(vectors) / len(vectors)
    applied = sum(projected) / len(projected)
    comparison = compare_gradient_updates(reference, applied)

    assert applied_summary.pair_count == 1
    assert applied_summary.conflict_count == 0
    assert comparison is not None
    assert comparison.cosine <= 1.0
    assert comparison.norm_ratio > 0.0
    assert comparison.delta_ratio > 0.0


def test_zero_reference_update_has_no_comparison():
    assert compare_gradient_updates(torch.zeros(2), torch.ones(2)) is None
