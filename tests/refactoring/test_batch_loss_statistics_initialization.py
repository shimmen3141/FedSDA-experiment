"""外部batch損失の初期集計を旧登録時の数値処理へ直接照合する。"""

from types import SimpleNamespace
import warnings

import pytest
import torch

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)


@pytest.mark.parametrize("loss_values, label_values, class_count", [
    ([0.25], [0], 2),
    ([0.75], [1], 2),
    ([0.2, 0.8], [1, 0], 2),
    ([0.2, 0.3, 0.8], [1, 1, 0], 2),
    ([0.0, 0.0, 0.0], [1, 0, 1], 2),
    ([1.0, 1.0, 1.0], [0, 1, 0], 2),
    ([0.5, 0.2, 0.8, 0.1], [3, 1, 3, 0], 5),
    ([0.2, 0.4, 0.6], [3, 3, 3], 5),
    ([0.10000001, 0.20000003, 0.29999998, 0.40000004, 0.50000006],
     [2, 0, 2, 1, 0], 4),
    ([1e-15, 2e-15, 3e-15, 4e-15], [2, 1, 2, 1], 4),
    ([0.4, 0.1, 0.8, 0.2, 0.6, 0.3], [1, 0, 1, 0, 1, 0], 2),
    ([0.3, 0.6, 0.2, 0.8, 0.1, 0.4], [0, 1, 0, 1, 0, 1], 2),
])
@pytest.mark.parametrize("operation", ["contiguous", "noncontiguous"])
def test_batch_initial_statistics_matches_legacy_registration(
    loss_values, label_values, class_count, operation,
):
    if operation == "contiguous":
        per_sample_bounded_losses = torch.tensor(loss_values, dtype=torch.float32, device="cpu")
        observed_class_labels = torch.tensor(label_values, dtype=torch.float32, device="cpu").reshape(-1, 1)
    else:
        per_sample_bounded_losses = torch.tensor(
            loss_values, dtype=torch.float32, device="cpu").repeat_interleave(2)[::2]
        observed_class_labels = torch.tensor(
            label_values, dtype=torch.float32, device="cpu").repeat_interleave(2).reshape(-1, 2)[:, :1]
        if len(loss_values) > 1:
            assert not per_sample_bounded_losses.is_contiguous()
            assert not observed_class_labels.is_contiguous()
    state_before_call = (per_sample_bounded_losses.clone(), observed_class_labels.clone())
    legacy_model = SimpleNamespace(
        num_classes=class_count,
        per_sample_error=lambda bx, by: per_sample_bounded_losses,
        get_params=lambda: {},
    )
    legacy_client = SimpleNamespace(
        models={}, model_stats={},
        _prepare_model_for_registration=lambda model: model,
        _record_model_compute=lambda *args: None,
    )
    # 旧singletonだけが発生させる未定義不偏分散warningをoracle内で抑制する。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        BaseClient._register_trained_new_model(
            legacy_client, temp_id=-100, new_model=legacy_model,
            bx=torch.zeros((len(loss_values), 1), dtype=torch.float32, device="cpu"),
            by=observed_class_labels, pending_ready=False,
        )
    result = initialize_model_and_class_loss_statistics_from_batch(
        per_sample_bounded_losses=per_sample_bounded_losses,
        observed_class_labels=observed_class_labels,
        class_count=class_count,
    )
    legacy_stats = legacy_client.model_stats[-100]
    assert (result.overall_loss_moments.observed_loss_count,
            result.overall_loss_moments.mean_loss,
            result.overall_loss_moments.sum_squared_loss_deviations) == (
                legacy_stats["n"], legacy_stats["mean"], legacy_stats["M2"])
    assert tuple(class_id for class_id, seed in result.class_loss_moments_by_class_id) == tuple(
        legacy_stats["class_stats"])
    for class_id, seed in result.class_loss_moments_by_class_id:
        assert (seed.observed_loss_count, seed.mean_loss, seed.sum_squared_loss_deviations) == (
            legacy_stats["class_stats"][class_id]["n"],
            legacy_stats["class_stats"][class_id]["mean"],
            legacy_stats["class_stats"][class_id]["M2"],
        )
    assert torch.equal(per_sample_bounded_losses, state_before_call[0])
    assert torch.equal(observed_class_labels, state_before_call[1])
    if len(loss_values) == 1:
        assert result.overall_loss_moments.sum_squared_loss_deviations == 0.1
        assert result.class_loss_moments_by_class_id[0][1].sum_squared_loss_deviations == 0.0
