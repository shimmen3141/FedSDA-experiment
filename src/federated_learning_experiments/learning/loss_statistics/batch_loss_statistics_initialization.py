"""外部計算済みbatch損失から旧torch順の初期集計を作る。"""

import torch

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)


def _validate_batch_loss_statistics_inputs(
    *, per_sample_bounded_losses: torch.Tensor,
    observed_class_labels: torch.Tensor, class_count: int,
) -> None:
    if type(class_count) is not int:
        raise TypeError("class_countはbool以外のbuiltin intが必要です。")
    if class_count < 2:
        raise ValueError("class_countは2以上が必要です。")
    if not isinstance(per_sample_bounded_losses, torch.Tensor):
        raise TypeError("per_sample_bounded_lossesはtorch.Tensorが必要です。")
    if not isinstance(observed_class_labels, torch.Tensor):
        raise TypeError("observed_class_labelsはtorch.Tensorが必要です。")
    if per_sample_bounded_losses.device.type != "cpu":
        raise ValueError("per_sample_bounded_lossesはCPU上のTensorが必要です。")
    if observed_class_labels.device.type != "cpu":
        raise ValueError("observed_class_labelsはCPU上のTensorが必要です。")
    if per_sample_bounded_losses.dtype != torch.float32:
        raise ValueError("per_sample_bounded_lossesはfloat32が必要です。")
    if observed_class_labels.dtype != torch.float32:
        raise ValueError("observed_class_labelsはfloat32が必要です。")
    if per_sample_bounded_losses.layout != torch.strided:
        raise ValueError("per_sample_bounded_lossesはdense strided layoutが必要です。")
    if observed_class_labels.layout != torch.strided:
        raise ValueError("observed_class_labelsはdense strided layoutが必要です。")
    if per_sample_bounded_losses.ndim != 1:
        raise ValueError("per_sample_bounded_lossesはshape[N]が必要です。")
    if observed_class_labels.ndim != 2 or observed_class_labels.shape[1] != 1:
        raise ValueError("observed_class_labelsはshape[N,1]が必要です。")
    if per_sample_bounded_losses.shape[0] == 0:
        raise ValueError("per_sample_bounded_lossesは非空batchが必要です。")
    if observed_class_labels.shape[0] != per_sample_bounded_losses.shape[0]:
        raise ValueError("observed_class_labelsはper_sample_bounded_lossesと同じ標本数が必要です。")
    if not torch.isfinite(per_sample_bounded_losses).all().item():
        raise ValueError("per_sample_bounded_lossesは有限値が必要です。")
    if ((per_sample_bounded_losses < 0) | (per_sample_bounded_losses > 1)).any().item():
        raise ValueError("per_sample_bounded_lossesは0以上1以下が必要です。")
    if not torch.isfinite(observed_class_labels).all().item():
        raise ValueError("observed_class_labelsは有限値が必要です。")
    if (observed_class_labels != torch.trunc(observed_class_labels)).any().item():
        raise ValueError("observed_class_labelsは整数値が必要です。")
    if (observed_class_labels < 0).any().item() or observed_class_labels.max().item() >= class_count:
        raise ValueError("observed_class_labelsは0以上class_count未満が必要です。")


@torch.no_grad()
def initialize_model_and_class_loss_statistics_from_batch(
    *, per_sample_bounded_losses: torch.Tensor,
    observed_class_labels: torch.Tensor, class_count: int,
) -> ModelAndClassLossStatistics:
    """全検査後、全体と存在classのbatch統計を不変値として返す。"""
    _validate_batch_loss_statistics_inputs(
        per_sample_bounded_losses=per_sample_bounded_losses,
        observed_class_labels=observed_class_labels, class_count=class_count,
    )
    batch_sample_count = len(per_sample_bounded_losses)
    overall_mean_loss = float(torch.mean(per_sample_bounded_losses).item())
    overall_sample_variance = (
        float(torch.var(per_sample_bounded_losses, correction=1).item())
        if batch_sample_count > 1 else 0.1)
    flat_class_labels = observed_class_labels.reshape(-1)
    class_loss_moments_by_class_id = []
    for class_id in range(class_count):
        class_bounded_losses = per_sample_bounded_losses[flat_class_labels == class_id]
        class_sample_count = len(class_bounded_losses)
        if class_sample_count == 0:
            continue
        class_mean_loss = float(torch.mean(class_bounded_losses).item())
        class_sample_variance = (
            float(torch.var(class_bounded_losses, correction=1).item())
            if class_sample_count > 1 else 0.0)
        class_loss_moments_by_class_id.append((class_id, BoundedLossMoments(
            observed_loss_count=class_sample_count,
            mean_loss=class_mean_loss,
            sum_squared_loss_deviations=class_sample_variance * max(0, class_sample_count - 1),
        )))
    return ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=batch_sample_count,
            mean_loss=overall_mean_loss,
            sum_squared_loss_deviations=overall_sample_variance * max(1, batch_sample_count - 1),
        ),
        class_loss_moments_by_class_id=tuple(class_loss_moments_by_class_id),
    )
