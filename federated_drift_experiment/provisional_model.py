"""新規モデル候補の時系列holdout分割と受入判定。"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalHoldout:
    training_x: torch.Tensor
    training_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor


def temporal_holdout(bx, by, validation_fraction):
    """最新側を検証用に残し、順序を崩さず学習・検証へ分ける。"""
    sample_count = len(bx)
    validation_count = max(2, int(round(sample_count * validation_fraction)))
    validation_count = min(validation_count, sample_count - 1)
    if validation_count < 2 or sample_count - validation_count < 1:
        return None
    split = sample_count - validation_count
    return TemporalHoldout(
        training_x=bx[:split],
        training_y=by[:split],
        validation_x=bx[split:],
        validation_y=by[split:],
    )


def has_consistent_validation_advantage(
    candidate_losses,
    reference_losses,
    min_delta,
):
    """検証区間全体と最新半分の双方で候補が優れるかを判定する。"""
    if len(candidate_losses) != len(reference_losses) or len(candidate_losses) < 2:
        return False
    recent_start = len(candidate_losses) // 2
    partitions = (
        (candidate_losses, reference_losses),
        (candidate_losses[recent_start:], reference_losses[recent_start:]),
    )
    return all(
        float(candidate.mean().item())
        < float(reference.mean().item()) - min_delta
        for candidate, reference in partitions
    )
