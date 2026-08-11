"""共有表現を学習する複数目的の勾配競合を計測・補正する。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class GradientConflictSummary:
    """一回の共同更新に含まれる概念勾配対の診断値。"""

    pair_count: int
    conflict_count: int
    cosine_sum: float
    negative_cosine_sum: float


@dataclass(frozen=True)
class GradientUpdateComparison:
    """基準更新に対して実際に適用する更新がどれだけ変形したかを表す。"""

    cosine: float
    norm_ratio: float
    delta_ratio: float


def flatten_parameter_gradients(gradients, parameters):
    """parameter順を保ち、未使用勾配をゼロとして一つのベクトルへ連結する。"""
    values = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None else gradient.detach().clone().reshape(-1)
        )
    return torch.cat(values) if values else torch.empty(0)


def summarize_gradient_conflicts(vectors):
    """元の勾配対についてcosine類似度と負の内積の件数を集計する。"""
    pair_count = 0
    conflict_count = 0
    cosine_sum = 0.0
    negative_cosine_sum = 0.0
    for first, second in combinations(vectors, 2):
        first_norm = torch.linalg.vector_norm(first)
        second_norm = torch.linalg.vector_norm(second)
        if first_norm.item() == 0.0 or second_norm.item() == 0.0:
            continue
        cosine = float(torch.dot(first, second).div(first_norm * second_norm).item())
        pair_count += 1
        cosine_sum += cosine
        if cosine < 0.0:
            conflict_count += 1
            negative_cosine_sum += cosine
    return GradientConflictSummary(
        pair_count=pair_count,
        conflict_count=conflict_count,
        cosine_sum=cosine_sum,
        negative_cosine_sum=negative_cosine_sum,
    )


def project_conflicting_gradients(vectors):
    """PCGradと同様に、他目的と負の内積を持つ成分だけを射影除去する。

    再現可能性を保つため、呼び出し側が与えた安定した順に他勾配を処理する。
    """
    projected = []
    for index, vector in enumerate(vectors):
        current = vector.clone()
        for other_index, other in enumerate(vectors):
            if index == other_index:
                continue
            denominator = torch.dot(other, other)
            if denominator.item() == 0.0:
                continue
            inner = torch.dot(current, other)
            if inner.item() < 0.0:
                current = current - inner.div(denominator) * other
        projected.append(current)
    return projected


def compare_gradient_updates(reference, applied):
    """二つの更新方向を方向・大きさ・差分の三つの尺度で比較する。

    基準更新がゼロの場合は比率を定義できないため ``None`` を返す。適用更新が
    ゼロの場合、方向の一致度は0として扱う。
    """
    if reference.shape != applied.shape:
        raise ValueError("比較する勾配更新の形状が一致していません")
    reference_norm = torch.linalg.vector_norm(reference)
    if reference_norm.item() == 0.0:
        return None
    applied_norm = torch.linalg.vector_norm(applied)
    cosine = 0.0
    if applied_norm.item() != 0.0:
        cosine = float(
            torch.dot(reference, applied).div(
                reference_norm * applied_norm
            ).item()
        )
    return GradientUpdateComparison(
        cosine=cosine,
        norm_ratio=float(applied_norm.div(reference_norm).item()),
        delta_ratio=float(
            torch.linalg.vector_norm(applied - reference)
            .div(reference_norm)
            .item()
        ),
    )


def assign_flat_gradient(parameters, gradient):
    """連結済み勾配を元のparameter形状へ戻してoptimizerへ渡す。"""
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        value = gradient[offset:offset + size].view_as(parameter)
        parameter.grad = value.clone()
        offset += size
    if offset != gradient.numel():
        raise ValueError("連結勾配とバックボーンparameterの要素数が一致しません")
