"""SoftRoutingのモデル別寄与を測る診断。"""

from collections import defaultdict
from dataclasses import asdict, dataclass

import torch


@dataclass
class RoutingContributionAggregate:
    """クライアント・モデル・保持集合epoch・通信区間ごとの十分統計。"""

    sample_count: int = 0
    probability_sum: float = 0.0
    bounded_delta_sum: float = 0.0
    bounded_delta_squared_sum: float = 0.0
    zero_one_delta_sum: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    hard_assignment_count: int = 0
    fallback_count: int = 0

    def as_dict(self):
        return asdict(self)


class RoutingLeaveOneOutDiagnostics:
    """実予測からモデルを一つ除いた反実仮想損失を集計する。

    既に計算済みのモデル別予測を再利用するため、追加forwardは発生しない。
    モデル集合が変わった場合はepochを分け、同じIDの異なるモデル実体を混同しない。
    """

    _EPSILON = 1e-12

    def __init__(self):
        self.pool_epoch = -1
        self.pool_signature = None
        self.records = defaultdict(RoutingContributionAggregate)

    @staticmethod
    def _score_loss(scores, target, num_classes):
        if num_classes == 2:
            return float(
                torch.abs(
                    scores.view(-1) - target.view(-1).float()
                ).mean().item()
            )
        labels = target.view(-1).long()
        return float(
            (1.0 - scores.gather(1, labels.unsqueeze(1)).squeeze(1))
            .mean().item()
        )

    @staticmethod
    def _correct(scores, target, num_classes):
        if num_classes == 2:
            prediction = (scores > 0.5).float()
        else:
            prediction = torch.argmax(scores, dim=1, keepdim=True).float()
        return bool(
            prediction.view(-1)[0].item()
            == target.view(-1)[0].item()
        )

    @staticmethod
    def _weighted_scores(prediction_scores, probabilities):
        return sum(
            prediction_scores[model_id] * probability
            for model_id, probability in probabilities.items()
        )

    def _update_pool_epoch(self, model_ids):
        signature = tuple(sorted(model_ids))
        if signature != self.pool_signature:
            self.pool_epoch += 1
            self.pool_signature = signature

    def _scores_without(
        self, removed_model_id, prediction_scores, actual_scores,
        effective_probabilities, fallback_probabilities,
    ):
        removed_probability = effective_probabilities[removed_model_id]
        remaining_mass = (
            sum(effective_probabilities.values()) - removed_probability
        )
        if remaining_mass > self._EPSILON:
            # 全モデルを再加算せず、既存混合から除外分を引いてO(M)で全寄与を得る。
            return (
                (
                    actual_scores
                    - prediction_scores[removed_model_id]
                    * removed_probability
                ) / remaining_mass,
                False,
            )

        remaining = {
            model_id: probability
            for model_id, probability in fallback_probabilities.items()
            if model_id != removed_model_id
        }
        total = sum(remaining.values())
        if total <= self._EPSILON:
            uniform = 1.0 / len(remaining)
            remaining = {
                model_id: uniform for model_id in remaining
            }
        else:
            remaining = {
                model_id: probability / total
                for model_id, probability in remaining.items()
            }
        return self._weighted_scores(prediction_scores, remaining), True

    def observe(
        self, *, prediction_scores, effective_probabilities,
        fallback_probabilities, target, num_classes, current_model_id,
        sample_index, aggregation_interval,
    ):
        """一標本について各モデルのleave-one-out寄与を記録する。"""
        model_ids = tuple(sorted(prediction_scores))
        self._update_pool_epoch(model_ids)
        if len(model_ids) < 2:
            return

        actual_scores = self._weighted_scores(
            prediction_scores, effective_probabilities
        )
        actual_bounded_loss = self._score_loss(
            actual_scores, target, num_classes
        )
        actual_zero_one_loss = float(
            not self._correct(actual_scores, target, num_classes)
        )
        block_index = sample_index // max(1, aggregation_interval)

        for model_id in model_ids:
            scores_without, fallback_used = self._scores_without(
                model_id,
                prediction_scores,
                actual_scores,
                effective_probabilities,
                fallback_probabilities,
            )
            bounded_delta = (
                self._score_loss(scores_without, target, num_classes)
                - actual_bounded_loss
            )
            zero_one_delta = (
                float(not self._correct(scores_without, target, num_classes))
                - actual_zero_one_loss
            )
            aggregate = self.records[
                (self.pool_epoch, block_index, model_id)
            ]
            aggregate.sample_count += 1
            aggregate.probability_sum += float(
                effective_probabilities[model_id]
            )
            aggregate.bounded_delta_sum += bounded_delta
            aggregate.bounded_delta_squared_sum += bounded_delta ** 2
            aggregate.zero_one_delta_sum += zero_one_delta
            aggregate.positive_count += int(bounded_delta > self._EPSILON)
            aggregate.negative_count += int(bounded_delta < -self._EPSILON)
            aggregate.hard_assignment_count += int(
                model_id == current_model_id
            )
            aggregate.fallback_count += int(fallback_used)

    def iter_records(self):
        """保存順が実行環境に依存しない形で十分統計を返す。"""
        for key in sorted(self.records):
            yield (*key, self.records[key])
