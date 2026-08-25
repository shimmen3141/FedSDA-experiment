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


@dataclass
class RoutingArchiveShadowAggregate:
    """前通信区間の寄与でローカル保持集合を絞ったshadow予測の集計。"""

    sample_count: int = 0
    actual_correct_count: int = 0
    shadow_correct_count: int = 0
    bounded_delta_sum: float = 0.0
    global_model_count_sum: int = 0
    retained_global_model_count_sum: int = 0
    reconfiguration_count: int = 0


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
        self.archive_shadow = RoutingArchiveShadowAggregate()
        self._archive_shadow_block_index = None
        self._archive_shadow_retained_ids = set()

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
            self._archive_shadow_block_index = None
            self._archive_shadow_retained_ids = set(signature)
            return True
        return False

    def _update_archive_shadow(self, model_ids, block_index, current_model_id):
        """直前区間で二つの損失を悪化させなかったモデルだけをshadowで外す。"""
        if self._archive_shadow_block_index is None:
            self._archive_shadow_block_index = block_index
            self._archive_shadow_retained_ids = set(model_ids)
            return
        if block_index == self._archive_shadow_block_index:
            return

        previous_block = self._archive_shadow_block_index
        retained_ids = set()
        for model_id in model_ids:
            aggregate = self.records.get(
                (self.pool_epoch, previous_block, model_id)
            )
            should_retain = (
                model_id < 0
                or model_id == current_model_id
                or aggregate is None
                or aggregate.bounded_delta_sum > 0.0
                or aggregate.zero_one_delta_sum > 0.0
            )
            if should_retain:
                retained_ids.add(model_id)
        if not retained_ids:
            retained_ids.add(current_model_id)
        if retained_ids != self._archive_shadow_retained_ids:
            self.archive_shadow.reconfiguration_count += 1
        self._archive_shadow_retained_ids = retained_ids
        self._archive_shadow_block_index = block_index

    def _scores_from_subset(
        self, retained_ids, prediction_scores, effective_probabilities,
        fallback_probabilities,
    ):
        """既存予測を再利用して、保持部分集合だけの正規化混合を作る。"""
        effective = {
            model_id: effective_probabilities[model_id]
            for model_id in retained_ids
        }
        total = sum(effective.values())
        if total > self._EPSILON:
            probabilities = {
                model_id: probability / total
                for model_id, probability in effective.items()
            }
            return self._weighted_scores(prediction_scores, probabilities)

        fallback = {
            model_id: fallback_probabilities[model_id]
            for model_id in retained_ids
        }
        total = sum(fallback.values())
        if total <= self._EPSILON:
            uniform = 1.0 / len(fallback)
            fallback = {model_id: uniform for model_id in fallback}
        else:
            fallback = {
                model_id: probability / total
                for model_id, probability in fallback.items()
            }
        return self._weighted_scores(prediction_scores, fallback)

    def _observe_archive_shadow(
        self, *, model_ids, block_index, prediction_scores,
        effective_probabilities, fallback_probabilities,
        actual_bounded_loss, actual_correct, target, num_classes,
        current_model_id,
    ):
        """前区間のLOO寄与で絞ったローカル保持集合を反実仮想評価する。"""
        self._update_archive_shadow(
            model_ids, block_index, current_model_id
        )
        retained_ids = set(self._archive_shadow_retained_ids)
        retained_ids.add(current_model_id)
        retained_ids.intersection_update(model_ids)
        if len(retained_ids) == len(model_ids):
            # 全保持なら実予測をそのまま使い、診断のテンソル演算を増やさない。
            shadow_correct = actual_correct
            shadow_bounded_loss = actual_bounded_loss
        else:
            shadow_scores = self._scores_from_subset(
                retained_ids,
                prediction_scores,
                effective_probabilities,
                fallback_probabilities,
            )
            shadow_correct = self._correct(
                shadow_scores, target, num_classes
            )
            shadow_bounded_loss = self._score_loss(
                shadow_scores, target, num_classes
            )
        aggregate = self.archive_shadow
        aggregate.sample_count += 1
        aggregate.actual_correct_count += int(actual_correct)
        aggregate.shadow_correct_count += int(shadow_correct)
        aggregate.bounded_delta_sum += (
            shadow_bounded_loss - actual_bounded_loss
        )
        global_ids = {model_id for model_id in model_ids if model_id >= 0}
        retained_global_ids = global_ids & retained_ids
        aggregate.global_model_count_sum += len(global_ids)
        aggregate.retained_global_model_count_sum += len(
            retained_global_ids
        )

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
        sample_index, aggregation_interval, archive_shadow_enabled=False,
    ):
        """一標本について各モデルのleave-one-out寄与を記録する。"""
        model_ids = tuple(sorted(prediction_scores))
        self._update_pool_epoch(model_ids)
        actual_scores = self._weighted_scores(
            prediction_scores, effective_probabilities
        )
        actual_bounded_loss = self._score_loss(
            actual_scores, target, num_classes
        )
        actual_zero_one_loss = float(
            not self._correct(actual_scores, target, num_classes)
        )
        actual_correct = not bool(actual_zero_one_loss)
        block_index = sample_index // max(1, aggregation_interval)
        if archive_shadow_enabled:
            self._observe_archive_shadow(
                model_ids=model_ids,
                block_index=block_index,
                prediction_scores=prediction_scores,
                effective_probabilities=effective_probabilities,
                fallback_probabilities=fallback_probabilities,
                actual_bounded_loss=actual_bounded_loss,
                actual_correct=actual_correct,
                target=target,
                num_classes=num_classes,
                current_model_id=current_model_id,
            )
        if len(model_ids) < 2:
            return

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
