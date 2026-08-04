"""保持モデルを専門家として扱う、予測用のオンライン重み付け。"""

import math


class AdaHedgeRouter:
    """[0, 1]損失向けAdaHedgeルータ。

    学習率は累積mixability gapから自動的に決まり、利用者が学習率を
    設定する必要はない。専門家集合が変わった場合は、途中参加モデルへ
    恣意的な累積損失を与えないため全状態を一度リセットする。
    """

    def __init__(self):
        self.cumulative_losses = {}
        self.mixability_gap = 0.0
        self.pool_reset_count = 0
        self.concept_restart_count = 0

    def restart_for_concept(self):
        """確定したモデル操作後に、過去概念の累積損失を破棄する。"""
        self.cumulative_losses = {}
        self.mixability_gap = 0.0
        self.concept_restart_count += 1

    def _synchronize(self, expert_ids):
        expert_ids = tuple(sorted(expert_ids))
        if not expert_ids:
            raise ValueError("AdaHedge requires at least one expert")
        if set(expert_ids) != set(self.cumulative_losses):
            if self.cumulative_losses:
                self.pool_reset_count += 1
            self.cumulative_losses = {expert_id: 0.0 for expert_id in expert_ids}
            self.mixability_gap = 0.0
        return expert_ids

    def learning_rate(self):
        expert_count = len(self.cumulative_losses)
        if expert_count <= 1 or self.mixability_gap <= 0.0:
            return math.inf
        return math.log(expert_count) / self.mixability_gap

    def probabilities(self, expert_ids):
        """現在までの累積損失から次の予測重みを返す。"""
        expert_ids = self._synchronize(expert_ids)
        if len(expert_ids) == 1:
            return {expert_ids[0]: 1.0}

        eta = self.learning_rate()
        minimum = min(self.cumulative_losses.values())
        if math.isinf(eta):
            leaders = [
                expert_id for expert_id in expert_ids
                if self.cumulative_losses[expert_id] == minimum
            ]
            weight = 1.0 / len(leaders)
            return {
                expert_id: weight if expert_id in leaders else 0.0
                for expert_id in expert_ids
            }

        unnormalized = {
            expert_id: math.exp(
                -eta * (self.cumulative_losses[expert_id] - minimum)
            )
            for expert_id in expert_ids
        }
        total = sum(unnormalized.values())
        return {
            expert_id: value / total
            for expert_id, value in unnormalized.items()
        }

    def update(self, losses, probabilities):
        """正解判明後に各専門家の有界損失で状態を更新する。"""
        expert_ids = self._synchronize(losses)
        if set(probabilities) != set(expert_ids):
            raise ValueError("losses and probabilities must use the same experts")
        bounded_losses = {
            expert_id: min(1.0, max(0.0, float(losses[expert_id])))
            for expert_id in expert_ids
        }
        expected_loss = sum(
            probabilities[expert_id] * bounded_losses[expert_id]
            for expert_id in expert_ids
        )

        eta = self.learning_rate()
        if math.isinf(eta):
            active_losses = [
                bounded_losses[expert_id] for expert_id in expert_ids
                if probabilities[expert_id] > 0.0
            ]
            mix_loss = min(active_losses)
        else:
            # etaが大きい場合もexpのアンダーフローを避けるためlog-sum-expで求める。
            log_terms = [
                math.log(probabilities[expert_id])
                - eta * bounded_losses[expert_id]
                for expert_id in expert_ids
                if probabilities[expert_id] > 0.0
            ]
            maximum = max(log_terms)
            log_mixture = maximum + math.log(
                sum(math.exp(value - maximum) for value in log_terms)
            )
            mix_loss = -log_mixture / eta
        self.mixability_gap += max(0.0, expected_loss - mix_loss)

        for expert_id in expert_ids:
            self.cumulative_losses[expert_id] += bounded_losses[expert_id]

    @staticmethod
    def effective_expert_count(probabilities):
        """重みの集中度を逆Simpson指数で返す（1以上、専門家数以下）。"""
        squared_sum = sum(weight * weight for weight in probabilities.values())
        return 1.0 / squared_sum
