"""保持モデルを専門家として扱う、予測用のオンライン重み付け。"""

import math

import torch


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
        self.aggregation_restart_count = 0
        self.aggregation_recalibration_count = 0
        self.aggregation_recalibration_sample_count = 0
        self.aggregation_recalibration_check_count = 0
        self.aggregation_recalibration_skip_count = 0

    def _clear_evidence(self):
        self.cumulative_losses = {}
        self.mixability_gap = 0.0

    def restart_for_concept(self):
        """確定したモデル操作後に、過去概念の累積損失を破棄する。"""
        self._clear_evidence()
        self.concept_restart_count += 1

    def restart_after_aggregation(self):
        """共有表現の集約更新後に、更新前のモデル比較証拠を破棄する。"""
        self._clear_evidence()
        self.aggregation_restart_count += 1
        self.aggregation_recalibration_count += 1

    def replay_after_aggregation(self, loss_sequence):
        """集約後モデルの損失系列で、ルーティング証拠を再構築する。"""
        loss_sequence = tuple(loss_sequence)
        if not loss_sequence:
            return
        self._clear_evidence()
        self.aggregation_recalibration_count += 1
        self.aggregation_recalibration_sample_count += len(loss_sequence)
        for losses in loss_sequence:
            probabilities = self.probabilities(losses)
            self.update(losses, probabilities)

    @staticmethod
    def _leader(losses, preferred_id=None):
        """累積損失が最小の専門家を、同率時は現行モデル優先で返す。"""
        minimum = min(losses.values())
        leaders = [
            expert_id for expert_id, loss in losses.items()
            if loss == minimum
        ]
        if preferred_id in leaders:
            return preferred_id
        return min(leaders)

    @staticmethod
    def _validated_loss_sequence(loss_sequence):
        """再較正損失を固定化し、全時点の専門家集合が同じことを確認する。"""
        loss_sequence = tuple(loss_sequence)
        if not loss_sequence:
            return loss_sequence, set()
        expert_ids = set(loss_sequence[0])
        if any(set(losses) != expert_ids for losses in loss_sequence):
            raise ValueError("all replay losses must use the same experts")
        return loss_sequence, expert_ids

    @staticmethod
    def _summed_losses(loss_sequence, expert_ids):
        return {
            expert_id: sum(losses[expert_id] for losses in loss_sequence)
            for expert_id in expert_ids
        }

    def replay_after_aggregation_if_leader_changed(
        self, loss_sequence, preferred_id=None,
    ):
        """FIFO最良モデルが旧leaderと異なる場合だけ証拠を再構築する。"""
        loss_sequence, expert_ids = self._validated_loss_sequence(loss_sequence)
        if not loss_sequence:
            return False
        self.aggregation_recalibration_check_count += 1

        # モデル集合が変わった場合、旧証拠を新しい集合へ安全に対応付けられない。
        if set(self.cumulative_losses) != expert_ids:
            self.replay_after_aggregation(loss_sequence)
            return True

        historical_leader = self._leader(
            self.cumulative_losses, preferred_id=preferred_id
        )
        recent_losses = self._summed_losses(loss_sequence, expert_ids)
        recent_leader = self._leader(recent_losses, preferred_id=preferred_id)
        if recent_leader == historical_leader:
            self.aggregation_recalibration_skip_count += 1
            return False

        self.replay_after_aggregation(loss_sequence)
        return True

    def replay_after_aggregation_if_leader_persists(
        self, loss_sequence, preferred_id=None,
    ):
        """同じchallengerがFIFO前半・後半の両方で勝つ場合だけ再構築する。"""
        loss_sequence, expert_ids = self._validated_loss_sequence(loss_sequence)
        if not loss_sequence:
            return False
        self.aggregation_recalibration_check_count += 1

        # モデル集合が変わった場合は、旧証拠を維持できないため通常のreplayを行う。
        if set(self.cumulative_losses) != expert_ids:
            self.replay_after_aggregation(loss_sequence)
            return True

        historical_leader = self._leader(
            self.cumulative_losses, preferred_id=preferred_id
        )
        recent_losses = self._summed_losses(loss_sequence, expert_ids)
        challenger = self._leader(recent_losses, preferred_id=preferred_id)
        midpoint = len(loss_sequence) // 2
        if challenger == historical_leader or midpoint == 0:
            self.aggregation_recalibration_skip_count += 1
            return False

        first_half = loss_sequence[:midpoint]
        second_half = loss_sequence[midpoint:]
        challenger_persists = all(
            sum(
                losses[challenger] - losses[historical_leader]
                for losses in interval
            ) < 0.0
            for interval in (first_half, second_half)
        )
        if not challenger_persists:
            self.aggregation_recalibration_skip_count += 1
            return False

        self.replay_after_aggregation(loss_sequence)
        return True

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


class FeatureConditionedRouter:
    """共有特徴からexpert重みを補正するオンライン線形gate。

    Global AdaHedgeの分布を事前分布とし、単位ノルム化した共有特徴に対する
    softmax線形gateで入力依存の補正を加える。正解判明後は全expertの有界損失を
    用いて期待損失を直接下げる。更新幅は累積勾配ノルムから自動調整するため、
    利用者が学習率や切替閾値を設定する必要はない。
    """

    def __init__(self):
        self.weights = {}
        self.squared_gradient_norm = 0.0
        self.pool_reset_count = 0
        self.concept_restart_count = 0
        self.aggregation_restart_count = 0
        self.aggregation_recalibration_sample_count = 0

    def _clear_evidence(self):
        self.weights = {}
        self.squared_gradient_norm = 0.0

    def restart_for_concept(self):
        self._clear_evidence()
        self.concept_restart_count += 1

    def restart_after_aggregation(self):
        self._clear_evidence()
        self.aggregation_restart_count += 1

    def replay_after_aggregation(self, loss_sequence, feature_sequence):
        """集約後の共有特徴と損失でgateを時系列順に再較正する。"""
        loss_sequence = tuple(loss_sequence)
        feature_sequence = tuple(feature_sequence)
        if len(loss_sequence) != len(feature_sequence):
            raise ValueError("loss and feature replay sequences must align")

        self._clear_evidence()
        self.aggregation_restart_count += 1
        self.aggregation_recalibration_sample_count += len(loss_sequence)
        prior_router = AdaHedgeRouter()
        for losses, features in zip(loss_sequence, feature_sequence):
            prior = prior_router.probabilities(losses)
            probabilities, context = self.probabilities(
                losses, features, prior
            )
            self.update(losses, probabilities, context)
            prior_router.update(losses, prior)

    @staticmethod
    def _context(features):
        """共有特徴を単位ノルム化し、切片成分を加える。"""
        vector = features.detach().reshape(len(features), -1).mean(dim=0)
        vector = vector.to(dtype=torch.float32)
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) > 1.0:
            vector = vector / norm
        return torch.cat((vector, vector.new_ones(1)))

    def _synchronize(self, expert_ids, context):
        expert_ids = tuple(sorted(expert_ids))
        if not expert_ids:
            raise ValueError("FeatureConditionedRouter requires at least one expert")
        if set(expert_ids) != set(self.weights):
            if self.weights:
                self.pool_reset_count += 1
            self.weights = {
                expert_id: context.new_zeros(context.numel())
                for expert_id in expert_ids
            }
            self.squared_gradient_norm = 0.0
        return expert_ids

    def probabilities(self, expert_ids, features, prior_probabilities):
        """Global分布を事前に、共有特徴から入力依存の重みを返す。"""
        context = self._context(features)
        expert_ids = self._synchronize(expert_ids, context)
        if set(prior_probabilities) != set(expert_ids):
            raise ValueError("prior probabilities must use the same experts")
        if len(expert_ids) == 1:
            return {expert_ids[0]: 1.0}, context

        tiny = torch.finfo(context.dtype).tiny
        logits = torch.stack([
            torch.dot(self.weights[expert_id], context)
            + math.log(max(tiny, float(prior_probabilities[expert_id])))
            for expert_id in expert_ids
        ])
        values = torch.softmax(logits, dim=0)
        return {
            expert_id: float(values[index].item())
            for index, expert_id in enumerate(expert_ids)
        }, context

    def update(self, losses, probabilities, context):
        """全expertの有界損失からscale-freeなオンライン勾配更新を行う。"""
        expert_ids = self._synchronize(losses, context)
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
        gradients = {
            expert_id: (
                probabilities[expert_id]
                * (bounded_losses[expert_id] - expected_loss)
                * context
            )
            for expert_id in expert_ids
        }
        current_norm = sum(
            float(torch.dot(gradient, gradient).item())
            for gradient in gradients.values()
        )
        if current_norm == 0.0:
            return
        self.squared_gradient_norm += current_norm
        step_size = 1.0 / math.sqrt(self.squared_gradient_norm)
        for expert_id, gradient in gradients.items():
            self.weights[expert_id] -= step_size * gradient
