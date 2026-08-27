"""保持モデルを専門家として扱う、予測用のオンライン重み付け。"""

import math
from collections import defaultdict


class PeriodicForwardProbeActiveSet:
    """LOO寄与から予測時のactive expert集合を周期的に選ぶ。

    最初の ``probe_samples`` 件では全expertを評価し、続く同数の標本では
    有界損失と0/1損失のどちらにも正の限界寄与がないexpertを休止する。
    現行モデルと一時モデルは必ず残し、次周期のprobeで全expertを可逆に戻す。
    """

    def __init__(self, probe_samples):
        self.probe_samples = max(1, int(probe_samples))
        self.pool_signature = ()
        self.cycle_origin = 0
        self.cycle_index = None
        self.retained_ids = set()
        self._records = defaultdict(
            lambda: {
                "sample_count": 0,
                "bounded_delta_sum": 0.0,
                "zero_one_delta_sum": 0.0,
            }
        )
        self.sample_count = 0
        self.probe_sample_count = 0
        self.global_model_count_sum = 0
        self.retained_global_model_count_sum = 0
        self.apply_global_model_count_sum = 0
        self.apply_retained_global_model_count_sum = 0
        self.reconfiguration_count = 0

    @property
    def cycle_size(self):
        return 2 * self.probe_samples

    def _set_retained_ids(self, retained_ids):
        retained_ids = set(retained_ids)
        if retained_ids != self.retained_ids:
            self.reconfiguration_count += 1
        self.retained_ids = retained_ids

    def _restart(self, model_ids, sample_index):
        signature = tuple(sorted(model_ids))
        self.pool_signature = signature
        self.cycle_origin = int(sample_index)
        self.cycle_index = None
        self._records.clear()
        self._set_retained_ids(signature)

    def _reset_for_pool(self, model_ids, sample_index):
        signature = tuple(sorted(model_ids))
        if signature == self.pool_signature:
            return False
        self._restart(signature, sample_index)
        return True

    def restart_for_concept(self, model_ids, sample_index):
        """確定した概念切替後に全expertで新しいprobeを開始する。"""
        self._restart(model_ids, sample_index)

    def _choose_retained_ids(self, model_ids, current_model_id):
        retained_ids = set()
        for model_id in model_ids:
            record = self._records.get(model_id)
            should_retain = (
                model_id < 0
                or model_id == current_model_id
                or record is None
                or record["sample_count"] < self.probe_samples
                or record["bounded_delta_sum"] > 0.0
                or record["zero_one_delta_sum"] > 0.0
            )
            if should_retain:
                retained_ids.add(model_id)
        if not retained_ids:
            retained_ids.add(current_model_id)
        return retained_ids

    def select(self, model_ids, current_model_id, sample_index):
        """予測前にactive集合を返し、保持率の十分統計を更新する。"""
        model_ids = tuple(sorted(model_ids))
        pool_changed = self._reset_for_pool(model_ids, sample_index)
        relative_index = max(0, int(sample_index) - self.cycle_origin)
        cycle_index = relative_index // self.cycle_size
        cycle_position = relative_index % self.cycle_size
        if self.cycle_index != cycle_index:
            self.cycle_index = cycle_index
            self._records.clear()
            self._set_retained_ids(model_ids)
        elif not pool_changed and cycle_position == self.probe_samples:
            self._set_retained_ids(
                self._choose_retained_ids(model_ids, current_model_id)
            )

        # 割当先が周期途中で変わっても学習中モデルは必ず予測候補に残す。
        if current_model_id not in self.retained_ids:
            self._set_retained_ids(self.retained_ids | {current_model_id})
        selected = tuple(
            model_id for model_id in model_ids
            if model_id in self.retained_ids
        )
        is_probe = cycle_position < self.probe_samples
        global_count = sum(model_id >= 0 for model_id in model_ids)
        retained_global_count = sum(model_id >= 0 for model_id in selected)
        self.sample_count += 1
        self.probe_sample_count += int(is_probe)
        self.global_model_count_sum += global_count
        self.retained_global_model_count_sum += retained_global_count
        if not is_probe:
            self.apply_global_model_count_sum += global_count
            self.apply_retained_global_model_count_sum += retained_global_count
        return selected, is_probe

    def observe(self, contributions, sample_index):
        """probe区間で得たモデル別LOO寄与だけを次の適用判定へ蓄積する。"""
        relative_index = max(0, int(sample_index) - self.cycle_origin)
        if relative_index % self.cycle_size >= self.probe_samples:
            return
        for model_id, values in contributions.items():
            record = self._records[model_id]
            record["sample_count"] += 1
            record["bounded_delta_sum"] += values["bounded_delta"]
            record["zero_one_delta_sum"] += values["zero_one_delta"]



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


class SwitchingExpertRouter:
    """時間とともに変わる最良expertを追跡するFixed-Shareルータ。

    全expertの有界損失を使う二次損失適応型Hedgeに、各時刻で一様分布への
    fixed-shareを加える。shareの時間尺度には既存のFIFO長を使うため、独立した
    数値ハイパーパラメータは増やさない。現段階ではshadow診断専用である。
    """

    def __init__(self, share_horizon):
        if int(share_horizon) < 2:
            raise ValueError("share_horizon must be at least 2")
        self.share_horizon = int(share_horizon)
        self.weights = {}
        self.cumulative_variance = 0.0
        self.pool_reset_count = 0
        self.leader_switch_count = 0
        self.aggregation_recalibration_count = 0
        self.aggregation_recalibration_sample_count = 0

    def _clear_evidence(self):
        self.weights = {}
        self.cumulative_variance = 0.0

    def _synchronize(self, expert_ids):
        expert_ids = tuple(sorted(expert_ids))
        if not expert_ids:
            raise ValueError("SwitchingExpertRouter requires at least one expert")
        if set(expert_ids) != set(self.weights):
            if self.weights:
                self.pool_reset_count += 1
            probability = 1.0 / len(expert_ids)
            self.weights = {
                expert_id: probability for expert_id in expert_ids
            }
            self.cumulative_variance = 0.0
        return expert_ids

    def probabilities(self, expert_ids):
        """正解判明前のfixed-share分布を返す。"""
        expert_ids = self._synchronize(expert_ids)
        return {expert_id: self.weights[expert_id] for expert_id in expert_ids}

    @staticmethod
    def _leader(probabilities):
        maximum = max(probabilities.values())
        return min(
            expert_id for expert_id, probability in probabilities.items()
            if probability == maximum
        )

    @staticmethod
    def leader(probabilities, preferred_id=None):
        """最大重みexpertを返し、同率なら指定候補を優先する。"""
        maximum = max(probabilities.values())
        leaders = [
            expert_id
            for expert_id, probability in probabilities.items()
            if probability == maximum
        ]
        if preferred_id in leaders:
            return preferred_id
        return min(leaders)

    def update(self, losses, probabilities):
        """全expertの損失で事後重みを更新し、切替確率を共有する。"""
        expert_ids = self._synchronize(losses)
        if set(probabilities) != set(expert_ids):
            raise ValueError("losses and probabilities must use the same experts")
        bounded_losses = {
            expert_id: min(1.0, max(0.0, float(losses[expert_id])))
            for expert_id in expert_ids
        }
        if len(expert_ids) == 1:
            return

        previous_leader = self._leader(probabilities)
        expected_loss = sum(
            probabilities[expert_id] * bounded_losses[expert_id]
            for expert_id in expert_ids
        )
        variance = sum(
            probabilities[expert_id]
            * (bounded_losses[expert_id] - expected_loss) ** 2
            for expert_id in expert_ids
        )
        self.cumulative_variance += variance
        learning_rate = min(
            1.0,
            math.sqrt(
                2.0 * math.log(len(expert_ids))
                / max(self.cumulative_variance, 1e-12)
            ),
        )
        unnormalized = {
            expert_id: probabilities[expert_id]
            * math.exp(-learning_rate * bounded_losses[expert_id])
            for expert_id in expert_ids
        }
        total = sum(unnormalized.values())
        posterior = {
            expert_id: unnormalized[expert_id] / total
            for expert_id in expert_ids
        }
        share = 1.0 / self.share_horizon
        uniform = 1.0 / len(expert_ids)
        self.weights = {
            expert_id: (1.0 - share) * posterior[expert_id] + share * uniform
            for expert_id in expert_ids
        }
        if self._leader(self.weights) != previous_leader:
            self.leader_switch_count += 1

    def restart_after_aggregation(self):
        """共有表現更新で過去のexpert比較が無効になる場合に初期化する。"""
        self._clear_evidence()
        self.aggregation_recalibration_count += 1

    def replay_after_aggregation(self, loss_sequence):
        """集約後モデルのFIFO損失から時系列状態を再構築する。"""
        loss_sequence = tuple(loss_sequence)
        if not loss_sequence:
            return
        self._clear_evidence()
        self.aggregation_recalibration_count += 1
        self.aggregation_recalibration_sample_count += len(loss_sequence)
        for losses in loss_sequence:
            probabilities = self.probabilities(losses)
            self.update(losses, probabilities)
