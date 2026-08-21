"""新規モデル候補の時系列holdout・逐次比較と受入判定。"""
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class ForwardCreationPolicy:
    """前向き検証方式を構成する直交した判断規則。"""

    requalify_references: bool
    prefer_current_reference: bool
    require_disjoint_persistence: bool
    train_reference_shadows: bool = False
    update_all_shadows: bool = False
    sequential_tournament: bool = False


FORWARD_CREATION_POLICIES = {
    "forward_validated": ForwardCreationPolicy(False, False, False),
    "forward_requalified": ForwardCreationPolicy(True, False, False),
    "forward_requalified_current_first": ForwardCreationPolicy(
        True, True, False
    ),
    "forward_persistent": ForwardCreationPolicy(True, True, True),
    "shadow_tournament": ForwardCreationPolicy(
        False, False, False, train_reference_shadows=True
    ),
    "sequential_tournament": ForwardCreationPolicy(
        False, False, False,
        update_all_shadows=True,
        sequential_tournament=True,
    ),
}


def forward_creation_policy(name):
    """前向き検証方式でなければNone、該当すれば方式定義を返す。"""
    return FORWARD_CREATION_POLICIES.get(name)


@dataclass(frozen=True)
class TemporalHoldout:
    training_x: torch.Tensor
    training_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor


@dataclass(frozen=True)
class ProvisionalModelDecision:
    """仮モデルの採否と、その判断根拠を再分析可能な形で保持する。"""

    position: int
    detector: str
    accepted: bool
    reason: str
    interval_count: int
    training_count: int
    validation_count: int
    reference_model_id: Optional[int]
    candidate_mean_loss: float
    reference_mean_loss: float
    candidate_recent_loss: float
    reference_recent_loss: float
    resolution_position: Optional[int] = None
    validation_source: str = "historical"
    reference_historical_mean: float = float("nan")

    @property
    def full_margin(self):
        """正なら仮モデルが検証区間全体で優れる。"""
        return self.reference_mean_loss - self.candidate_mean_loss

    @property
    def recent_margin(self):
        """正なら仮モデルが検証区間の直近半分で優れる。"""
        return self.reference_recent_loss - self.candidate_recent_loss

    @property
    def resolution_delay(self):
        """警報から採否確定までに観測したサンプル数を返す。"""
        if self.resolution_position is None:
            return 0
        return self.resolution_position - self.position

    @property
    def reference_excess(self):
        """前向き区間損失が参照モデルの履歴平均をどれだけ上回ったかを返す。"""
        return self.reference_mean_loss - self.reference_historical_mean


@dataclass
class ForwardValidationSession:
    """警報後データでshadow candidateを検証する未確定状態。"""

    proposal_position: int
    estimated_change_point: Optional[int]
    episode_id: Optional[int]
    old_model_id: int
    detector: str
    candidate: Any
    training_x: torch.Tensor
    training_y: torch.Tensor
    held_data: list
    reference_models: dict
    target_count: Optional[int]
    candidate_training_examples: int = 0
    candidate_optimizer_steps: int = 0
    reference_training_examples: dict[int, int] = field(default_factory=dict)
    reference_optimizer_steps: dict[int, int] = field(default_factory=dict)
    reference_historical_means: dict[int, float] = field(default_factory=dict)
    candidate_losses: list[float] = field(default_factory=list)
    reference_losses: dict[int, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.target_count is not None and self.target_count < 2:
            raise ValueError("forward validation requires at least 2 samples")
        if not self.reference_losses:
            self.reference_losses = {
                model_id: [] for model_id in self.reference_models
            }
        if not self.reference_training_examples:
            self.reference_training_examples = {
                model_id: 0 for model_id in self.reference_models
            }
        if not self.reference_optimizer_steps:
            self.reference_optimizer_steps = {
                model_id: 0 for model_id in self.reference_models
            }

    @property
    def ready(self):
        return (
            self.target_count is not None
            and len(self.candidate_losses) >= self.target_count
        )

    @property
    def validation_count(self):
        return len(self.candidate_losses)

    def append_losses(self, candidate_loss, reference_losses):
        self.candidate_losses.append(float(candidate_loss))
        for model_id in self.reference_losses:
            self.reference_losses[model_id].append(
                float(reference_losses[model_id])
            )


_TOURNAMENT_CANDIDATE = "candidate"
_TOURNAMENT_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)


def _log_mixture_e_value(differences):
    """平均差が0以下という帰無仮説に対するbounded betting e値を返す。"""
    log_wealths = []
    for betting_fraction in _TOURNAMENT_BETTING_FRACTIONS:
        log_wealth = 0.0
        for difference in differences:
            bounded = max(-1.0, min(1.0, float(difference)))
            log_wealth += math.log1p(betting_fraction * bounded)
        log_wealths.append(log_wealth)
    maximum = max(log_wealths)
    return maximum + math.log(sum(
        math.exp(value - maximum) for value in log_wealths
    )) - math.log(len(log_wealths))


def sequential_tournament_winner(
    candidate_losses,
    reference_losses,
    alpha,
):
    """全対戦相手への逐次e値が誤選択予算を超えた唯一の勝者を返す。

    各損失は[0, 1]、二候補のpaired loss差は[-1, 1]とする。方向付きの
    全ペアへalphaをBonferroni配分するため、任意時点で誤った優越判定を
    一つ以上行う確率をalpha以下に制御する。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("sequential tournament alpha must be in (0, 1)")
    if not candidate_losses or not reference_losses:
        return None
    sample_count = len(candidate_losses)
    if any(len(losses) != sample_count for losses in reference_losses.values()):
        raise ValueError("all tournament candidates require paired losses")

    losses_by_candidate = {
        _TOURNAMENT_CANDIDATE: candidate_losses,
        **reference_losses,
    }
    candidate_ids = tuple(losses_by_candidate)
    ordered_comparison_count = len(candidate_ids) * (len(candidate_ids) - 1)
    log_threshold = math.log(ordered_comparison_count / alpha)
    certified = []
    for candidate_id in candidate_ids:
        candidate_series = losses_by_candidate[candidate_id]
        if all(
            _log_mixture_e_value([
                opponent_loss - candidate_loss
                for candidate_loss, opponent_loss in zip(
                    candidate_series, losses_by_candidate[opponent_id]
                )
            ]) >= log_threshold
            for opponent_id in candidate_ids
            if opponent_id != candidate_id
        ):
            certified.append(candidate_id)
    if not certified:
        return None
    return min(
        certified,
        key=lambda candidate_id: sum(losses_by_candidate[candidate_id]),
    )


def tournament_candidate_id():
    """逐次比較で新規候補を表す衝突しない識別子を返す。"""
    return _TOURNAMENT_CANDIDATE


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


def validation_rejection_reason(candidate_losses, reference_losses, min_delta):
    """採用ならaccepted、棄却なら満たさなかった時間範囲を返す。"""
    recent_start = len(candidate_losses) // 2
    full_margin = float(reference_losses.mean() - candidate_losses.mean())
    recent_margin = float(
        reference_losses[recent_start:].mean()
        - candidate_losses[recent_start:].mean()
    )
    full_failed = full_margin <= min_delta
    recent_failed = recent_margin <= min_delta
    if full_failed and recent_failed:
        return "full_and_recent"
    if full_failed:
        return "full_interval"
    if recent_failed:
        return "recent_interval"
    return "accepted"


def has_disjoint_validation_advantage(
    candidate_losses,
    reference_losses,
    min_delta,
):
    """重複しない前半・後半の両方で候補が優れるかを判定する。"""
    if len(candidate_losses) != len(reference_losses) or len(candidate_losses) < 2:
        return False
    split = len(candidate_losses) // 2
    partitions = (
        (candidate_losses[:split], reference_losses[:split]),
        (candidate_losses[split:], reference_losses[split:]),
    )
    return all(
        len(candidate) > 0
        and float(candidate.mean().item())
        < float(reference.mean().item()) - min_delta
        for candidate, reference in partitions
    )


def disjoint_validation_rejection_reason(
    candidate_losses,
    reference_losses,
    min_delta,
):
    """持続性判定で不合格になった独立区間を返す。"""
    split = len(candidate_losses) // 2
    first_margin = float(
        reference_losses[:split].mean() - candidate_losses[:split].mean()
    )
    second_margin = float(
        reference_losses[split:].mean() - candidate_losses[split:].mean()
    )
    first_failed = first_margin <= min_delta
    second_failed = second_margin <= min_delta
    if first_failed and second_failed:
        return "first_and_second"
    if first_failed:
        return "first_interval"
    if second_failed:
        return "second_interval"
    return "accepted"


def select_forward_fitting_reference(
    reference_losses,
    reference_historical_means,
    distance_threshold,
    preferred_model_id=None,
):
    """前向きデータでも履歴平均から閾値内に収まる既存モデルを返す。"""
    fitting = []
    for model_id, losses in reference_losses.items():
        if model_id not in reference_historical_means or not losses:
            continue
        mean_loss = float(torch.as_tensor(losses).mean().item())
        historical_mean = float(reference_historical_means[model_id])
        if mean_loss - historical_mean <= distance_threshold:
            fitting.append((mean_loss, model_id))
    if not fitting:
        return None
    fitting_ids = {model_id for _, model_id in fitting}
    if preferred_model_id in fitting_ids:
        return preferred_model_id
    return min(fitting)[1]
