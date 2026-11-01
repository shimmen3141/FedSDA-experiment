"""公式AdaStateに従う、サーバだけが所有する学習率の推定。"""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class AdaptiveFedavgLearningRateSettings:
    """初期学習率と3つの指数移動平均の係数。"""

    initial_learning_rate: float = field(
        default=0.01, metadata={"omit_default_from_saved_settings": True}
    )
    beta1: float = field(default=0.5, metadata={"omit_default_from_saved_settings": True})
    beta2: float = field(default=0.5, metadata={"omit_default_from_saved_settings": True})
    beta3: float = field(default=0.5, metadata={"omit_default_from_saved_settings": True})

    def __post_init__(self) -> None:
        if (
            type(self.initial_learning_rate) is not float
            or not isfinite(self.initial_learning_rate)
            or self.initial_learning_rate <= 0
        ):
            raise ValueError("初期学習率は正で有限のbuiltin floatが必要です。")
        for value in (self.beta1, self.beta2, self.beta3):
            if type(value) is not float or value != 0.5:
                raise ValueError("移動平均の係数は公式と同じ0.5に固定します。")


@dataclass(frozen=True, kw_only=True)
class AdaptiveFedavgLearningRateRecord:
    """1回の集約に対応する次時刻用学習率と比の記録。"""

    timestep_index: int
    aggregation_count: int
    variance_ratio: float
    smoothed_ratio: float
    learning_rate_for_next_timestep: float

    def __post_init__(self) -> None:
        if (
            type(self.timestep_index) is not int
            or self.timestep_index < 0
            or type(self.aggregation_count) is not int
            or self.aggregation_count != self.timestep_index + 1
        ):
            raise ValueError("時刻と集約回数が対応していません。")
        if any(
            type(v) is not float or not isfinite(v) or v < 0
            for v in (
                self.variance_ratio,
                self.smoothed_ratio,
                self.learning_rate_for_next_timestep,
            )
        ):
            raise ValueError("学習率と比は非負で有限のbuiltin floatが必要です。")


class AdaptiveFedavgLearningRateState:
    """float64の全state_dictベクトルからAdaStateの式を更新する。"""

    def __init__(self, *, settings: AdaptiveFedavgLearningRateSettings) -> None:
        if type(settings) is not AdaptiveFedavgLearningRateSettings:
            raise TypeError("exact AdaptiveFedavgLearningRateSettingsが必要です。")
        settings.__post_init__()
        self._settings = settings
        self._mu: NDArray[np.float64] | None = None
        self._s = 0.0
        self._gamma = 0.0
        self._records: list[AdaptiveFedavgLearningRateRecord] = []

    def update(
        self, *, aggregated_parameters: NDArray[np.float64]
    ) -> AdaptiveFedavgLearningRateRecord:
        """全候補値の有限性を検査した後だけ状態を進める。"""
        theta = np.asarray(aggregated_parameters, dtype=np.float64)
        if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("パラメータは有限の非空ベクトルが必要です。")
        if self._mu is not None and theta.shape != self._mu.shape:
            raise ValueError("パラメータの次元数は変更できません。")
        t = len(self._records) + 1
        b1, b2, b3 = self._settings.beta1, self._settings.beta2, self._settings.beta3
        prev_mu = np.zeros(theta.shape) if self._mu is None else self._mu
        prev_muh = prev_mu / (1 - b1 ** (t - 1)) if t != 1 else 0
        prev_sh = self._s / (1 - b2 ** (t - 1)) if t != 1 else 0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mu = b1 * prev_mu + (1 - b1) * theta
            s = b2 * self._s + (1 - b2) * np.mean((theta - prev_muh) * (theta - prev_muh))
            sh = s / (1 - b2**t)
            ratio = sh / prev_sh if prev_sh != 0 else 1.0
            gamma = b3 * self._gamma + (1 - b3) * ratio
            gamh = gamma / (1 - b3**t)
            eta = min(
                self._settings.initial_learning_rate,
                self._settings.initial_learning_rate * gamh / t,
            )
        if not np.all(np.isfinite(mu)) or not all(
            np.isfinite(v) for v in (s, sh, ratio, gamma, gamh, eta)
        ):
            raise ValueError("学習率の推定が有限値ではありません。")
        record = AdaptiveFedavgLearningRateRecord(
            timestep_index=t - 1,
            aggregation_count=t,
            variance_ratio=float(ratio),
            smoothed_ratio=float(gamh),
            learning_rate_for_next_timestep=float(eta),
        )
        self._mu = mu.copy()
        self._s = float(s)
        self._gamma = float(gamma)
        self._records.append(record)
        return record

    def snapshot_records(self) -> tuple[AdaptiveFedavgLearningRateRecord, ...]:
        """更新順の不変な記録を返す。"""
        for t, record in enumerate(self._records):
            record.__post_init__()
            if record.timestep_index != t:
                raise ValueError("学習率の記録は連続した時刻が必要です。")
        return tuple(self._records)
