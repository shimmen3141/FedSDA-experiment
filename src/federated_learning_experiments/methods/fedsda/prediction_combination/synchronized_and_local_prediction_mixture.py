"""同期の時点のモデルの結合予測と、局所更新したモデルの結合予測を混ぜる、clientごとの2つの候補の重み。

予測に使うパラメータを`mixture_of_synchronized_and_local_parameters`にした設定で使う。重みは(同期, 局所)の順で、
clientを組み立てるときに半々で作り、clientの寿命の間、作り直さない（警報・配布・集約・再較正でも変えない）。
予測の後、それぞれの結合予測の損失で、Fixed-Shareと同じ形に更新する（指数の重みを掛けて正規化し、分け合う）。
"""

from math import exp, isfinite

from torch import Tensor


def true_class_loss_of_prediction_probabilities(
    *, prediction_probabilities: Tensor, observed_class_id: int, class_count: int
) -> float:
    """結合予測の確率（1標本）から、真のクラスの損失（1−真のクラスの確率）を、倍精度のfloatで求める。

    確率をいったんfloatへ直してから1から引く（2値で真のクラスが0なら`1.0 - (1.0 - v)`）。式を簡略化すると、
    最後の桁が変わる。
    """
    values = prediction_probabilities.reshape(-1)
    if class_count == 2:
        true_class_probability = (
            float(values[0]) if observed_class_id == 1 else 1.0 - float(values[0])
        )
    else:
        true_class_probability = float(values[observed_class_id])
    return 1.0 - true_class_probability


class SynchronizedAndLocalPredictionMixture:
    """(同期, 局所)の2つの候補の重みを持ち、結合予測を混ぜ、損失で更新する。"""

    # 重みの更新の学習率（手元の診断で、0.5〜4の間で精度がほぼ同じだった）。
    LEARNING_RATE = 2.0

    def __init__(self, *, share_rate: float) -> None:
        if type(share_rate) is not float:
            raise TypeError("share_rate must be builtin float")
        if not isfinite(share_rate) or not 0.0 < share_rate < 1.0:
            raise ValueError("share_rate must be finite and in (0, 1)")
        self._share_rate = share_rate
        self._synchronized_weight = 0.5
        self._local_weight = 0.5

    def snapshot_weights(self) -> tuple[float, float]:
        """(同期, 局所)の重み。"""
        return self._synchronized_weight, self._local_weight

    def mix_prediction_probabilities(
        self, *, synchronized_probabilities: Tensor, local_probabilities: Tensor
    ) -> Tensor:
        """2つの結合予測の確率を、今の重みで混ぜる（同期×重み＋局所×重みの順で計算する）。"""
        return (
            synchronized_probabilities * self._synchronized_weight
            + local_probabilities * self._local_weight
        )

    def update_weights_after_loss_observation(
        self, *, synchronized_loss: float, local_loss: float
    ) -> None:
        """損失の指数の重みを掛けて正規化し、分け合う。"""
        updated_synchronized = self._synchronized_weight * exp(
            -self.LEARNING_RATE * synchronized_loss
        )
        updated_local = self._local_weight * exp(-self.LEARNING_RATE * local_loss)
        total = updated_synchronized + updated_local
        updated_synchronized, updated_local = updated_synchronized / total, updated_local / total
        share_rate = self._share_rate
        self._synchronized_weight = (
            1 - share_rate
        ) * updated_synchronized + share_rate * updated_local
        self._local_weight = (1 - share_rate) * updated_local + share_rate * updated_synchronized
