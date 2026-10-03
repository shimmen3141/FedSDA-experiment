"""明示した固定条件と独立したモデル別予測重みの状態を管理する。"""

import math
from collections.abc import Iterable

from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)


class FixedSharePredictionWeightController:
    """予測用重みと分散・計数を一実体で所有し、診断値をコピーで公開する。"""

    def __init__(self, *, prediction_combination_settings: PredictionCombinationSettings) -> None:
        if type(prediction_combination_settings) is not PredictionCombinationSettings:
            raise TypeError("prediction_combination_settingsにはPredictionCombinationSettingsを指定してください。")
        prediction_combination_settings.__post_init__()
        try:
            if not math.isfinite(float(
                prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
            )):
                raise ValueError("fixed_share_weight_redistribution_time_scale_samplesは有限floatへ変換できる整数にしてください。")
        except OverflowError as caught_exception:
            raise ValueError(
                "fixed_share_weight_redistribution_time_scale_samplesは有限floatへ変換できる整数にしてください。",
            ) from caught_exception
        self._prediction_combination_settings = prediction_combination_settings
        self._weights_by_model_id: dict[int, float] = {}
        self._cumulative_observed_loss_variance = 0.0
        self._model_pool_reset_count = 0
        self._prediction_weight_leader_switch_count = 0
        self._aggregation_recalibration_count = 0
        self._aggregation_recalibration_sample_count = 0

    @property
    def weights_by_model_id(self) -> dict[int, float]:
        """内部辞書と結合しない現在の予測重みを返す。"""
        return dict(self._weights_by_model_id)

    @property
    def cumulative_observed_loss_variance(self) -> float:
        """観測損失から累積した分散を返す。"""
        return self._cumulative_observed_loss_variance

    @property
    def model_pool_reset_count(self) -> int:
        """初回を除くモデルID集合変更の回数を返す。"""
        return self._model_pool_reset_count

    @property
    def prediction_weight_leader_switch_count(self) -> int:
        """重み更新による最大重みモデルの変更回数を返す。"""
        return self._prediction_weight_leader_switch_count

    @property
    def aggregation_recalibration_count(self) -> int:
        """集約後の再較正回数を返す。"""
        return self._aggregation_recalibration_count

    @property
    def aggregation_recalibration_sample_count(self) -> int:
        """集約後に再生した標本数を返す。"""
        return self._aggregation_recalibration_sample_count

    @staticmethod
    def _validated_model_ids(*, model_ids: Iterable[int]) -> tuple[int, ...]:
        """空・重複・boolを含む非整数を拒否して昇順ID列を返す。"""
        validated_model_ids = tuple(model_ids)
        if not validated_model_ids:
            raise ValueError("model_idsには非空のモデルID集合を指定してください。")
        for model_id in validated_model_ids:
            if type(model_id) is not int:
                raise TypeError("model_idsにはbuiltin intのIDを指定してください。boolは受理しません。")
        if len(set(validated_model_ids)) != len(validated_model_ids):
            raise ValueError("model_idsに重複したモデルIDを含めないでください。")
        return tuple(sorted(validated_model_ids))

    def _synchronize_model_pool(self, *, validated_model_ids: tuple[int, ...]) -> None:
        """集合が変わった場合だけ一様重みとゼロ分散へ再構成する。"""
        if set(validated_model_ids) != set(self._weights_by_model_id):
            if self._weights_by_model_id:
                self._model_pool_reset_count += 1
            uniform_model_weight = 1.0 / len(validated_model_ids)
            self._weights_by_model_id = {
                model_id: uniform_model_weight for model_id in validated_model_ids
            }
            self._cumulative_observed_loss_variance = 0.0

    def get_prediction_weights_before_label_observation(
        self, *, model_ids: Iterable[int],
    ) -> dict[int, float]:
        """ラベルを入力にせずモデル集合を同期し、独立した事前重みを返す。"""
        validated_model_ids = self._validated_model_ids(model_ids=model_ids)
        self._synchronize_model_pool(validated_model_ids=validated_model_ids)
        return self.weights_by_model_id
