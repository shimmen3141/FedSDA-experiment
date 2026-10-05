"""明示した固定条件と独立したモデル別予測重みの状態を管理する。"""

import math
from collections.abc import Iterable, Mapping

from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)


class FixedSharePredictionWeightController:
    """予測用重みと分散・計数を一実体で所有し、診断値をコピーで公開する。"""

    def __init__(self, *, prediction_combination_settings: PredictionCombinationSettings) -> None:
        if type(prediction_combination_settings) is not PredictionCombinationSettings:
            raise TypeError(
                "prediction_combination_settingsにはPredictionCombinationSettingsを指定してください。"
            )
        prediction_combination_settings.__post_init__()
        try:
            if not math.isfinite(
                float(
                    prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
                )
            ):
                raise ValueError(
                    "fixed_share_weight_redistribution_time_scale_samplesは有限floatへ変換できる整数にしてください。"
                )
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
                raise TypeError(
                    "model_idsにはbuiltin intのIDを指定してください。boolは受理しません。"
                )
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
        self,
        *,
        model_ids: Iterable[int],
    ) -> dict[int, float]:
        """ラベルを入力にせずモデル集合を同期し、独立した事前重みを返す。"""
        validated_model_ids = self._validated_model_ids(model_ids=model_ids)
        self._synchronize_model_pool(validated_model_ids=validated_model_ids)
        return self.weights_by_model_id

    @staticmethod
    def _validated_observed_losses(
        *,
        observed_losses_by_model_id: Mapping[int, float],
    ) -> dict[int, float]:
        """状態変更前に有限損失を検証し、昇順の独立辞書へ変換する。"""
        if not isinstance(observed_losses_by_model_id, Mapping):
            raise TypeError("observed_losses_by_model_idにはMappingを指定してください。")
        validated_model_ids = FixedSharePredictionWeightController._validated_model_ids(
            model_ids=observed_losses_by_model_id,
        )
        observed_losses_by_model_id = dict(observed_losses_by_model_id)
        for model_id in validated_model_ids:
            observed_loss = observed_losses_by_model_id[model_id]
            if type(observed_loss) not in (int, float):
                raise TypeError(
                    "observed_losses_by_model_idの損失はbuiltin int/floatにしてください。boolは受理しません。"
                )
            try:
                observed_loss = float(observed_loss)
            except OverflowError as caught_exception:
                raise ValueError(
                    "observed_losses_by_model_idの損失は有限floatへ変換できる値にしてください。"
                ) from caught_exception
            if not math.isfinite(observed_loss):
                raise ValueError("observed_losses_by_model_idの損失は有限値にしてください。")
            observed_losses_by_model_id[model_id] = observed_loss
        return {model_id: observed_losses_by_model_id[model_id] for model_id in validated_model_ids}

    @staticmethod
    def _validated_prediction_weights(
        *,
        prediction_weights_by_model_id: Mapping[int, float],
    ) -> dict[int, float]:
        """確率の型・値域・総和を検証し、再正規化せずコピーを返す。"""
        if not isinstance(prediction_weights_by_model_id, Mapping):
            raise TypeError("prediction_weights_by_model_idにはMappingを指定してください。")
        validated_model_ids = FixedSharePredictionWeightController._validated_model_ids(
            model_ids=prediction_weights_by_model_id,
        )
        prediction_weights_by_model_id = dict(prediction_weights_by_model_id)
        for model_id in validated_model_ids:
            prediction_weight = prediction_weights_by_model_id[model_id]
            if type(prediction_weight) not in (int, float):
                raise TypeError(
                    "prediction_weights_by_model_idの確率はbuiltin int/floatにしてください。boolは受理しません。"
                )
            try:
                prediction_weight = float(prediction_weight)
            except OverflowError as caught_exception:
                raise ValueError(
                    "prediction_weights_by_model_idの確率は有限floatへ変換できる値にしてください。"
                ) from caught_exception
            if not math.isfinite(prediction_weight) or not 0.0 <= prediction_weight <= 1.0:
                raise ValueError(
                    "prediction_weights_by_model_idの確率は有限な0以上1以下の値にしてください。"
                )
            prediction_weights_by_model_id[model_id] = prediction_weight
        total_prediction_weight = math.fsum(prediction_weights_by_model_id.values())
        if abs(total_prediction_weight - 1.0) > 1e-12:
            raise ValueError(
                "prediction_weights_by_model_idの確率総和は1との差を1e-12以内にしてください。"
            )
        return {
            model_id: prediction_weights_by_model_id[model_id] for model_id in validated_model_ids
        }

    @staticmethod
    def select_maximum_weight_model_id(
        *,
        prediction_weights_by_model_id: Mapping[int, float],
        preferred_model_id: int | None = None,
    ) -> int:
        """同率候補なら優先ID、それ以外なら最小IDを状態変更なしで返す。"""
        if preferred_model_id is not None and type(preferred_model_id) is not int:
            raise TypeError(
                "preferred_model_idにはbuiltin intまたはNoneを指定してください。boolは受理しません。"
            )
        prediction_weights_by_model_id = (
            FixedSharePredictionWeightController._validated_prediction_weights(
                prediction_weights_by_model_id=prediction_weights_by_model_id,
            )
        )
        maximum_prediction_weight = max(prediction_weights_by_model_id.values())
        maximum_weight_model_ids = [
            model_id
            for model_id, prediction_weight in prediction_weights_by_model_id.items()
            if prediction_weight == maximum_prediction_weight
        ]
        if preferred_model_id in maximum_weight_model_ids:
            return preferred_model_id
        return min(maximum_weight_model_ids)

    def update_weights_after_loss_observation(
        self,
        *,
        observed_losses_by_model_id: Mapping[int, float],
        prediction_weights_by_model_id: Mapping[int, float],
    ) -> None:
        """全入力検証後、旧基準の昇順・演算順を保持して観測後重みを更新する。"""
        observed_losses_by_model_id = self._validated_observed_losses(
            observed_losses_by_model_id=observed_losses_by_model_id,
        )
        prediction_weights_by_model_id = self._validated_prediction_weights(
            prediction_weights_by_model_id=prediction_weights_by_model_id,
        )
        if set(observed_losses_by_model_id) != set(prediction_weights_by_model_id):
            raise ValueError(
                "observed_losses_by_model_idとprediction_weights_by_model_idは同じID集合にしてください。"
            )
        validated_model_ids = tuple(observed_losses_by_model_id)
        self._synchronize_model_pool(validated_model_ids=validated_model_ids)
        bounded_losses_by_model_id = {
            model_id: min(1.0, max(0.0, observed_losses_by_model_id[model_id]))
            for model_id in validated_model_ids
        }
        if len(validated_model_ids) == 1:
            return
        previous_leader_model_id = self.select_maximum_weight_model_id(
            prediction_weights_by_model_id=prediction_weights_by_model_id,
        )
        expected_observed_loss = sum(
            prediction_weights_by_model_id[model_id] * bounded_losses_by_model_id[model_id]
            for model_id in validated_model_ids
        )
        observed_loss_variance = sum(
            prediction_weights_by_model_id[model_id]
            * (bounded_losses_by_model_id[model_id] - expected_observed_loss) ** 2
            for model_id in validated_model_ids
        )
        self._cumulative_observed_loss_variance += observed_loss_variance
        learning_rate = min(
            1.0,
            math.sqrt(
                2.0
                * math.log(len(validated_model_ids))
                / max(self._cumulative_observed_loss_variance, 1e-12),
            ),
        )
        unnormalized_weights_by_model_id = {
            model_id: prediction_weights_by_model_id[model_id]
            * math.exp(-learning_rate * bounded_losses_by_model_id[model_id])
            for model_id in validated_model_ids
        }
        total_unnormalized_weight = sum(unnormalized_weights_by_model_id.values())
        posterior_weights_by_model_id = {
            model_id: unnormalized_weights_by_model_id[model_id] / total_unnormalized_weight
            for model_id in validated_model_ids
        }
        share_probability = (
            1.0
            / self._prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        )
        uniform_model_weight = 1.0 / len(validated_model_ids)
        self._weights_by_model_id = {
            model_id: (1.0 - share_probability) * posterior_weights_by_model_id[model_id]
            + share_probability * uniform_model_weight
            for model_id in validated_model_ids
        }
        if (
            self.select_maximum_weight_model_id(
                prediction_weights_by_model_id=self._weights_by_model_id,
            )
            != previous_leader_model_id
        ):
            self._prediction_weight_leader_switch_count += 1

    def _clear_prediction_weight_evidence(self) -> None:
        """重みと累積分散だけを消去し、診断計数を保持する。"""
        self._weights_by_model_id = {}
        self._cumulative_observed_loss_variance = 0.0

    @staticmethod
    def _validated_observed_loss_sequence(
        *,
        observed_loss_sequence: Iterable[Mapping[int, float]],
    ) -> tuple[dict[int, float], ...]:
        """後段や列挙の失敗より前に状態を変更しないため、全行をコピーする。"""
        return tuple(
            FixedSharePredictionWeightController._validated_observed_losses(
                observed_losses_by_model_id=observed_losses_by_model_id,
            )
            for observed_losses_by_model_id in observed_loss_sequence
        )

    def reset_weights_after_aggregation(self) -> None:
        """集約後の明示的resetを、証拠が空でも一回として計数する。"""
        self._clear_prediction_weight_evidence()
        self._aggregation_recalibration_count += 1

    def replay_observed_losses(
        self,
        *,
        observed_loss_sequence: Iterable[Mapping[int, float]],
    ) -> None:
        """全行を検証後に証拠を消去し、損失の列順で再構成する。"""
        validated_observed_loss_sequence = self._validated_observed_loss_sequence(
            observed_loss_sequence=observed_loss_sequence,
        )
        if not validated_observed_loss_sequence:
            return
        self._clear_prediction_weight_evidence()
        for observed_losses_by_model_id in validated_observed_loss_sequence:
            prediction_weights_by_model_id = self.get_prediction_weights_before_label_observation(
                model_ids=observed_losses_by_model_id,
            )
            self.update_weights_after_loss_observation(
                observed_losses_by_model_id=observed_losses_by_model_id,
                prediction_weights_by_model_id=prediction_weights_by_model_id,
            )

    def replay_observed_losses_after_aggregation(
        self,
        *,
        observed_loss_sequence: Iterable[Mapping[int, float]],
    ) -> None:
        """非空の集約後損失列だけを再較正・再生標本数へ加算する。"""
        validated_observed_loss_sequence = self._validated_observed_loss_sequence(
            observed_loss_sequence=observed_loss_sequence,
        )
        if not validated_observed_loss_sequence:
            return
        self._aggregation_recalibration_count += 1
        self._aggregation_recalibration_sample_count += len(validated_observed_loss_sequence)
        self.replay_observed_losses(observed_loss_sequence=validated_observed_loss_sequence)
