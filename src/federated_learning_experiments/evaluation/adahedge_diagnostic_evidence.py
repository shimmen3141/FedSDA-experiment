"""モデル推論から独立した単一AdaHedge診断証拠を所有する。"""

import math
from collections.abc import Iterable, Mapping


class AdaHedgeDiagnosticEvidence:
    """累積損失・gapと、再始動・再較正の計数を、独立した状態として保持する。"""

    def __init__(self) -> None:
        self._cumulative_losses_by_model_id: dict[int, float] = {}
        self._mixability_gap = 0.0
        self._model_pool_reset_count = 0
        self._concept_operation_restart_count = 0
        self._aggregation_restart_count = 0
        self._aggregation_recalibration_count = 0
        self._aggregation_recalibration_sample_count = 0

    @property
    def cumulative_losses_by_model_id(self) -> dict[int, float]:
        """呼出側で変更できる独立した累積損失のコピーを返す。"""
        return dict(self._cumulative_losses_by_model_id)

    @property
    def mixability_gap(self) -> float:
        return self._mixability_gap

    @property
    def model_pool_reset_count(self) -> int:
        return self._model_pool_reset_count

    @property
    def concept_operation_restart_count(self) -> int:
        return self._concept_operation_restart_count

    @property
    def aggregation_restart_count(self) -> int:
        """集約後に、証拠を消して始め直した回数。"""
        return self._aggregation_restart_count

    @property
    def aggregation_recalibration_count(self) -> int:
        """集約後の再較正（再始動と、空でない列の再生）の回数。"""
        return self._aggregation_recalibration_count

    @property
    def aggregation_recalibration_sample_count(self) -> int:
        """集約後の再生で使った損失の行の総数。"""
        return self._aggregation_recalibration_sample_count

    @staticmethod
    def _validate_model_ids(*, model_ids: Iterable[int]) -> tuple[int, ...]:
        """列挙を完了してから型・非空・重複を検査し昇順へ固定する。"""
        validated_model_ids = tuple(model_ids)
        for model_id in validated_model_ids:
            if type(model_id) is not int:
                raise TypeError("モデルIDはbuiltin intにしてください。boolは受理しません。")
        if not validated_model_ids:
            raise ValueError("モデルID集合は非空にしてください。")
        if len(set(validated_model_ids)) != len(validated_model_ids):
            raise ValueError("モデルIDの重複は受理しません。")
        return tuple(sorted(validated_model_ids))

    @staticmethod
    def _validate_numeric_mapping(
        *, mapping: Mapping[int, float], parameter_name: str
    ) -> dict[int, float]:
        """独立した入力辞書の全IDと有限数値を状態変更前に検証する。"""
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{parameter_name}にはMappingを指定してください。")
        validated_model_ids = AdaHedgeDiagnosticEvidence._validate_model_ids(model_ids=mapping)
        validated_mapping = {}
        for model_id in validated_model_ids:
            value = mapping[model_id]
            if type(value) not in (int, float):
                raise TypeError(f"{parameter_name}の値はbuiltin int/floatにしてください。")
            try:
                value = float(value)
            except OverflowError as caught_exception:
                raise ValueError(
                    f"{parameter_name}の値は有限floatへ変換可能にしてください。"
                ) from caught_exception
            if not math.isfinite(value):
                raise ValueError(f"{parameter_name}の値は有限値にしてください。")
            validated_mapping[model_id] = value
        return validated_mapping

    def _prepare_synchronized_evidence(
        self, *, validated_model_ids: tuple[int, ...]
    ) -> tuple[dict[int, float], float, int]:
        """集合同期後の状態候補だけを作り、ownerを更新しない。"""
        if set(validated_model_ids) != set(self._cumulative_losses_by_model_id):
            return (
                {model_id: 0.0 for model_id in validated_model_ids},
                0.0,
                self._model_pool_reset_count + int(bool(self._cumulative_losses_by_model_id)),
            )
        return (
            dict(self._cumulative_losses_by_model_id),
            self._mixability_gap,
            self._model_pool_reset_count,
        )

    @staticmethod
    def _get_evidence_learning_rate(
        *, proposed_cumulative_losses: dict[int, float], proposed_mixability_gap: float
    ) -> float:
        if len(proposed_cumulative_losses) <= 1 or proposed_mixability_gap <= 0.0:
            return math.inf
        return math.log(len(proposed_cumulative_losses)) / proposed_mixability_gap

    def _commit_evidence(
        self,
        *,
        proposed_cumulative_losses: dict[int, float],
        proposed_mixability_gap: float,
        proposed_pool_reset_count: int,
    ) -> None:
        self._cumulative_losses_by_model_id = proposed_cumulative_losses
        self._mixability_gap = proposed_mixability_gap
        self._model_pool_reset_count = proposed_pool_reset_count

    def get_diagnostic_weights_before_loss_observation(
        self, *, model_ids: Iterable[int]
    ) -> dict[int, float]:
        """全演算成功後に集合を同期し、観測前の診断重みを返す。"""
        validated_model_ids = self._validate_model_ids(model_ids=model_ids)
        proposed_cumulative_losses, proposed_mixability_gap, proposed_pool_reset_count = (
            self._prepare_synchronized_evidence(validated_model_ids=validated_model_ids)
        )
        learning_rate = self._get_evidence_learning_rate(
            proposed_cumulative_losses=proposed_cumulative_losses,
            proposed_mixability_gap=proposed_mixability_gap,
        )
        minimum_cumulative_loss = min(proposed_cumulative_losses.values())
        if len(validated_model_ids) == 1:
            diagnostic_weights_by_model_id = {validated_model_ids[0]: 1.0}
        elif math.isinf(learning_rate):
            minimum_loss_model_ids = [
                model_id
                for model_id in validated_model_ids
                if proposed_cumulative_losses[model_id] == minimum_cumulative_loss
            ]
            weight = 1.0 / len(minimum_loss_model_ids)
            diagnostic_weights_by_model_id = {
                model_id: weight if model_id in minimum_loss_model_ids else 0.0
                for model_id in validated_model_ids
            }
        else:
            unnormalized_weights = {
                model_id: math.exp(
                    -learning_rate
                    * (proposed_cumulative_losses[model_id] - minimum_cumulative_loss)
                )
                for model_id in validated_model_ids
            }
            weight_sum = sum(unnormalized_weights.values())
            diagnostic_weights_by_model_id = {
                model_id: value / weight_sum for model_id, value in unnormalized_weights.items()
            }
        self._commit_evidence(
            proposed_cumulative_losses=proposed_cumulative_losses,
            proposed_mixability_gap=proposed_mixability_gap,
            proposed_pool_reset_count=proposed_pool_reset_count,
        )
        return diagnostic_weights_by_model_id

    def update_evidence_after_loss_observation(
        self,
        *,
        observed_losses_by_model_id: Mapping[int, float],
        diagnostic_weights_by_model_id: Mapping[int, float],
    ) -> None:
        """全入力を先行検証し、旧実装の昇順・演算順で証拠を更新する。"""
        if not isinstance(observed_losses_by_model_id, Mapping):
            raise TypeError("observed_losses_by_model_idにはMappingを指定してください。")
        if not isinstance(diagnostic_weights_by_model_id, Mapping):
            raise TypeError("diagnostic_weights_by_model_idにはMappingを指定してください。")
        observed_losses_by_model_id = dict(observed_losses_by_model_id)
        diagnostic_weights_by_model_id = dict(diagnostic_weights_by_model_id)
        validated_losses = self._validate_numeric_mapping(
            mapping=observed_losses_by_model_id, parameter_name="observed_losses_by_model_id"
        )
        validated_diagnostic_weights = self._validate_numeric_mapping(
            mapping=diagnostic_weights_by_model_id, parameter_name="diagnostic_weights_by_model_id"
        )
        if set(validated_losses) != set(validated_diagnostic_weights):
            raise ValueError("損失と診断重みは同じID集合にしてください。")
        if any(not 0.0 <= weight <= 1.0 for weight in validated_diagnostic_weights.values()):
            raise ValueError("診断重みは0から1の範囲にしてください。")
        if abs(math.fsum(validated_diagnostic_weights.values()) - 1.0) > 1e-12:
            raise ValueError("診断重みの総和は1との差を1e-12以内にしてください。")
        validated_model_ids = tuple(validated_losses)
        proposed_cumulative_losses, proposed_mixability_gap, proposed_pool_reset_count = (
            self._prepare_synchronized_evidence(validated_model_ids=validated_model_ids)
        )
        bounded_losses_by_model_id = {
            model_id: min(1.0, max(0.0, float(validated_losses[model_id])))
            for model_id in validated_model_ids
        }
        expected_loss = sum(
            validated_diagnostic_weights[model_id] * bounded_losses_by_model_id[model_id]
            for model_id in validated_model_ids
        )
        learning_rate = self._get_evidence_learning_rate(
            proposed_cumulative_losses=proposed_cumulative_losses,
            proposed_mixability_gap=proposed_mixability_gap,
        )
        if math.isinf(learning_rate):
            active_losses = [
                bounded_losses_by_model_id[model_id]
                for model_id in validated_model_ids
                if validated_diagnostic_weights[model_id] > 0.0
            ]
            mix_loss = min(active_losses)
        else:
            log_terms = [
                math.log(validated_diagnostic_weights[model_id])
                - learning_rate * bounded_losses_by_model_id[model_id]
                for model_id in validated_model_ids
                if validated_diagnostic_weights[model_id] > 0.0
            ]
            maximum_log_term = max(log_terms)
            log_mixture = maximum_log_term + math.log(
                sum(math.exp(value - maximum_log_term) for value in log_terms)
            )
            mix_loss = -log_mixture / learning_rate
        proposed_mixability_gap += max(0.0, expected_loss - mix_loss)
        for model_id in validated_model_ids:
            proposed_cumulative_losses[model_id] += bounded_losses_by_model_id[model_id]
        self._commit_evidence(
            proposed_cumulative_losses=proposed_cumulative_losses,
            proposed_mixability_gap=proposed_mixability_gap,
            proposed_pool_reset_count=proposed_pool_reset_count,
        )

    def restart_evidence_after_concept_operation(self) -> None:
        """証拠だけを消去し、空状態でも概念操作の再始動を一回として数える。"""
        self._cumulative_losses_by_model_id = {}
        self._mixability_gap = 0.0
        self._concept_operation_restart_count += 1

    def restart_evidence_after_aggregation(self) -> None:
        """集約の前の比較の証拠を消し、空状態でも、集約後の再始動と再較正を一回ずつ数える。"""
        self._cumulative_losses_by_model_id = {}
        self._mixability_gap = 0.0
        self._aggregation_restart_count += 1
        self._aggregation_recalibration_count += 1

    def replay_observed_losses_after_aggregation(
        self, *, observed_loss_sequence: Iterable[Mapping[int, float]]
    ) -> None:
        """全行を検証した後、空でない列だけを計数へ足し、証拠を消して、列の順に再構成する。"""
        validated_observed_loss_sequence = tuple(
            self._validate_numeric_mapping(
                mapping=observed_losses_by_model_id,
                parameter_name="observed_loss_sequenceの行",
            )
            for observed_losses_by_model_id in observed_loss_sequence
        )
        if not validated_observed_loss_sequence:
            return
        self._aggregation_recalibration_count += 1
        self._aggregation_recalibration_sample_count += len(validated_observed_loss_sequence)
        self._cumulative_losses_by_model_id = {}
        self._mixability_gap = 0.0
        for observed_losses_by_model_id in validated_observed_loss_sequence:
            diagnostic_weights_by_model_id = self.get_diagnostic_weights_before_loss_observation(
                model_ids=observed_losses_by_model_id
            )
            self.update_evidence_after_loss_observation(
                observed_losses_by_model_id=observed_losses_by_model_id,
                diagnostic_weights_by_model_id=diagnostic_weights_by_model_id,
            )
