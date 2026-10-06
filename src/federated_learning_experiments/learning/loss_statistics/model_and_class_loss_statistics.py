"""モデル全体・正解クラス別の帰属損失統計を独立保持する。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
    accumulate_bounded_loss_observation,
)


def _validate_identifier(
    *,
    identifier: int,
    parameter_name: str,
    minimum_value: int | None = None,
) -> None:
    if type(identifier) is not int:
        raise TypeError(f"{parameter_name}はbool以外のbuiltin intが必要です。")
    if minimum_value is not None and identifier < minimum_value:
        raise ValueError(f"{parameter_name}は{minimum_value}以上が必要です。")


def _copy_loss_moments(*, loss_moments: BoundedLossMoments) -> BoundedLossMoments:
    if type(loss_moments) is not BoundedLossMoments:
        raise TypeError("loss_momentsはBoundedLossMomentsのexact型が必要です。")
    return BoundedLossMoments(
        observed_loss_count=loss_moments.observed_loss_count,
        mean_loss=loss_moments.mean_loss,
        sum_squared_loss_deviations=loss_moments.sum_squared_loss_deviations,
    )


@dataclass(frozen=True, kw_only=True)
class ModelAndClassLossStatistics:
    """一モデルの全体集計と到着順のクラス別集計。"""

    overall_loss_moments: BoundedLossMoments
    class_loss_moments_by_class_id: tuple[tuple[int, BoundedLossMoments], ...] = ()

    def __post_init__(self) -> None:
        copied_overall_loss_moments = _copy_loss_moments(loss_moments=self.overall_loss_moments)
        if type(self.class_loss_moments_by_class_id) is not tuple:
            raise TypeError("class_loss_moments_by_class_idはexact tupleが必要です。")
        copied_class_loss_moments: list[tuple[int, BoundedLossMoments]] = []
        seen_class_ids: set[int] = set()
        for class_loss_moments_pair in self.class_loss_moments_by_class_id:
            if type(class_loss_moments_pair) is not tuple:
                raise TypeError("class_loss_moments_by_class_idの各要素はexact tupleが必要です。")
            if len(class_loss_moments_pair) != 2:
                raise ValueError("class_loss_moments_by_class_idの各要素は2要素が必要です。")
            class_id, class_loss_moments = class_loss_moments_pair
            _validate_identifier(identifier=class_id, parameter_name="class_id", minimum_value=0)
            if class_id in seen_class_ids:
                raise ValueError("class_idは重複できません。")
            seen_class_ids.add(class_id)
            copied_class_loss_moments.append(
                (class_id, _copy_loss_moments(loss_moments=class_loss_moments))
            )
        object.__setattr__(self, "overall_loss_moments", copied_overall_loss_moments)
        object.__setattr__(self, "class_loss_moments_by_class_id", tuple(copied_class_loss_moments))


def _copy_model_and_class_loss_statistics(
    *,
    loss_statistics: ModelAndClassLossStatistics,
) -> ModelAndClassLossStatistics:
    if type(loss_statistics) is not ModelAndClassLossStatistics:
        raise TypeError("loss_statisticsはModelAndClassLossStatisticsのexact型が必要です。")
    return ModelAndClassLossStatistics(
        overall_loss_moments=loss_statistics.overall_loss_moments,
        class_loss_moments_by_class_id=loss_statistics.class_loss_moments_by_class_id,
    )


class ModelAndClassLossStatisticsStore:
    """一所有者のモデル別統計を保持し、全更新候補の検査後に確定する。"""

    def __init__(
        self,
        *,
        initial_loss_statistics_by_model_id: dict[int, ModelAndClassLossStatistics] | None = None,
    ) -> None:
        validated_model_loss_statistics_by_model_id: dict[int, ModelAndClassLossStatistics] = {}
        if initial_loss_statistics_by_model_id is not None:
            if type(initial_loss_statistics_by_model_id) is not dict:
                raise TypeError(
                    "initial_loss_statistics_by_model_idはexact dictまたはNoneが必要です。"
                )
            for model_id, loss_statistics in initial_loss_statistics_by_model_id.items():
                _validate_identifier(identifier=model_id, parameter_name="model_id")
                validated_model_loss_statistics_by_model_id[model_id] = (
                    _copy_model_and_class_loss_statistics(loss_statistics=loss_statistics)
                )
        self._model_loss_statistics_by_model_id = validated_model_loss_statistics_by_model_id

    def set_model_loss_statistics(
        self,
        *,
        model_id: int,
        loss_statistics: ModelAndClassLossStatistics,
    ) -> None:
        """全体と全クラスを一括置換し、既存モデルの位置を維持する。"""
        _validate_identifier(identifier=model_id, parameter_name="model_id")
        loss_statistics = _copy_model_and_class_loss_statistics(loss_statistics=loss_statistics)
        self._model_loss_statistics_by_model_id[model_id] = loss_statistics

    def reassign_model_loss_statistics_id(
        self,
        *,
        original_model_id: int,
        reassigned_model_id: int,
    ) -> None:
        """一モデルの統計を移し、既存先は位置保持で上書き、同IDは末尾へ置く。"""
        _validate_identifier(identifier=original_model_id, parameter_name="original_model_id")
        _validate_identifier(identifier=reassigned_model_id, parameter_name="reassigned_model_id")
        if original_model_id in self._model_loss_statistics_by_model_id:
            loss_statistics = self._model_loss_statistics_by_model_id.pop(original_model_id)
            self._model_loss_statistics_by_model_id[reassigned_model_id] = loss_statistics

    def record_assigned_loss(
        self,
        *,
        model_id: int,
        observed_loss: float,
        observed_class_id: int | None = None,
    ) -> None:
        """全体と指定クラスの更新を算出・検査してから一回だけ保存する。"""
        _validate_identifier(identifier=model_id, parameter_name="model_id")
        if observed_class_id is not None:
            _validate_identifier(
                identifier=observed_class_id, parameter_name="observed_class_id", minimum_value=0
            )
        loss_statistics = self._model_loss_statistics_by_model_id.get(model_id)
        if loss_statistics is None:
            loss_statistics = ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                )
            )
        else:
            loss_statistics = _copy_model_and_class_loss_statistics(loss_statistics=loss_statistics)
        updated_overall_loss_moments = accumulate_bounded_loss_observation(
            loss_moments=loss_statistics.overall_loss_moments, observed_loss=observed_loss
        )
        updated_class_loss_moments_by_class_id = dict(
            loss_statistics.class_loss_moments_by_class_id
        )
        if observed_class_id is not None:
            class_loss_moments = updated_class_loss_moments_by_class_id.get(observed_class_id)
            if class_loss_moments is None:
                class_loss_moments = BoundedLossMoments(
                    observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                )
            updated_class_loss_moments = accumulate_bounded_loss_observation(
                loss_moments=class_loss_moments, observed_loss=observed_loss
            )
            updated_class_loss_moments_by_class_id[observed_class_id] = updated_class_loss_moments
        loss_statistics = ModelAndClassLossStatistics(
            overall_loss_moments=updated_overall_loss_moments,
            class_loss_moments_by_class_id=tuple(updated_class_loss_moments_by_class_id.items()),
        )
        self._model_loss_statistics_by_model_id[model_id] = loss_statistics

    def get_model_loss_statistics(self, *, model_id: int) -> ModelAndClassLossStatistics | None:
        """未登録はNone、登録済み値は全fieldの独立コピーを返す。"""
        _validate_identifier(identifier=model_id, parameter_name="model_id")
        loss_statistics = self._model_loss_statistics_by_model_id.get(model_id)
        if loss_statistics is None:
            return None
        return _copy_model_and_class_loss_statistics(loss_statistics=loss_statistics)

    def get_state_snapshot(self) -> tuple[tuple[int, ModelAndClassLossStatistics], ...]:
        """受取順を保った全モデルの独立した不変参照を返す。"""
        return tuple(
            (model_id, _copy_model_and_class_loss_statistics(loss_statistics=loss_statistics))
            for model_id, loss_statistics in self._model_loss_statistics_by_model_id.items()
        )
