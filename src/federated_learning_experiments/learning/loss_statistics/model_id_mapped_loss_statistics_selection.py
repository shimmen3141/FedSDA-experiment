"""一回のモデルID対応後に損失統計を丸ごと選択する。"""

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)


def _copy_validated_model_loss_statistics(
    *,
    loss_statistics_snapshot: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    parameter_name: str,
) -> tuple[tuple[int, ModelAndClassLossStatistics], ...]:
    if type(loss_statistics_snapshot) is not tuple:
        raise TypeError(f"{parameter_name}はexact tupleが必要です。")
    copied_model_loss_statistics_snapshot: list[tuple[int, ModelAndClassLossStatistics]] = []
    seen_model_ids: set[int] = set()
    for model_loss_statistics_pair in loss_statistics_snapshot:
        if type(model_loss_statistics_pair) is not tuple:
            raise TypeError(f"{parameter_name}の各要素はexact tupleが必要です。")
        if len(model_loss_statistics_pair) != 2:
            raise ValueError(f"{parameter_name}の各要素は2要素が必要です。")
        model_id, loss_statistics = model_loss_statistics_pair
        if type(model_id) is not int:
            raise TypeError(f"{parameter_name}のmodel_idはbool以外のbuiltin intが必要です。")
        if model_id in seen_model_ids:
            raise ValueError(f"{parameter_name}のmodel_idは重複できません。")
        seen_model_ids.add(model_id)
        if type(loss_statistics) is not ModelAndClassLossStatistics:
            raise TypeError(
                f"{parameter_name}のloss_statisticsはModelAndClassLossStatisticsのexact型が必要です。"
            )
        try:
            copied_loss_statistics = ModelAndClassLossStatistics(
                overall_loss_moments=loss_statistics.overall_loss_moments,
                class_loss_moments_by_class_id=loss_statistics.class_loss_moments_by_class_id,
            )
        except (TypeError, ValueError) as validation_error:
            raise type(validation_error)(
                f"{parameter_name}: {validation_error}"
            ) from validation_error
        copied_model_loss_statistics_snapshot.append((model_id, copied_loss_statistics))
    return tuple(copied_model_loss_statistics_snapshot)


def _validate_model_id_mapping(*, model_id_mapping: dict[int, int]) -> None:
    if type(model_id_mapping) is not dict:
        raise TypeError("model_id_mappingはexact dictが必要です。")
    for original_model_id, mapped_model_id in model_id_mapping.items():
        if type(original_model_id) is not int:
            raise TypeError("model_id_mappingの元IDはbool以外のbuiltin intが必要です。")
        if type(mapped_model_id) is not int:
            raise TypeError("model_id_mappingの変更先IDはbool以外のbuiltin intが必要です。")


def select_loss_statistics_after_model_id_mapping(
    *,
    local_model_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    model_id_mapping: dict[int, int],
    server_model_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...] | None = None,
) -> tuple[tuple[int, ModelAndClassLossStatistics], ...]:
    """最大全体件数のローカル統計を選び、欠落・0件だけサーバで補完する。"""
    validated_local_model_loss_statistics = _copy_validated_model_loss_statistics(
        loss_statistics_snapshot=local_model_loss_statistics,
        parameter_name="local_model_loss_statistics",
    )
    validated_server_model_loss_statistics = _copy_validated_model_loss_statistics(
        loss_statistics_snapshot=()
        if server_model_loss_statistics is None
        else server_model_loss_statistics,
        parameter_name="server_model_loss_statistics",
    )
    _validate_model_id_mapping(model_id_mapping=model_id_mapping)
    selected_loss_statistics_by_model_id: dict[int, ModelAndClassLossStatistics] = {}
    for original_model_id, loss_statistics in validated_local_model_loss_statistics:
        mapped_model_id = model_id_mapping.get(original_model_id, original_model_id)
        selected_loss_statistics = selected_loss_statistics_by_model_id.get(mapped_model_id)
        if (
            selected_loss_statistics is None
            or loss_statistics.overall_loss_moments.observed_loss_count
            > selected_loss_statistics.overall_loss_moments.observed_loss_count
        ):
            selected_loss_statistics_by_model_id[mapped_model_id] = loss_statistics
    for model_id, loss_statistics in validated_server_model_loss_statistics:
        selected_loss_statistics = selected_loss_statistics_by_model_id.get(model_id)
        if (
            selected_loss_statistics is None
            or selected_loss_statistics.overall_loss_moments.observed_loss_count == 0
        ):
            selected_loss_statistics_by_model_id[model_id] = loss_statistics
    return tuple(selected_loss_statistics_by_model_id.items())
