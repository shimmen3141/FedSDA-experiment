"""警報後の有界lossから既存参照の適合性と候補採否を評価する。"""

import math

import torch


def _validate_nonnegative_finite_number(*, specified_value: float, parameter_name: str) -> float:
    """boolや非有限値を拒否し、非負の明示閾値を返す。"""
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name} must be a builtin int or float")
    try:
        specified_value = float(specified_value)
    except OverflowError:
        raise ValueError(f"{parameter_name} must be representable as a finite float") from None
    if not math.isfinite(specified_value) or specified_value < 0:
        raise ValueError(f"{parameter_name} must be finite and nonnegative")
    return specified_value


def _validate_model_id(*, model_id: int, parameter_name: str) -> None:
    """負IDを含む整数IDを検証し、boolを除く。"""
    if type(model_id) is not int:
        raise TypeError(f"{parameter_name} must be a builtin int model ID")


def _validate_bounded_loss(*, specified_value: float, parameter_name: str) -> None:
    """有限の有界lossを強制変換せず検証する。"""
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name} must be a builtin int or float")
    if not 0 <= specified_value <= 1 or not math.isfinite(specified_value):
        raise ValueError(f"{parameter_name} must be finite and in [0, 1]")


def _validate_loss_sequence(*, losses: tuple[float, ...], parameter_name: str) -> None:
    """観測順の不変loss列を検証する。"""
    if type(losses) is not tuple:
        raise TypeError(f"{parameter_name} must be a tuple")
    if len(losses) < 2:
        raise ValueError(f"{parameter_name} must contain at least two observations")
    for specified_value in losses:
        _validate_bounded_loss(specified_value=specified_value, parameter_name=parameter_name)


def _validate_reference_loss_inputs(
    *,
    reference_losses_by_model_id: dict[int, tuple[float, ...]],
    reference_historical_mean_losses_by_model_id: dict[int, float],
    available_reference_model_ids: tuple[int, ...],
    current_training_model_id: int,
    maximum_reference_mean_loss_increase: float,
) -> None:
    """平均計算の前に参照入力の全契約を確認する。"""
    if type(reference_losses_by_model_id) is not dict:
        raise TypeError("reference_losses_by_model_id must be a dict")
    if not reference_losses_by_model_id:
        raise ValueError("reference_losses_by_model_id must not be empty")
    validation_sample_count = None
    for model_id, losses in reference_losses_by_model_id.items():
        _validate_model_id(model_id=model_id, parameter_name="reference_losses_by_model_id")
        _validate_loss_sequence(losses=losses, parameter_name="reference_losses_by_model_id")
        if validation_sample_count is None:
            validation_sample_count = len(losses)
        elif len(losses) != validation_sample_count:
            raise ValueError("reference_losses_by_model_id must have equal observation counts")
    if type(reference_historical_mean_losses_by_model_id) is not dict:
        raise TypeError("reference_historical_mean_losses_by_model_id must be a dict")
    for model_id, specified_value in reference_historical_mean_losses_by_model_id.items():
        _validate_model_id(model_id=model_id, parameter_name="reference_historical_mean_losses_by_model_id")
        _validate_bounded_loss(specified_value=specified_value, parameter_name="reference_historical_mean_losses_by_model_id")
    if type(available_reference_model_ids) is not tuple:
        raise TypeError("available_reference_model_ids must be a tuple")
    for model_id in available_reference_model_ids:
        _validate_model_id(model_id=model_id, parameter_name="available_reference_model_ids")
    if len(set(available_reference_model_ids)) != len(available_reference_model_ids):
        raise ValueError("available_reference_model_ids must not contain duplicate IDs")
    _validate_model_id(model_id=current_training_model_id, parameter_name="current_training_model_id")
    _validate_nonnegative_finite_number(specified_value=maximum_reference_mean_loss_increase, parameter_name="maximum_reference_mean_loss_increase")


def _mean_bounded_loss(*, losses: tuple[float, ...]) -> float:
    """共有default dtype/deviceに依存せず旧CPUfloat32平均を返す。"""
    return float(torch.tensor(losses, dtype=torch.float32, device="cpu").mean().item())


def _select_validated_reusable_reference(
    *,
    reference_losses_by_model_id: dict[int, tuple[float, ...]],
    reference_historical_mean_losses_by_model_id: dict[int, float],
    available_reference_model_ids: tuple[int, ...],
    current_training_model_id: int,
    maximum_reference_mean_loss_increase: float,
) -> int | None:
    """検査済み参照から現行優先・平均/ID同率順で適合IDを選ぶ。"""
    available_model_ids = set(available_reference_model_ids)
    fitting_references = []
    for model_id, losses in reference_losses_by_model_id.items():
        if model_id not in available_model_ids or model_id not in reference_historical_mean_losses_by_model_id:
            continue
        mean_loss = _mean_bounded_loss(losses=losses)
        historical_mean_loss = float(reference_historical_mean_losses_by_model_id[model_id])
        if mean_loss - historical_mean_loss <= maximum_reference_mean_loss_increase:
            fitting_references.append((mean_loss, model_id))
    if current_training_model_id in {model_id for mean_loss, model_id in fitting_references}:
        return current_training_model_id
    return min(fitting_references)[1] if fitting_references else None


def select_available_reference_within_historical_loss_tolerance(
    *,
    reference_losses_by_model_id: dict[int, tuple[float, ...]],
    reference_historical_mean_losses_by_model_id: dict[int, float],
    available_reference_model_ids: tuple[int, ...],
    current_training_model_id: int,
    maximum_reference_mean_loss_increase: float,
) -> int | None:
    """利用可能参照の履歴平均超過を判定し、適合IDだけを返す。"""
    _validate_reference_loss_inputs(
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=current_training_model_id,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
    )
    return _select_validated_reusable_reference(
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=current_training_model_id,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
    )
