"""警報後に算出された候補・固定参照の有界損失を収集する。"""

import math
from dataclasses import dataclass

from .candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateLossCollectionState:
    """内部の可変系列を公開しない途中・完了の診断copy。"""

    proposal_sample_index: int
    last_validation_sample_index: int | None
    candidate_losses: tuple[float, ...]
    reference_losses_by_model_id: tuple[tuple[int, tuple[float, ...]], ...]
    validation_sample_count: int
    required_validation_sample_count: int
    ready_for_acceptance_evaluation: bool


def _validate_bounded_observed_loss(*, specified_value: float, parameter_name: str) -> float:
    """暗黙の文字列/配列変換をせず、検査後のfloatを返す。"""
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name} must be a builtin int or float, excluding bool")
    if not 0 <= specified_value <= 1 or not math.isfinite(specified_value):
        raise ValueError(f"{parameter_name} must be finite and in [0, 1]")
    return float(specified_value)


class PostAlarmCandidateLossCollection:
    """モデルやpayloadを持たず、提案次位置から規定件数を収集する。"""

    def __init__(
        self,
        *,
        candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
        proposal_sample_index: int,
        reference_model_ids: tuple[int, ...],
    ) -> None:
        if (
            type(candidate_model_training_and_acceptance_settings)
            is not CandidateModelTrainingAndAcceptanceSettings
        ):
            raise TypeError(
                "candidate_model_training_and_acceptance_settings must be CandidateModelTrainingAndAcceptanceSettings"
            )
        candidate_model_training_and_acceptance_settings.__post_init__()
        if type(proposal_sample_index) is not int:
            raise TypeError("proposal_sample_index must be a builtin int, excluding bool")
        if proposal_sample_index < 0:
            raise ValueError("proposal_sample_index must be nonnegative")
        if type(reference_model_ids) is not tuple:
            raise TypeError("reference_model_ids must be a tuple")
        if not reference_model_ids:
            raise ValueError("reference_model_ids must not be empty")
        if any(type(model_id) is not int for model_id in reference_model_ids):
            raise TypeError("reference_model_ids must contain builtin ints, excluding bool")
        if len(set(reference_model_ids)) != len(reference_model_ids):
            raise ValueError("reference_model_ids must not contain duplicate IDs")
        self._candidate_model_training_and_acceptance_settings = (
            candidate_model_training_and_acceptance_settings
        )
        self._proposal_sample_index = proposal_sample_index
        self._reference_model_ids = reference_model_ids
        self._candidate_losses: list[float] = []
        self._reference_losses_by_model_id: dict[int, list[float]] = {
            model_id: [] for model_id in reference_model_ids
        }
        self._last_validation_sample_index: int | None = None

    @property
    def validation_sample_count(self) -> int:
        """系列に実際に追加された検証標本件数。"""
        return len(self._candidate_losses)

    @property
    def ready_for_acceptance_evaluation(self) -> bool:
        """採否を自動起動せず、規定件数への到達だけを返す。"""
        return (
            self.validation_sample_count
            >= self._candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count
        )

    def observe_losses_after_label_observation(
        self,
        *,
        sample_index: int,
        candidate_loss: float,
        reference_losses_by_model_id: dict[int, float],
    ) -> None:
        """全入力の検査・float化後に、同じ観測回を全系列へ追加する。"""
        if self.ready_for_acceptance_evaluation:
            raise RuntimeError("collection is already ready_for_acceptance_evaluation")
        if type(sample_index) is not int:
            raise TypeError("sample_index must be a builtin int, excluding bool")
        if (
            sample_index
            != (
                self._proposal_sample_index
                if self._last_validation_sample_index is None
                else self._last_validation_sample_index
            )
            + 1
        ):
            raise ValueError(
                "sample_index must immediately follow proposal_sample_index or last_validation_sample_index"
            )
        candidate_loss = _validate_bounded_observed_loss(
            specified_value=candidate_loss, parameter_name="candidate_loss"
        )
        if type(reference_losses_by_model_id) is not dict:
            raise TypeError("reference_losses_by_model_id must be a dict")
        if any(type(model_id) is not int for model_id in reference_losses_by_model_id):
            raise TypeError("reference_losses_by_model_id must use builtin int IDs, excluding bool")
        if set(reference_losses_by_model_id) != set(self._reference_model_ids):
            raise ValueError(
                "reference_losses_by_model_id must contain exactly the initial reference IDs"
            )
        validated_reference_losses_by_model_id = {
            model_id: _validate_bounded_observed_loss(
                specified_value=reference_losses_by_model_id[model_id],
                parameter_name=f"reference_losses_by_model_id[{model_id}]",
            )
            for model_id in self._reference_model_ids
        }
        self._candidate_losses.append(candidate_loss)
        for model_id in self._reference_model_ids:
            self._reference_losses_by_model_id[model_id].append(
                validated_reference_losses_by_model_id[model_id]
            )
        self._last_validation_sample_index = sample_index

    def get_state_snapshot(self) -> PostAlarmCandidateLossCollectionState:
        """規定未満でも損失を消去せず、独立した変更不能なcopyを返す。"""
        return PostAlarmCandidateLossCollectionState(
            proposal_sample_index=self._proposal_sample_index,
            last_validation_sample_index=self._last_validation_sample_index,
            candidate_losses=tuple(self._candidate_losses),
            reference_losses_by_model_id=tuple(
                (model_id, tuple(self._reference_losses_by_model_id[model_id]))
                for model_id in self._reference_model_ids
            ),
            validation_sample_count=self.validation_sample_count,
            required_validation_sample_count=self._candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count,
            ready_for_acceptance_evaluation=self.ready_for_acceptance_evaluation,
        )
