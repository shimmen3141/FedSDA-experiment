"""標本ごとの予測の記録（結合した予測の正否、最大重みのモデル、診断の項目）を観測順に持つ。"""

from dataclasses import dataclass
from math import isfinite

# 実効モデル数（重みの二乗和の逆数）が、丸めで1を下回ることを許す幅。
_EFFECTIVE_MODEL_COUNT_ROUNDING_TOLERANCE = 1e-9


@dataclass(frozen=True, kw_only=True)
class SamplePredictionRecord:
    """標本1件の予測の結果。正否は、その標本の観測ラベルと比べた結果。入力の検査はownerが行う。

    「結合した予測」は、Fixed-Shareの予測重みで全モデルの確率を結合した予測。診断の項目は、
    globalと真の概念別の診断重みで結合した予測の正否で、予測と学習の判断には使わない。
    """

    sample_index: int
    observed_concept_id: int | None
    observed_class_id: int
    combined_prediction_is_correct: bool
    maximum_weight_model_id: int
    maximum_prediction_weight: float
    effective_model_count: float
    any_model_or_combined_prediction_is_correct: bool
    maximum_weight_model_prediction_is_correct: bool
    global_diagnostic_prediction_is_correct: bool
    true_concept_diagnostic_prediction_is_correct: bool | None
    highest_confidence_model_prediction_is_correct: bool


_REQUIRED_BOOL_FIELD_NAMES = (
    "combined_prediction_is_correct",
    "any_model_or_combined_prediction_is_correct",
    "maximum_weight_model_prediction_is_correct",
    "global_diagnostic_prediction_is_correct",
    "highest_confidence_model_prediction_is_correct",
)


def _validate_sample_prediction_record(
    *, sample_prediction_record: SamplePredictionRecord, last_recorded_sample_index: int | None
) -> None:
    if type(sample_prediction_record) is not SamplePredictionRecord:
        raise TypeError("sample_prediction_record must be exact SamplePredictionRecord")
    for int_field_name in ("sample_index", "observed_class_id", "maximum_weight_model_id"):
        if type(getattr(sample_prediction_record, int_field_name)) is not int:
            raise TypeError(f"{int_field_name} must be builtin int")
    observed_concept_id = sample_prediction_record.observed_concept_id
    if observed_concept_id is not None and type(observed_concept_id) is not int:
        raise TypeError("observed_concept_id must be builtin int or None")
    for bool_field_name in _REQUIRED_BOOL_FIELD_NAMES:
        if type(getattr(sample_prediction_record, bool_field_name)) is not bool:
            raise TypeError(f"{bool_field_name} must be builtin bool")
    true_concept_diagnostic_prediction_is_correct = (
        sample_prediction_record.true_concept_diagnostic_prediction_is_correct
    )
    if (
        true_concept_diagnostic_prediction_is_correct is not None
        and type(true_concept_diagnostic_prediction_is_correct) is not bool
    ):
        raise TypeError(
            "true_concept_diagnostic_prediction_is_correct must be builtin bool or None"
        )
    for float_field_name in ("maximum_prediction_weight", "effective_model_count"):
        if type(getattr(sample_prediction_record, float_field_name)) is not float:
            raise TypeError(f"{float_field_name} must be builtin float")
    if sample_prediction_record.sample_index < 0 or sample_prediction_record.observed_class_id < 0:
        raise ValueError("sample_index and observed_class_id must be nonnegative")
    if not 0.0 < sample_prediction_record.maximum_prediction_weight <= 1.0:
        raise ValueError("maximum_prediction_weight must be greater than 0 and at most 1")
    if not (
        isfinite(sample_prediction_record.effective_model_count)
        and sample_prediction_record.effective_model_count
        >= 1.0 - _EFFECTIVE_MODEL_COUNT_ROUNDING_TOLERANCE
    ):
        raise ValueError("effective_model_count must be finite and at least 1")
    if (observed_concept_id is None) != (true_concept_diagnostic_prediction_is_correct is None):
        raise ValueError(
            "true concept diagnostic result must be held exactly when the concept ID is held"
        )
    if (
        last_recorded_sample_index is not None
        and sample_prediction_record.sample_index != last_recorded_sample_index + 1
    ):
        raise ValueError("sample_index must follow the last recorded sample index by one")


class SamplePredictionRecordStore:
    """記録だけを所有し、予測や判断をしない。"""

    def __init__(self) -> None:
        self._sample_prediction_records: list[SamplePredictionRecord] = []

    @property
    def last_recorded_sample_index(self) -> int | None:
        """最後に足した記録の位置。記録がなければNone。"""
        if not self._sample_prediction_records:
            return None
        return self._sample_prediction_records[-1].sample_index

    def append_sample_prediction_record(
        self, *, sample_prediction_record: SamplePredictionRecord
    ) -> None:
        """記録を検査して末尾へ足す。位置は、最後の記録の位置の次であること。"""
        _validate_sample_prediction_record(
            sample_prediction_record=sample_prediction_record,
            last_recorded_sample_index=self.last_recorded_sample_index,
        )
        self._sample_prediction_records.append(sample_prediction_record)

    def snapshot_sample_prediction_records(self) -> tuple[SamplePredictionRecord, ...]:
        return tuple(self._sample_prediction_records)
