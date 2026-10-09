"""クロス評価（モデルを、別のモデルの標本でclientに評価させること）の診断の記録を、評価の順に持つ。"""

from dataclasses import dataclass, replace
from math import isfinite


def _validate_nonnegative_count(*, count: int, count_name: str) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < 0:
        raise ValueError(f"{count_name} must be nonnegative")


def _validate_finite_sum(*, loss_sum: float, sum_name: str) -> None:
    if type(loss_sum) is not float:
        raise TypeError(f"{sum_name} must be builtin float")
    if not isfinite(loss_sum):
        raise ValueError(f"{sum_name} must be finite")


def _validate_correctness_counts(
    *, correctness_counts: tuple[int, ...], evaluated_sample_count: int, counts_name: str
) -> None:
    """（候補だけ正解、対象だけ正解、両方正解、両方不正解）の4つの数。和は、評価した標本数。"""
    if type(correctness_counts) is not tuple:
        raise TypeError(f"{counts_name} must be builtin tuple")
    if len(correctness_counts) != 4:
        raise ValueError(f"{counts_name} must hold four counts")
    for correctness_count in correctness_counts:
        _validate_nonnegative_count(count=correctness_count, count_name=counts_name)
    if sum(correctness_counts) != evaluated_sample_count:
        raise ValueError(f"{counts_name} must sum to the evaluated sample count")


@dataclass(frozen=True, kw_only=True)
class ClientCrossEvaluationRecord:
    """clientの評価1回の診断の記録。候補（candidate）は評価する側、対象（target）は標本を持つ側のモデル。

    正誤の数は（候補だけ正解、対象だけ正解、両方正解、両方不正解）。正誤を比べなかった評価ではNone。
    クラス別は（クラス、件数、候補だけ正解、対象だけ正解、両方正解、両方不正解）の列で、クラスの昇順。
    """

    round_index: int
    client_id: int
    candidate_model_id: int
    target_model_id: int
    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float
    correctness_counts: tuple[int, int, int, int] | None
    class_correctness_counts: tuple[tuple[int, int, int, int, int, int], ...]

    def __post_init__(self) -> None:
        for identifier_name in (
            "round_index",
            "client_id",
            "candidate_model_id",
            "target_model_id",
            "evaluated_sample_count",
        ):
            _validate_nonnegative_count(
                count=getattr(self, identifier_name), count_name=identifier_name
            )
        _validate_finite_sum(loss_sum=self.bounded_loss_sum, sum_name="bounded_loss_sum")
        _validate_finite_sum(
            loss_sum=self.squared_bounded_loss_sum, sum_name="squared_bounded_loss_sum"
        )
        if type(self.class_correctness_counts) is not tuple:
            raise TypeError("class_correctness_counts must be builtin tuple")
        if self.correctness_counts is None:
            if self.class_correctness_counts:
                raise ValueError(
                    "class_correctness_counts must be empty without correctness_counts"
                )
            return
        _validate_correctness_counts(
            correctness_counts=self.correctness_counts,
            evaluated_sample_count=self.evaluated_sample_count,
            counts_name="correctness_counts",
        )
        previous_class_id = -1
        class_sample_count_sum = 0
        for class_counts in self.class_correctness_counts:
            if type(class_counts) is not tuple:
                raise TypeError("class_correctness_counts elements must be builtin tuple")
            if len(class_counts) != 6:
                raise ValueError("class_correctness_counts elements must hold six values")
            class_id, class_sample_count, *class_correctness_counts = class_counts
            _validate_nonnegative_count(count=class_id, count_name="class_id")
            if class_id <= previous_class_id:
                raise ValueError("class_correctness_counts must be in ascending class order")
            previous_class_id = class_id
            _validate_nonnegative_count(count=class_sample_count, count_name="class sample count")
            _validate_correctness_counts(
                correctness_counts=tuple(class_correctness_counts),
                evaluated_sample_count=class_sample_count,
                counts_name="class correctness counts",
            )
            class_sample_count_sum += class_sample_count
        if self.class_correctness_counts and class_sample_count_sum != self.evaluated_sample_count:
            raise ValueError(
                "class sample counts must sum to the evaluated sample count when present"
            )


class CrossEvaluationRecordStore:
    """clientの評価の記録を、足された順に所有する。集計や判定はしない。"""

    def __init__(self) -> None:
        self._client_cross_evaluation_records: list[ClientCrossEvaluationRecord] = []

    def append_client_cross_evaluation_record(
        self, *, client_cross_evaluation_record: ClientCrossEvaluationRecord
    ) -> None:
        if type(client_cross_evaluation_record) is not ClientCrossEvaluationRecord:
            raise TypeError(
                "client_cross_evaluation_record must be exact ClientCrossEvaluationRecord"
            )
        # 入力が手動で書き換えられていても全fieldを再検査し、保存用の写しを更新前に確定する。
        self._client_cross_evaluation_records.append(replace(client_cross_evaluation_record))

    def snapshot_client_cross_evaluation_records(
        self,
    ) -> tuple[ClientCrossEvaluationRecord, ...]:
        """評価の順の、その後の追加の影響を受けない写し。"""
        return tuple(self._client_cross_evaluation_records)
