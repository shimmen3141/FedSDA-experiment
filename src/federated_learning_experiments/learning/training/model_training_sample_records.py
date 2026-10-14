"""観測標本・モデル別population・抽出batchの参照を明示する。"""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ObservedTrainingSample:
    """一標本の特徴と観測ラベルを借用する。"""

    input_features: Tensor
    observed_class_labels: Tensor


@dataclass(frozen=True, kw_only=True)
class ModelTrainingSampleCollection:
    """モデルIDと標本位置の順序を持つpopulationを借用する。"""

    model_id: int
    training_samples: tuple[ObservedTrainingSample, ...]
    # 最新の区切りの位置（区切りより後の標本は、警報の後に同じモデルで続けると決まった後に足されたもの）。
    # 無ければNone。範囲の検査は、抜き出しが行う（記録は参照を束ねるだけ）。
    latest_alarm_boundary_position: int | None


@dataclass(frozen=True, kw_only=True)
class SampledModelTrainingBatch:
    """参加モデルIDと抽出順に連結したbatchを保持する。"""

    model_id: int
    input_features: Tensor
    observed_class_labels: Tensor
