"""評価用観測標本とモデル別一覧の借用参照を宣言する。"""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ObservedEvaluationSample:
    """評価用の特徴と観測済み正解を借用する。payloadの検査は利用側が行う。"""

    input_features: Tensor
    observed_class_labels: Tensor


@dataclass(frozen=True, kw_only=True)
class ModelEvaluationSampleCollection:
    """一モデルの評価標本列を構造分離した不変snapshot。"""

    model_id: int
    evaluation_samples: tuple[ObservedEvaluationSample, ...]
