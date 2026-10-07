"""観測位置と借用標本、診断用概念IDを結び付ける。"""

from dataclasses import dataclass

from .model_training_sample_records import ObservedTrainingSample


@dataclass(frozen=True, kw_only=True)
class IndexedObservedTrainingSample:
    """位置対応は供給側が保証し、入力の検査は利用側で行う。"""

    sample_index: int
    training_sample: ObservedTrainingSample
    observed_concept_id: int | None
