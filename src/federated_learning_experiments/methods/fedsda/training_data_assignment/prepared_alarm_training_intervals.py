"""前区間の処理を済ませた警報区間の構造を記録する。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)


@dataclass(frozen=True, kw_only=True)
class PreparedAlarmTrainingIntervals:
    """列構造を固定し、観測標本とTensorは借用する。"""

    earlier_observations: tuple[IndexedObservedTrainingSample, ...]
    change_interval_observations: tuple[IndexedObservedTrainingSample, ...]
    change_interval_start_sample_index: int | None
    earlier_interval_absorbed_model_id: int | None
