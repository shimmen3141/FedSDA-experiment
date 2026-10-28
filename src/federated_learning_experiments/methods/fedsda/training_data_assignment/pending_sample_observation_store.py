"""学習帰属が未確定の標本（位置・標本・診断用概念ID）を保持する。"""

from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)


class PendingSampleObservationStore:
    """保留位置のownerと同じ並びの標本を所有する。どの位置を残すかは呼出側が保留位置のownerから渡す。"""

    def __init__(self) -> None:
        self._pending_sample_observations: list[IndexedObservedTrainingSample] = []

    def append_pending_sample_observation(
        self, *, indexed_observation: IndexedObservedTrainingSample
    ) -> None:
        """観測した標本を末尾へ足す。位置は、保持中の最後の位置より後であること。"""
        if type(indexed_observation) is not IndexedObservedTrainingSample:
            raise TypeError("indexed_observation must be exact IndexedObservedTrainingSample")
        if type(indexed_observation.sample_index) is not int:
            raise TypeError("sample_index must be builtin int")
        if (
            self._pending_sample_observations
            and indexed_observation.sample_index
            <= self._pending_sample_observations[-1].sample_index
        ):
            raise ValueError("sample_index must follow the last pending sample index")
        self._pending_sample_observations.append(indexed_observation)

    def snapshot_pending_sample_observations(self) -> tuple[IndexedObservedTrainingSample, ...]:
        """保持中の標本を古い順のtupleで返す（標本そのものは借用）。"""
        return tuple(self._pending_sample_observations)

    def retain_latest_pending_sample_observations(
        self, *, retained_sample_indices: tuple[int, ...]
    ) -> None:
        """新しい側の標本だけを残す。残す位置は、保持中の並びの末尾と一致していること。"""
        if type(retained_sample_indices) is not tuple:
            raise TypeError("retained_sample_indices must be exact tuple")
        retained_sample_count = len(retained_sample_indices)
        pending_sample_count = len(self._pending_sample_observations)
        if (
            retained_sample_count > pending_sample_count
            or tuple(
                indexed_observation.sample_index
                for indexed_observation in self._pending_sample_observations[
                    pending_sample_count - retained_sample_count :
                ]
            )
            != retained_sample_indices
        ):
            raise ValueError("retained sample indices must be the latest pending sample indices")
        del self._pending_sample_observations[: pending_sample_count - retained_sample_count]
