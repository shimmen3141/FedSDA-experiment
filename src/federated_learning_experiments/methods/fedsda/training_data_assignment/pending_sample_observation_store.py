"""学習帰属が未確定の標本（位置・標本・診断用概念ID）と、候補検証へ渡した標本の概念IDを保持する。"""

from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)


def _validate_concept_ids(*, sample_concept_ids: tuple[int | None, ...]) -> None:
    if type(sample_concept_ids) is not tuple:
        raise TypeError("sample_concept_ids must be exact tuple")
    for observed_concept_id in sample_concept_ids:
        if observed_concept_id is not None and type(observed_concept_id) is not int:
            raise TypeError("concept IDs must be builtin int or None")


class PendingSampleObservationStore:
    """保留位置のownerと同じ並びの標本を所有する。どの位置を残すかは呼出側が保留位置のownerから渡す。"""

    def __init__(self) -> None:
        self._pending_sample_observations: list[IndexedObservedTrainingSample] = []
        self._validation_assignment_sample_concept_ids: tuple[int | None, ...] | None = None

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

    @property
    def validation_assignment_sample_concept_ids(self) -> tuple[int | None, ...] | None:
        """候補検証へ渡した標本の概念ID。候補検証へ渡した標本がなければNone。"""
        return self._validation_assignment_sample_concept_ids

    def hold_validation_assignment_sample_concept_ids(
        self, *, sample_concept_ids: tuple[int | None, ...]
    ) -> None:
        """開始した候補検証へ渡した標本の概念IDを、確定まで保持する。保持中の上書きは拒否する。"""
        _validate_concept_ids(sample_concept_ids=sample_concept_ids)
        if self._validation_assignment_sample_concept_ids is not None:
            raise ValueError("validation assignment sample concept IDs are already held")
        self._validation_assignment_sample_concept_ids = sample_concept_ids

    def release_validation_assignment_sample_concept_ids(self) -> tuple[int | None, ...]:
        """候補検証の確定で、保持していた概念IDを外して返す。"""
        if self._validation_assignment_sample_concept_ids is None:
            raise ValueError("validation assignment sample concept IDs are not held")
        sample_concept_ids = self._validation_assignment_sample_concept_ids
        self._validation_assignment_sample_concept_ids = None
        return sample_concept_ids
