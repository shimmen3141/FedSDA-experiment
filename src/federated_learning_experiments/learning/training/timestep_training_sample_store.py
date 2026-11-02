"""役割の窓へ渡すため、直近10時刻の観測標本を保管する。"""

from .model_training_sample_records import ObservedTrainingSample


class TimestepTrainingSampleStore:
    """時刻順に保存し、窓に含む時刻を昇順に連結する。"""

    def __init__(self) -> None:
        self._samples: dict[int, tuple[ObservedTrainingSample, ...]] = {}
        self._last_timestep = -1

    @property
    def timestep_indices(self) -> tuple[int, ...]:
        return tuple(self._samples)

    def append_timestep(
        self, *, timestep_index: int, training_samples: tuple[ObservedTrainingSample, ...]
    ) -> None:
        if type(timestep_index) is not int or timestep_index != self._last_timestep + 1:
            raise ValueError("標本の時刻は0から連続している必要があります。")
        if type(training_samples) is not tuple:
            raise TypeError("標本列はtupleが必要です。")
        if any(type(sample) is not ObservedTrainingSample for sample in training_samples):
            raise TypeError("標本はexact ObservedTrainingSampleが必要です。")
        self._samples[timestep_index] = training_samples
        self._last_timestep = timestep_index
        if len(self._samples) > 10:
            del self._samples[min(self._samples)]

    def samples_for_window(
        self, *, timestep_indices: tuple[int, ...]
    ) -> tuple[ObservedTrainingSample, ...]:
        if any(t not in self._samples for t in timestep_indices) or len(
            set(timestep_indices)
        ) != len(timestep_indices):
            raise ValueError("窓の時刻は保管済みで重複しない必要があります。")
        return tuple(s for t in sorted(timestep_indices) for s in self._samples[t])
