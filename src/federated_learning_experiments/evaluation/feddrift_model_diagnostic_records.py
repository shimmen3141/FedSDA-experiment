"""FedDriftの通常同期と終了時モデル学習量の不変記録。"""

from dataclasses import dataclass


def _validate_integer(value: int, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError("記録の整数はbuiltin intが必要です。")
    if minimum is not None and value < minimum:
        raise ValueError("記録の整数が下限を満たしていません。")


@dataclass(frozen=True, kw_only=True)
class FeddriftSynchronizedTimestepModelCounts:
    """通常同期の最後の配布の後の実モデル数。"""

    timestep_index: int
    global_model_count: int
    client_held_model_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_integer(self.timestep_index, 0)
        _validate_integer(self.global_model_count, 1)
        if type(self.client_held_model_counts) is not tuple:
            raise TypeError("client保有数はtupleが必要です。")
        if not self.client_held_model_counts:
            raise ValueError("client保有数は空にできません。")
        for count in self.client_held_model_counts:
            _validate_integer(count, 1)


@dataclass(frozen=True, kw_only=True)
class FeddriftClientHeldModelTrainingAmount:
    """実行終了時にclientが保有するモデル1個の学習量。"""

    client_id: int
    model_id: int
    stored_training_sample_count: int
    trained_sample_count: int
    parameter_update_step_count: int

    def __post_init__(self) -> None:
        _validate_integer(self.client_id, 0)
        _validate_integer(self.model_id)
        for count in (
            self.stored_training_sample_count,
            self.trained_sample_count,
            self.parameter_update_step_count,
        ):
            _validate_integer(count, 0)
