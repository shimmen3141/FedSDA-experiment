"""サーバが持つグローバルモデル（正式IDのモデル）のパラメータ・損失統計・次の正式ID・登録の来歴。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)


@dataclass(frozen=True, kw_only=True)
class GlobalModelRegistrationRecord:
    """正式IDが付いた事実。初期モデルは、ラウンドとclientを持たない（None）。"""

    model_id: int
    registered_round_index: int | None
    registering_client_id: int | None


def _validate_nonnegative_identifier(*, identifier: int, identifier_name: str) -> None:
    if type(identifier) is not int:
        raise TypeError(f"{identifier_name} must be builtin int")
    if identifier < 0:
        raise ValueError(f"{identifier_name} must be nonnegative")


def _copy_validated_parameter_snapshot(
    *, parameter_snapshot: dict[str, Tensor]
) -> dict[str, Tensor]:
    if type(parameter_snapshot) is not dict:
        raise TypeError("parameter_snapshot must be builtin dict")
    if not parameter_snapshot:
        raise ValueError("parameter_snapshot must not be empty")
    for parameter_name, parameter_values in parameter_snapshot.items():
        if type(parameter_name) is not str:
            raise TypeError("parameter_snapshot keys must be builtin str")
        if type(parameter_values) is not Tensor:
            raise TypeError("parameter_snapshot values must be exact torch.Tensor")
    return {
        parameter_name: parameter_values.detach().clone()
        for parameter_name, parameter_values in parameter_snapshot.items()
    }


def _validate_loss_statistics(*, loss_statistics: ModelAndClassLossStatistics) -> None:
    if type(loss_statistics) is not ModelAndClassLossStatistics:
        raise TypeError("loss_statistics must be exact ModelAndClassLossStatistics")


class GlobalModelRepository:
    """グローバルモデルの状態を所有する。パラメータは、モデルIDごとの完全なパラメータ（共有部を含む）で持つ。

    集約や統合の判断はしない。受け取るパラメータと返すパラメータは、内部と結合しない写しにする。
    """

    def __init__(
        self,
        *,
        initial_model_id: int,
        initial_parameter_snapshot: dict[str, Tensor],
        initial_loss_statistics: ModelAndClassLossStatistics,
    ) -> None:
        _validate_nonnegative_identifier(
            identifier=initial_model_id, identifier_name="initial_model_id"
        )
        copied_parameter_snapshot = _copy_validated_parameter_snapshot(
            parameter_snapshot=initial_parameter_snapshot
        )
        _validate_loss_statistics(loss_statistics=initial_loss_statistics)
        self._parameter_snapshots_by_model_id: dict[int, dict[str, Tensor]] = {
            initial_model_id: copied_parameter_snapshot
        }
        self._loss_statistics_by_model_id: dict[int, ModelAndClassLossStatistics] = {
            initial_model_id: initial_loss_statistics
        }
        self._next_global_model_id = initial_model_id + 1
        self._registration_records_by_model_id: dict[int, GlobalModelRegistrationRecord] = {
            initial_model_id: GlobalModelRegistrationRecord(
                model_id=initial_model_id, registered_round_index=None, registering_client_id=None
            )
        }

    @property
    def next_global_model_id(self) -> int:
        """次に採番する正式ID。"""
        return self._next_global_model_id

    @property
    def global_model_ids(self) -> tuple[int, ...]:
        """パラメータを持つモデルのID。最初に設定した順。"""
        return tuple(self._parameter_snapshots_by_model_id)

    def allocate_global_model_id(self) -> int:
        """次の正式IDを返して、1つ進める。"""
        allocated_model_id = self._next_global_model_id
        self._next_global_model_id += 1
        return allocated_model_id

    def record_model_registration(
        self, *, model_id: int, registered_round_index: int, registering_client_id: int
    ) -> None:
        """登録の来歴を記録する。同じモデルIDへの記録は、置き換える。"""
        _validate_nonnegative_identifier(identifier=model_id, identifier_name="model_id")
        _validate_nonnegative_identifier(
            identifier=registered_round_index, identifier_name="registered_round_index"
        )
        _validate_nonnegative_identifier(
            identifier=registering_client_id, identifier_name="registering_client_id"
        )
        self._registration_records_by_model_id[model_id] = GlobalModelRegistrationRecord(
            model_id=model_id,
            registered_round_index=registered_round_index,
            registering_client_id=registering_client_id,
        )

    def snapshot_model_registration_records(self) -> tuple[GlobalModelRegistrationRecord, ...]:
        """登録の来歴を、モデルIDの昇順で返す。"""
        return tuple(
            self._registration_records_by_model_id[model_id]
            for model_id in sorted(self._registration_records_by_model_id)
        )

    def set_global_model_parameters(
        self, *, model_id: int, parameter_snapshot: dict[str, Tensor]
    ) -> None:
        """モデルIDのパラメータを、渡された値の写しで置く（既にあれば置き換える。順は最初に置いた位置のまま）。"""
        _validate_nonnegative_identifier(identifier=model_id, identifier_name="model_id")
        self._parameter_snapshots_by_model_id[model_id] = _copy_validated_parameter_snapshot(
            parameter_snapshot=parameter_snapshot
        )

    def get_global_model_parameters(self, *, model_id: int) -> dict[str, Tensor]:
        """モデルIDのパラメータの写しを返す。持っていないIDはKeyError。"""
        _validate_nonnegative_identifier(identifier=model_id, identifier_name="model_id")
        return {
            parameter_name: parameter_values.detach().clone()
            for parameter_name, parameter_values in self._parameter_snapshots_by_model_id[
                model_id
            ].items()
        }

    def set_global_model_loss_statistics(
        self, *, model_id: int, loss_statistics: ModelAndClassLossStatistics
    ) -> None:
        _validate_nonnegative_identifier(identifier=model_id, identifier_name="model_id")
        _validate_loss_statistics(loss_statistics=loss_statistics)
        self._loss_statistics_by_model_id[model_id] = loss_statistics

    def snapshot_global_model_loss_statistics(
        self,
    ) -> tuple[tuple[int, ModelAndClassLossStatistics], ...]:
        """損失統計を持つモデルの（ID、統計）を、最初に置いた順で返す。"""
        return tuple(self._loss_statistics_by_model_id.items())

    def get_global_model_loss_statistics(
        self, *, model_id: int
    ) -> ModelAndClassLossStatistics | None:
        """モデルIDの損失統計。持っていなければNone。"""
        _validate_nonnegative_identifier(identifier=model_id, identifier_name="model_id")
        return self._loss_statistics_by_model_id.get(model_id)
