"""clientとサーバの間の通信量（モデル転送数、軽量メッセージ数、パラメータの値の数、バイト数）を数える。"""

from dataclasses import dataclass

from torch import Tensor

_UPLOAD = "upload"
_DOWNLOAD = "download"


@dataclass(frozen=True, kw_only=True)
class CommunicationVolumeSnapshot:
    """上り（client→サーバ）と下り（サーバ→client）の通信量。

    モデル数は論理的なモデル転送の回数、メッセージ数はパラメータを伴わない通知の回数。
    値の数とバイト数は、実際に転送したパラメータの量。
    """

    uploaded_model_count: int
    downloaded_model_count: int
    uploaded_message_count: int
    downloaded_message_count: int
    uploaded_parameter_value_count: int
    downloaded_parameter_value_count: int
    uploaded_byte_count: int
    downloaded_byte_count: int


def _validate_transfer_direction(*, transfer_direction: str) -> None:
    if type(transfer_direction) is not str:
        raise TypeError("transfer_direction must be builtin str")
    if transfer_direction not in (_UPLOAD, _DOWNLOAD):
        raise ValueError("transfer_direction must be 'upload' or 'download'")


def _validate_count(*, count: int, count_name: str) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < 0:
        raise ValueError(f"{count_name} must be nonnegative")


class CommunicationVolumeRecordStore:
    """通信量の計数だけを所有し、何を送るかの判断をしない。"""

    def __init__(self) -> None:
        self._model_counts = {_UPLOAD: 0, _DOWNLOAD: 0}
        self._message_counts = {_UPLOAD: 0, _DOWNLOAD: 0}
        self._parameter_value_counts = {_UPLOAD: 0, _DOWNLOAD: 0}
        self._byte_counts = {_UPLOAD: 0, _DOWNLOAD: 0}

    def record_model_transfers(self, *, transfer_direction: str, model_count: int) -> None:
        """論理的なモデル転送の回数を足す（パラメータの量は足さない）。"""
        _validate_transfer_direction(transfer_direction=transfer_direction)
        _validate_count(count=model_count, count_name="model_count")
        self._model_counts[transfer_direction] += model_count

    def record_messages(self, *, transfer_direction: str, message_count: int) -> None:
        """パラメータを伴わない軽量メッセージの回数を足す。"""
        _validate_transfer_direction(transfer_direction=transfer_direction)
        _validate_count(count=message_count, count_name="message_count")
        self._message_counts[transfer_direction] += message_count

    def record_parameter_transfer(
        self,
        *,
        transfer_direction: str,
        parameter_snapshot: dict[str, Tensor],
        transfer_count: int = 1,
    ) -> None:
        """パラメータの値の数とバイト数（値の数×1値のバイト数）に、転送回数を掛けて足す。"""
        _validate_transfer_direction(transfer_direction=transfer_direction)
        _validate_count(count=transfer_count, count_name="transfer_count")
        if type(parameter_snapshot) is not dict:
            raise TypeError("parameter_snapshot must be builtin dict")
        for parameter_name, parameter_values in parameter_snapshot.items():
            if type(parameter_name) is not str:
                raise TypeError("parameter_snapshot keys must be builtin str")
            if type(parameter_values) is not Tensor:
                raise TypeError("parameter_snapshot values must be exact torch.Tensor")
        parameter_value_count = sum(
            parameter_values.numel() for parameter_values in parameter_snapshot.values()
        )
        byte_count = sum(
            parameter_values.numel() * parameter_values.element_size()
            for parameter_values in parameter_snapshot.values()
        )
        self._parameter_value_counts[transfer_direction] += parameter_value_count * transfer_count
        self._byte_counts[transfer_direction] += byte_count * transfer_count

    def get_state_snapshot(self) -> CommunicationVolumeSnapshot:
        return CommunicationVolumeSnapshot(
            uploaded_model_count=self._model_counts[_UPLOAD],
            downloaded_model_count=self._model_counts[_DOWNLOAD],
            uploaded_message_count=self._message_counts[_UPLOAD],
            downloaded_message_count=self._message_counts[_DOWNLOAD],
            uploaded_parameter_value_count=self._parameter_value_counts[_UPLOAD],
            downloaded_parameter_value_count=self._parameter_value_counts[_DOWNLOAD],
            uploaded_byte_count=self._byte_counts[_UPLOAD],
            downloaded_byte_count=self._byte_counts[_DOWNLOAD],
        )
