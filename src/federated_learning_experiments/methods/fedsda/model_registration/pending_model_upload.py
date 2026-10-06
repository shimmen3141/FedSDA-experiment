"""一つの生成済みモデル値を保留し、ラウンド待機後に送信可能にする。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, strided


def _validate_pending_model_upload_inputs(
    *, model_id: int, parameter_snapshot: dict[str, Tensor]
) -> None:
    if type(model_id) is not int:
        raise TypeError("model_idはbool・派生型以外のbuiltin intが必要です。")
    if type(parameter_snapshot) is not dict:
        raise TypeError("parameter_snapshotはexact plain dictが必要です。")
    if not parameter_snapshot:
        raise ValueError("parameter_snapshotは非空が必要です。")
    for parameter_name, parameter_values in parameter_snapshot.items():
        if type(parameter_name) is not str:
            raise TypeError("parameter_snapshotのkeyはexact strが必要です。")
        if not parameter_name:
            raise ValueError("parameter_snapshotのkeyは非空が必要です。")
        if not isinstance(parameter_values, Tensor):
            raise TypeError(f"parameter_snapshot.{parameter_name}はTensorが必要です。")
        if (
            parameter_values.device.type != "cpu"
            or parameter_values.dtype != float32
            or parameter_values.layout != strided
            or parameter_values.is_nested
        ):
            raise ValueError(
                f"parameter_snapshot.{parameter_name}はCPU float32 stridedが必要です。"
            )
        if parameter_values.requires_grad or parameter_values.grad_fn is not None:
            raise ValueError(f"parameter_snapshot.{parameter_name}は勾配記録のない値が必要です。")
        if not isfinite(parameter_values).all().item():
            raise ValueError(f"parameter_snapshot.{parameter_name}は有限値が必要です。")


@dataclass(frozen=True, kw_only=True)
class PendingModelUpload:
    """対応IDと生成済みsnapshotを借用する。Tensor/辞書は深い不変値ではない。"""

    model_id: int
    parameter_snapshot: dict[str, Tensor]

    def __post_init__(self) -> None:
        _validate_pending_model_upload_inputs(
            model_id=self.model_id, parameter_snapshot=self.parameter_snapshot
        )


class PendingModelUploadState:
    """一所有者の一保留枠と、送信可能までの残ラウンド境界回数。"""

    def __init__(self) -> None:
        self._pending_model_upload: PendingModelUpload | None = None
        self._remaining_upload_delay_round_count = 0

    @property
    def remaining_upload_delay_round_count(self) -> int:
        return self._remaining_upload_delay_round_count

    def queue_model_upload(
        self,
        *,
        model_id: int,
        parameter_snapshot: dict[str, Tensor],
        upload_delay_round_count: int,
    ) -> None:
        if type(upload_delay_round_count) is not int:
            raise TypeError("upload_delay_round_countはbool・派生型以外のbuiltin intが必要です。")
        if upload_delay_round_count < 1:
            raise ValueError("upload_delay_round_countは1以上が必要です。")
        pending_model_upload = PendingModelUpload(
            model_id=model_id, parameter_snapshot=parameter_snapshot
        )
        self._pending_model_upload = pending_model_upload
        self._remaining_upload_delay_round_count = upload_delay_round_count

    def get_pending_model_upload(self) -> PendingModelUpload | None:
        return self._pending_model_upload

    def has_ready_model_upload(self) -> bool:
        return (
            self._pending_model_upload is not None and self._remaining_upload_delay_round_count == 0
        )

    def advance_upload_readiness_at_round_boundary(self) -> None:
        if self._pending_model_upload is not None and self._remaining_upload_delay_round_count > 0:
            self._remaining_upload_delay_round_count -= 1

    def clear_pending_model_upload(self) -> None:
        self._pending_model_upload = None
        self._remaining_upload_delay_round_count = 0
