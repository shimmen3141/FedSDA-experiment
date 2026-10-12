"""生成済みのモデルの値を、サーバの登録まで保留し、ラウンドの待ちの後に送信可能にする。"""

from dataclasses import dataclass

from torch import Tensor, float32, isfinite, strided

from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    PENDING_MODEL_REGISTRATION_POLICIES,
)


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
    """一所有者の送信保留（採用の順）と、それぞれの、送信可能までの残りのラウンド境界の回数。

    1つだけ保持する方針では、足す操作が、前の送信保留を置き換える。全部保持する方針では、末尾へ足す。
    """

    def __init__(self, *, pending_model_registration_policy: str) -> None:
        if type(pending_model_registration_policy) is not str:
            raise TypeError("pending_model_registration_policyはbuiltin strが必要です。")
        if pending_model_registration_policy not in PENDING_MODEL_REGISTRATION_POLICIES:
            raise ValueError(
                "pending_model_registration_policyは、宣言した値（"
                + "・".join(PENDING_MODEL_REGISTRATION_POLICIES)
                + "）が必要です。"
            )
        self._pending_model_registration_policy = pending_model_registration_policy
        # （送信保留、残りの待ち）を、採用の順に持つ。
        self._pending_model_uploads: list[tuple[PendingModelUpload, int]] = []

    @property
    def pending_model_registration_policy(self) -> str:
        return self._pending_model_registration_policy

    @property
    def remaining_upload_delay_round_count(self) -> int:
        """最も古い送信保留の、残りの待ち（送信保留が無ければ0）。"""
        if not self._pending_model_uploads:
            return 0
        return self._pending_model_uploads[0][1]

    def _append_or_replace_pending_model_upload(
        self, *, pending_model_upload: PendingModelUpload, remaining_upload_delay_round_count: int
    ) -> None:
        if (
            self._pending_model_registration_policy
            == KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
        ):
            self._pending_model_uploads = [
                (pending_model_upload, remaining_upload_delay_round_count)
            ]
            return
        if any(
            held_pending_model_upload.model_id == pending_model_upload.model_id
            for held_pending_model_upload, _ in self._pending_model_uploads
        ):
            raise ValueError("model_idは、既存の送信保留の対応IDです。")
        self._pending_model_uploads.append(
            (pending_model_upload, remaining_upload_delay_round_count)
        )

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
        self._append_or_replace_pending_model_upload(
            pending_model_upload=pending_model_upload,
            remaining_upload_delay_round_count=upload_delay_round_count,
        )

    def get_pending_model_upload(self) -> PendingModelUpload | None:
        """最も古い送信保留（待ちが済んでいるとは限らない）。無ければNone。"""
        if not self._pending_model_uploads:
            return None
        return self._pending_model_uploads[0][0]

    def get_all_pending_model_uploads(self) -> tuple[PendingModelUpload, ...]:
        """送信保留の全部を、採用の順で返す。"""
        return tuple(
            pending_model_upload for pending_model_upload, _ in self._pending_model_uploads
        )

    def get_ready_pending_model_upload(self) -> PendingModelUpload | None:
        """待ちの済んだ送信保留のうち、最も古いもの。無ければNone。"""
        for pending_model_upload, remaining_upload_delay_round_count in self._pending_model_uploads:
            if remaining_upload_delay_round_count == 0:
                return pending_model_upload
        return None

    def queue_ready_model_upload(
        self, *, model_id: int, parameter_snapshot: dict[str, Tensor]
    ) -> None:
        """検査後、即時送信可能なsnapshotを、方針に応じて、置き換えるか、末尾へ足す。"""
        pending_model_upload = PendingModelUpload(
            model_id=model_id, parameter_snapshot=parameter_snapshot
        )
        self._append_or_replace_pending_model_upload(
            pending_model_upload=pending_model_upload, remaining_upload_delay_round_count=0
        )

    def has_ready_model_upload(self) -> bool:
        return self.get_ready_pending_model_upload() is not None

    def count_ready_model_uploads(self) -> int:
        """待ちの済んだ送信保留の数（サーバが、1回の同期で登録する件数）。"""
        return sum(
            remaining_upload_delay_round_count == 0
            for _, remaining_upload_delay_round_count in self._pending_model_uploads
        )

    def advance_upload_readiness_at_round_boundary(self) -> None:
        self._pending_model_uploads = [
            (pending_model_upload, max(remaining_upload_delay_round_count - 1, 0))
            for pending_model_upload, remaining_upload_delay_round_count in self._pending_model_uploads
        ]

    def remove_pending_model_upload(self, *, model_id: int) -> None:
        """指定したIDの送信保留を、1つ外す。無ければ、何も変えずに拒否する。"""
        if type(model_id) is not int:
            raise TypeError("model_idはbool・派生型以外のbuiltin intが必要です。")
        remaining_pending_model_uploads = [
            (pending_model_upload, remaining_upload_delay_round_count)
            for pending_model_upload, remaining_upload_delay_round_count in self._pending_model_uploads
            if pending_model_upload.model_id != model_id
        ]
        if len(remaining_pending_model_uploads) == len(self._pending_model_uploads):
            raise ValueError("model_idに対応する送信保留がありません。")
        self._pending_model_uploads = remaining_pending_model_uploads

    def clear_pending_model_upload(self) -> None:
        """送信保留の全部を外す。"""
        self._pending_model_uploads = []
