"""clientが採用したモデルの、サーバへの登録の扱い。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)

# サーバの登録の扱い。送信保留の保持の仕方と、正式IDの確認で付け替えるモデルを、1つの値で決める。
# 全部を保持し、待ちの済んだそれぞれを、現行かどうかに依らず、自身の一時IDから正式IDへ付け替える（FedSDAの既定。
# 基準版2・3）。
REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME = "register_each_pending_model_by_its_own_id"
# 最後に採用した1つだけを保持し、確認のとき、現行モデルのIDを付け替える（固定旧FedSDAとの照合と、FedDrift）。
KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME = (
    "keep_latest_pending_model_and_reassign_current_model_id"
)
# 全部を保持し、待ちは、現行である送信保留だけ、同期の後に進める。待ちが済み、登録の同期の時点でも現行である
# 送信保留だけを、自身の一時IDから正式IDへ付け替える（現行モデルとして待った後に登録する。通信量を優先する
# ときに選ぶ。基準版4。2026-10-14の比較で、精度がわずかに下がったので、既定から戻した）。
REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME = (
    "register_pending_model_after_waiting_as_current_model"
)
PENDING_MODEL_REGISTRATION_POLICIES = (
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
)


@dataclass(frozen=True, kw_only=True)
class PendingModelRegistrationSettings:
    """送信保留の保持の仕方と、正式IDの確認で付け替えるモデルを決める方針だけを宣言する。

    送信までの待ちのラウンド数は、ここには持たない（clientの設定の、置き場所のない値の束にある）。
    """

    pending_model_registration_policy: str = field(
        metadata={"allowed_parameter_values": PENDING_MODEL_REGISTRATION_POLICIES}
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
