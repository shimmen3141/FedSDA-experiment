"""clientが採用したモデルの、サーバへの登録の扱い。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)

# サーバの登録の扱い。送信保留の保持の仕方と、正式IDの確認で付け替えるモデルを、1つの値で決める。
# 全部を保持し、それぞれを、自身の一時IDから正式IDへ付け替える（FedSDAの既定）。
REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME = "register_each_pending_model_by_its_own_id"
# 最後に採用した1つだけを保持し、確認のとき、現行モデルのIDを付け替える（固定旧FedSDAとの照合と、FedDrift）。
KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME = (
    "keep_latest_pending_model_and_reassign_current_model_id"
)
PENDING_MODEL_REGISTRATION_POLICIES = (
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
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
