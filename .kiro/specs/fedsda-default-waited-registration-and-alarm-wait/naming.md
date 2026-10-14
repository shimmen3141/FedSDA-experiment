# 命名と役割

このspecは、sourceに新しい公開名を足さない（既存の定数の値を、既定として使うだけ）。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `build_final_configuration_fedsda_run_settings`（`runtime/fedsda_final_configuration_run_settings.py`） | 登録は全部登録、警報は即時を既定にする | 登録は現行モデルとして待った後に登録、警報は最小件数まで待つを既定にする |
| `REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME`・`WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME` | 設定で選べる扱い | FedSDAの既定（定数の注記を直す） |
| `REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME`・`RAISE_ALARM_IMMEDIATELY_POLICY_NAME` | FedSDAの既定 | 設定で選べる扱い（警報の即時は、ADWINの2構成の値） |
