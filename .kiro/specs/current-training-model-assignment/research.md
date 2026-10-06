# 根拠と範囲

- 旧clients/base.py:34で初期ID0、189/196で現在IDの単一予測と履歴。単一帰属は複数モデルの混合予測重みとは別の概念。
- 旧clients/fedsda.py::_set_local_current_modelはID更新後、異なる場合だけ_on_local_model_change(old,new)を通知する。
- 旧clients/feddrift.pyでも選択済み・新規のIDを現在IDに設定する。このためFedSDA固有のrouting配下ではなくlearning/trainingへ置く。
- 旧BaseClient.apply_server_mappingは対応表を一段適用。実変更時だけ現在のprocessed_samplesとserver_merge理由を記録する。新ownerは位置/理由を持たず変更記録を返し、上位に記録させる。
- confirm_model_registrationは負の現在IDを正式IDに付け替える。統計/モデル/標本の上書きと計数加算は別ownerの既存契約。正式登録全体の呼出し順序とpending消去は次specへ残す。
- LEGACY011の同ID計数移管拒否と、単一IDの同値設定no-opは別の契約。本specで旧不具合を修正したとは扱わない。
