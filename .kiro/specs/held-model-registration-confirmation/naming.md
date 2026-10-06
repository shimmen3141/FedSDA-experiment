# 命名 revision 1

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/held_model_registration_confirmation.py|保有済みモデルへの正式ID通知を組み立てる。初期登録/送信ではない|
|confirm_held_model_registration|7ownerへ確認を適用しTrainingModelAssignmentChangeまたはNoneを返す。自身の状態なし|
|registered_global_model_id|exact int非負、サーバが返した正式ID。採番/対応表ではない|
|held_model_training_state_registry|既存具体型、classifierとoptimizer参照の一覧を付替え|
|loss_statistics_store|既存具体型、元統計で先を上書き|
|training_sample_store / evaluation_sample_store|既存具体型、元標本列で先を上書き。抽出しない|
|model_training_and_assignment_counts_store|既存具体型、元計数を先へ加算移管|
|current_training_model_assignment|既存具体型、最後に現在帰属ID変更|
|pending_model_upload_state|既存具体型、最後に一保留枠と待機を解除|
|original_model_id|現在負ID、登録確認の元。pending IDから推測しない|
|assignment_change|既存変更record、現在ID変更後/保留解除後に返す|
|_validate_confirmation_inputs|IDと7ownerを全検証、変更なし|
|state_owner / expected_owner_type / owner_name|型検査ループの具体owner/期待具体型/例外で示す名前|

## testで使用する名前

|名前|役割|
|---|---|
|build_registration_confirmation_oracle|7新ownerと実旧confirm用状態を構築|
|confirmation_arguments|新関数のkeyword引数dict、owner一式と正式ID|
|legacy_client|固定旧confirmを呼ぶfixture|
|assert_confirmation_matches_legacy|各ownerのID/順序/上書き/加算/保留を対照|
|snapshot_confirmation_owners|拒否時不変を比較する各owner構造snapshot|
|initial_model_ids / auxiliary_source_present / receiving_model_id|一覧順/元補助項目有無/通知先のtest軸|
|classifier / optimizer_owner / registry|既存NN/個別optimizer owner/registry fixtureの意味を再利用|
|loss_statistics_store / training_sample_store / evaluation_sample_store / counts_store / training_assignment / pending_upload_state|既存owner fixtureの意味を再利用|
|previous_states / previous_bindings / previous_optimizer_state / parameter_snapshots / pending_model_upload / previous_snapshot|取得済み参照/状態の保持比較|
|training_samples / evaluation_samples / sample_index / model_id / observed_class_id|標本列/位置/識別子/観測ラベル|
|expected_call_order / actual_call_order / operation_name / original_operation / arguments / keyword_arguments|順序観測用wrapper、callを委譲し結果を記録|
|invalid_global_model_id / invalid_owner_name / invalid_owner / IntSubclass / OwnerSubclass|拒否入力・派生型|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数、永続状態なし|

共同学習接続のbatch/optimizer/settings/Random状態の名前は上流joint-training/registry-ID/count specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。
