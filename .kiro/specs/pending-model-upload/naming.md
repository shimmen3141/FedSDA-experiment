# 命名: 新規モデルの送信保留

revision: 1

## production
|名前|役割・型/単位・状態と類似名との違い|
|---|---|
|model_registration / __init__.py|手法固有のローカル登録/サーバへの登録準備境界。空packageでre-exportなし|
|pending_model_upload.py|一モデルの送信待機。作成/学習/実送信とは区別|
|PendingModelUpload|frozen record: model_idと借用parameter_snapshot。送信済み/統計固定snapshotではない|
|PendingModelUploadState|一所有者の一保留枠と残ラウンド数。NN/統計/ID採番を所有しない|
|__post_init__ / _validate_pending_model_upload_inputs|record入力検査→None、更新なし|
|model_id|対応するbuiltin int ID。統計は上位が同IDで解決、単位なし|
|parameter_snapshot|生成済みdict[str,Tensor]の全モデル値。再取得/二重コピーしない借用参照|
|upload_delay_round_count|queue時の正の整数、単位ラウンド境界回数。呼出時点で確定|
|_pending_model_upload|内部のPendingModelUploadまたはNone。queue/clearだけ交換|
|_remaining_upload_delay_round_count|内部の非負残回数。queue/advance/clearで更新|
|remaining_upload_delay_round_count|読取専用property。設定値と残回数を区別|
|queue_model_upload|keyword ID/snapshot/delay→None。全検証後に一枠を置換|
|get_pending_model_upload|同一借用recordまたはNone。取得では消費しない|
|has_ready_model_upload|bool。待機中のpayload有無と送信可否を区別|
|advance_upload_readiness_at_round_boundary|一回のラウンド境界通知→None。時間やround番号を自動取得しない|
|clear_pending_model_upload|一枠を解除→None、登録確認/ID変更とは区別|
|pending_model_upload|全検証済みrecordの一時値。構築後に内部へ保存|
|parameter_name / parameter_values|検査中のkey/Tensor。値は各parameterの単位、換算なし|

## tests / guard / smoke
|名前|役割|
|---|---|
|test_pending_model_upload.py|状態/拒否/実旧照合/上位統計接続|
|test_pending_upload_transition_sequence_matches_legacy|delay・IDごとの実旧ラウンド状態列|
|test_pending_upload_retains_borrowed_snapshot_and_nonconsuming_record|値/順序/参照・record不変/過去record保持|
|test_pending_upload_rejects_invalid_inputs_without_mutation|不正queue/直接record構築を空/待機/readyで拒否|
|test_pending_upload_connects_fixed_parameters_to_current_statistics|実旧register/生統計と新producer/store/pending接続|
|test_pending_model_upload_dependency_contract|exact許可/拒否注入|
|build_legacy_pending_upload_client|__new__で実旧具象ClassConditionalESRFedSDAClientの必要な状態だけ用意。constructor省略はtest-only|
|assert_pending_upload_state_matches_legacy|取得値/送信可否/counterと旧Base/FedSDA公開関数を比較|
|pending_upload_state / legacy_client / pending_model_upload / previous_pending_model_upload|新所有者/旧oracle/取得record/置換前record|
|initial_parameter_snapshot / replacement_parameter_snapshot / previous_parameter_snapshot|生成済み値/置換値/変更検査用独立コピー|
|model_id / upload_delay_round_count / round_boundary_index / invalid_input_kind / state_variant|ID/設定回数/0始まり通知位置/不正条件/空待機ready条件|
|invalid_inputs / parameter_name / parameter_values / previous_remaining_round_count|queue用keyword辞書/key/Tensor/拒否前残回数|
|class_count / monkeypatch / classifier / training_batches|既存test/helperと同じ役割|
|input_features / observed_class_labels / per_sample_bounded_losses / initial_loss_statistics|batch/ラベル/標本別損失/初期統計|
|loss_statistics_store / current_loss_statistics / legacy_loss_statistics / observed_loss / observed_class_id|既存store/現在値/旧生辞書/観測誤差/正解クラス|
|expected_parameter_values / previous_parameter_state / previous_random_states|比較する全値/parameterとgradの既存snapshot/RNG|
|first_parameter / source_module_path / source_text / expected_acceptance / dependency_boundary_violations|モデル変更対象/既存AST入力/期待許可/違反列|
|assert_initial_loss_statistics_match_legacy / build_joint_update_oracle_pair / build_attachment_classifier / snapshot_parameter_values_and_gradients / assert_parameter_values_and_gradients_unchanged / assert_nested_state_equal|既存test helperを同じ役割で借用|

pytest fixture/標準型/一時ループ添字は既存慣例を使用し、公開API/永続状態の追加はレビューへ戻す。
