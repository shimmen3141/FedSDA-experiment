# 命名: 分類器parameter snapshot

revision: 2

## production
|名前|役割・型/入出力・単位・状態|
|---|---|
|classifier_parameter_snapshot.py|一分類器の全parameter値を独立コピーするファイル。候補選択/平均/送信状態を扱わない|
|snapshot_classifier_parameters|keyword classifier→dict[str,Tensor]。全parameterの現在値をnative順で取得。元モデル更新なし。送信以外の初期化にも使うのでuploadと命名しない|
|_validate_classifier_parameter_snapshot_inputs|classifier→None。モデル型と全parameter環境/有限性を検査。snapshotの適用/内容検査とは区別|
|classifier|取得元ResidualAdapterClassifier。準備/学習は完了済み|
|parameter_name / parameter_values|検査中のnamed_parametersの名前/Parameter。値の単位はparameterごとで追加の換算なし|
|classifier_parameter_values|state_dict由来の借用辞書。返却snapshotと違いモデルstorageを参照|
|parameter_snapshot|独立した返却辞書。optimizer/grad/統計/IDを含まない|

## testとguard（task開始前の追加名）
|名前|役割|
|---|---|
|test_classifier_parameter_snapshot.py|独立コピー契約・旧対照・test-only initializer接続|
|ResidualAdapterClassifierSubclass|exact型拒否用のテスト専用subclass|
|test_snapshot_matches_legacy_values_and_native_key_order|実旧snapshotとのprefix対応/全値/順序を確認|
|test_snapshot_is_independent_in_both_directions_and_across_calls|モデル/出力相互変更・再取得の独立性|
|test_snapshot_preserves_model_state_and_caller_environment|既存grad/参照/flags/乱数/環境とforward未実行|
|test_snapshot_rejects_invalid_classifier / test_snapshot_rejects_invalid_parameter_values|型・parameter契約の拒否|
|test_training_snapshot_and_candidate_initialization_match_legacy|学習後snapshot→既存initializer→native復元を実旧対照|
|legacy_parameter_snapshot / native_parameter_name / legacy_parameter_name|旧get_params出力/新native名/テストでのみ対応する旧名|
|previous_parameter_state / previous_training_flags / previous_feature_extractor|呼出前の値/grad、各module flags、共有参照|
|first_parameter_snapshot / second_parameter_snapshot / current_parameter_snapshot|異なる時点の独立結果|
|snapshot_parameter_values_and_gradients / assert_parameter_values_and_gradients_unchanged|既存test helperを再利用、値/grad保持|
|class_count / hidden_layer_widths / optimizer_variant / update_shared_features|既存条件名を再利用。クラス数/層幅/optimizer選択/共有更新有無|
|training_batches / shared_parameter_optimizer / legacy_client / local_training_settings|既存共同更新対照helperと同役割|
|initialized_parameter_snapshot / restored_classifier / initialization_settings|initializer出力/復元先/既存初期化設定|
|previous_optimizer_state / previous_random_states / previous_default_dtype / previous_default_device / previous_grad_mode|検証前状態、更新しない|
|previous_parameter_gradients|不正parameterに既存のgrad参照。optimizer stateとは区別する|
|invalid_parameter_kind / invalid_classifier / first_parameter / forward_calls / forward_hook|不正条件/型/検査対象/実行回数/観測hook|
|expected_parameter_values / parameter_values / parameter_name / prefix_pairs / native_prefix / legacy_prefix|期待値/比較値/名前/明示prefix対応。変換はtest-only|
|model_id / input_features / selected_parameter_snapshot / first_classifier / second_classifier|test/smoke内のID/入力/選択出力/取得元/復元先|
|source_module_path / imported_module_name / source_text|既存AST guard引数を再利用|

他の永続状態・設定・公開型は追加しない。testのpytest/既存helper引数は既存役割を維持する。
