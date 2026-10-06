# 命名 revision 2

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/post_alarm_candidate_validation_sample_observation.py|警報後の候補検証で、標本1件の損失を評価して収集へ追加する組立。収集owner（post_alarm_candidate_loss_collection.py）、評価（post_alarm_candidate_loss_evaluation.py）、確定（post_alarm_candidate_validation_resolution.py）ではない|
|observe_post_alarm_candidate_validation_sample|候補と参照の損失を評価し収集へ追加、到達したかのboolを返す。observe_*は標本の観測（方針の命名規則）。自身の状態なし。収集のobserve_losses_after_label_observation（評価済みの損失を受け取る）と区別|
|sample_index|int。標本位置。収集APIの同名引数へそのまま渡す|
|input_features / observed_class_labels|Tensor。ラベル観測後の1標本（[1,F]と[1,1]）。既存損失評価の同名引数|
|candidate_classifier|ResidualAdapterClassifier。検証中の候補。採用前なのでadopted_candidate_classifier（採用が決まった後の採用APIの引数）と区別|
|reference_classifiers_by_model_id|exact dict[int, ResidualAdapterClassifier]。警報時点の値で固定した参照モデル。保有モデル本体（学習で更新される）ではない。収集のreference_losses_by_model_idと同じkey集合|
|post_alarm_candidate_loss_collection|既存具体型。同じ観測回を全系列へ追加|
|candidate_loss|float。候補の損失（収集APIの同名引数）|
|reference_losses_by_model_id|dict[int, float]。参照ごとの損失（収集APIの同名引数）|
|per_sample_bounded_losses|既存損失評価の戻り値（1要素）|
|_evaluate_single_sample_bounded_loss|1つの分類器で標本1件の損失を評価しfloatを返す内部関数。1件でなければValueError|
|classifier / model_id / reference_classifier|内部関数の引数/対応の反復変数|

## testで使用する名前

|名前|役割|
|---|---|
|build_observation_oracle|上流の採用oracleの実旧sessionの損失列を空にして実旧の参照snapshotを設定し、同じ値の新の候補・参照分類器・損失収集と検証標本列を作る|
|observation_arguments|新関数のkeyword引数dict（標本と位置を除く共通部分）|
|validation_samples|検証に使う(特徴, ラベル)の列。1件ずつ両実装へ渡す|
|legacy_client / legacy_session|実旧client/実旧ForwardValidationSession|
|assert_collected_losses_match_legacy|収集snapshotの候補損失列・参照損失列・ID順を実旧sessionと対照|
|required_validation_sample_count|規定件数のtest軸|
|invalid_case / IntSubclass|拒否入力|
|expected_call_order / actual_call_order|順序観測|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

評価・確定・共同学習への接続の名前は上流resolution/adoption/absorption specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1・Task2で追加した名前（revision 2）

productionの追加名はない。

|名前|役割|
|---|---|
|LAST_VALIDATION_SAMPLE_INDEX|test定数57。最後の検証標本の位置。上流の確定testが旧の切替位置として照合する値に合わせる|
|make_acceptance_settings|規定件数を指定して既存の採否設定を作るtest helper|
|prepare_validation_session_in_both_implementations|現在の保有モデルの値で、実旧の参照snapshot（実旧_snapshot_reference_models）と同じ値の新の参照分類器・損失収集・検証標本列を作るtest helper。学習後に参照を固定する12条件testからも呼ぶ|
|historical_mean_case|履歴平均の与え方のtest軸（none / current / other）。実旧の分岐（棄却または採用 / 現行維持 / 別モデル再利用）を変える|
|observe_and_resolve_in_both_implementations|全検証標本を両実装へ渡すtest helper。実旧は到達時に確定まで進み、新は到達を見て既存の評価関数と確定をtest-only接続する|
|observation_module|monkeypatchで損失評価の呼出しを観測するための新moduleの別名|
|collection / collection_state / previous_collection_state|損失収集owner/そのsnapshot/拒否前のsnapshot|
|classifiers / classifier_index / training_modes / parameter_snapshots / random_states / numpy_state|候補と参照の分類器列と、学習mode・parameter・乱数の不変確認用の観測値|
|observation_index / sample_offset / is_last_observation / legacy_result / ready_for_acceptance_evaluation|観測回の位置/標本の通し番号/最後の観測回か/実旧の戻り値/新の戻り値|
|first_sample_index / first_input_features / first_observed_class_labels|拒否testで先に正常観測する1件目|
|forward_call_count_before_rejection / count_then_evaluate / original_evaluation|1標本契約違反がforwardより前に拒否されることを数えるための記録と観測wrapper|
|record_then_evaluate / record_then_observe / original_observation|順序観測wrapperと元の収集API|
|legacy_reference_model / reference_classifier / reference_classifiers_by_model_id|実旧の参照モデル/同じ値の新の参照分類器/その対応|
|evaluation / resolution / resolution_arguments / legacy_drift_type / previous_model_id|上流の確定testと同じ役割|
|run_joint_update_in_both_implementations とその局所名|上流の採用・吸収・確定testと同じ役割|
|post_alarm_candidate_validation_sample_observation_cpu_smoke.py / post_alarm_candidate_validation_sample_observation_red_evidence.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke/実装をstubと誤実装へ一時差し替えてtestの失敗を確かめる検証script。Git管理外|
|test_post_alarm_candidate_validation_sample_observation_exact_dependency_contract|依存境界testの注入契約|
