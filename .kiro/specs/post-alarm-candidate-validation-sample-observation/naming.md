# 命名 revision 1

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
