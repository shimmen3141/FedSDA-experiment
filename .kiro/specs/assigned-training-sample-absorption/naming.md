# 命名 revision 3

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/assigned_training_sample_absorption.py|帰属先が確定した標本を保有モデルへ吸収する組立。採用時の標本追加（統計を更新しない）や、帰属を保留するbuffer（pending_training_assignment_buffer）ではない|
|absorb_assigned_training_samples_into_held_model|標本追加・割当概念計数・損失評価・統計更新を行いNoneを返す。自身の状態なし。append_model_training_samples（標本追加だけ）と区別|
|model_id|exact int。吸収先の保有済みモデル。一時IDでも正式IDでもよい|
|assigned_training_samples|exact tuple[ObservedTrainingSample,...]。帰属先がmodel_idに確定した標本。1recordは1標本。pending_assignment_training_samples（採用APIの、採用で新モデルへ移す保留標本）と区別|
|assigned_sample_concept_ids|exact tuple[int|None,...]、標本列と同じ長さ。各標本の真の概念ID（診断専用）。既存計数APIのobserved_concept_idへ1件ずつ渡す。Noneは概念不明|
|held_model_training_state_registry|既存具体型。吸収先の分類器を読むだけ。変更しない|
|training_sample_store|既存具体型。標本を1件ずつ末尾へ追加|
|model_training_and_assignment_counts_store|既存具体型。割当概念計数だけを更新。学習計数は変更しない|
|loss_statistics_store|既存具体型。全体と観測クラスの統計を1件ずつ更新|
|_validate_absorption_inputs|model_id・4owner・標本列・概念ID列を検証。変更なし|
|classifier|吸収先の分類器（読取り）|
|training_sample / observed_concept_id|標本列/概念ID列の一要素|
|per_sample_bounded_losses|既存損失評価の戻り値（1要素）|
|observed_losses / observed_class_ids|全標本の損失float列/観測クラスint列。状態変更前に揃える|
|observed_loss / observed_class_id|既存統計APIの引数名と同じ。1標本の損失/観測クラス|
|state_owner / expected_owner_type / owner_name|型検査ループ（登録・採用と同じ）|

## testで使用する名前

|名前|役割|
|---|---|
|build_absorption_oracle|上流の採用oracleから新4ownerと実旧clientを取り出し、吸収する標本列と概念ID列、対応する旧標本tuple列を作る|
|absorption_arguments|新関数のkeyword引数dict|
|legacy_client / legacy_training_samples|実旧clientと、旧形式の標本tuple列|
|assert_absorption_matches_legacy|標本列・割当概念計数・統計全fieldを実旧と対照（上流のassert_training_samples_match_legacy等を使う）|
|snapshot_absorption_state / assert_absorption_state_unchanged|拒否時不変の観測と確認|
|absorbed_sample_count / concept_id_case / target_model_case|標本数/概念IDの与え方/吸収先のtest軸|
|invalid_case / IntSubclass|拒否入力|
|expected_call_order / actual_call_order / operation_name / original_operation|順序観測|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

共同学習接続・確定処理接続の名前は上流registration/adoption specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1・Task2で追加した名前（revision 2）

productionの追加名はない。

|名前|役割|
|---|---|
|OWNER_NAMES|新関数が受け取る4ownerの引数名のtuple。owner別拒否のtest軸|
|absorption_module|monkeypatchで損失評価の呼出しを観測するための新moduleの別名|
|adoption_arguments / shared_optimizer_owners|上流の採用oracleが返す引数一式/共有optimizer管理器列。現在ID ownerや初期統計の特徴を読むために使う|
|assert_models_optimizers_and_random_state_unchanged|一覧record identity・全parameter/grad・全optimizer・torch/random/NumPy乱数が観測時から変わらないことを確認。成功時の不変（1.5）と拒否時不変の両方で使う|
|valid_absorption_arguments|不正値へ差し替える前の引数一式|
|replace_training_sample|test内closure。標本列の指定位置だけを、特徴またはラベルを差し替えたrecordへ置き換える|
|invalid_owner / owner_type / owner_name|owner別拒否の不正値の種類/正しいownerの型/引数名|
|invalid_input_features|旧の途中失敗を再現する、特徴数が合わない特徴|
|previous_legacy_sample_count / previous_legacy_statistics_count / previous_legacy_concept_counts|旧の部分更新を観測するための吸収前の件数|
|original_evaluation / record_then_evaluate / record_then_update|呼出しを記録してから元の損失評価/owner APIへ委譲する順序観測|
|legacy_resolution_case / legacy_session / legacy_drift_type / expected_model_id|実旧確定処理の非採用分岐（棄却/現行維持/別モデル再利用）の種類、実旧session、戻り値、旧が決めた帰属先|
|previous_held_model_ids / previous_next_temporary_model_id / parameter_snapshots|非採用分岐で保有一覧・採番次値・parameterが変わらないことの比較用|
|current_training_model_assignment|既存owner。再利用分岐で旧が決めた帰属先へ現在IDを合わせるtest-only接続に使う|
|run_joint_update_in_both_implementations / legacy_training_batches / training_bindings / model_training_sample_collections / participating_training_batches / expected_joint_loss / actual_joint_loss / local_training_settings / active / random_states / numpy_state|上流の採用testと同じ役割|
|current_counts / previous_model_id / previous_training_samples / expected_training_samples / other_model_id / previous_loss_statistics / current_model_id|吸収前後の観測値|
|assigned_training_sample_absorption_cpu_smoke.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke。Git管理外|
|test_assigned_training_sample_absorption_exact_dependency_contract|依存境界testの注入契約|

## RED証拠の補強で追加した名前（revision 3）

productionの追加名はない。

|名前|役割|
|---|---|
|target_model_caseの値without_training_samples|吸収先のモデルが標本列を持たない条件。空列の吸収が標本列を作らないこと（要求1.4）を実旧と値で照合する|
|current_training_samples / assigned_training_sample|吸収後の標本列snapshot/吸収した標本の一要素（比較用）|
|assigned_training_sample_absorption_red_evidence.py|共有venvのrefactoring-testsへ置く、承認対象の実装を一時的にstubと誤実装へ差し替えてtestの失敗を確かめ、元へ戻してhashを照合する検証script。Git管理外|
