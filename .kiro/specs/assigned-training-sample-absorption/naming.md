# 命名 revision 1

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
