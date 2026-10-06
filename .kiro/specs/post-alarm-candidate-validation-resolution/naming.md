# 命名 revision 2

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/post_alarm_candidate_validation_resolution.py|警報後の候補検証を確定し、評価結果に応じた状態更新を選んで適用する。評価（methods/fedsdaのpost_alarm_candidate_loss_evaluation.py）や損失収集ではない|
|POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES|結果種別の正式な4値のtuple|
|candidate_adopted_as_new_model|結果種別。候補を新モデルとして採用し学習帰属を新しい一時IDへ切り替えた。旧action create|
|held_reference_model_reused|結果種別。保有済みの別モデルへ学習帰属を切り替え、保留標本を吸収した。旧action reuse|
|current_model_maintained|結果種別。現行モデルが履歴の損失範囲内で再利用可能と評価され、学習帰属を変えずに保留標本を吸収した。旧action maintain|
|candidate_rejected|結果種別。再利用可能な参照がなく候補も採用されず、学習帰属を変えずに保留標本を現行モデルへ吸収した。旧action create_rejected。current_model_maintainedとは評価理由が異なる|
|PostAlarmCandidateValidationResolution|frozen dataclass。適用した結果種別・帰属先・変更記録。PostAlarmCandidateLossEvaluation（状態を変えない評価結果）と区別|
|resolution_outcome|str、上の4値のいずれか|
|assigned_model_id|int。保留標本の帰属先のモデルID。確定後の現在の学習帰属IDと同じ値|
|training_model_assignment_change|既存変更recordまたはNone。現在の学習帰属IDが変わった場合だけ|
|apply_post_alarm_candidate_validation_resolution|評価結果に応じて採用/吸収/現在ID切替えを適用し結果recordを返す。apply_*は決定済み変更の適用（方針の命名規則）。自身の状態なし|
|post_alarm_candidate_loss_evaluation|既存評価結果record。candidate_acceptedとreusable_reference_model_idだけを読む|
|pending_assignment_training_samples|採用APIと同じ名前・役割。判定中に帰属を保留していた標本。採用では新モデルの学習標本、それ以外では帰属先モデルへ吸収|
|pending_assignment_sample_concept_ids|exact tuple[int|None,...]、保留標本列と同じ長さ。各標本の真の概念ID（診断専用）。吸収APIのassigned_sample_concept_idsへ渡す。採用では使わない|
|temporary_model_id_allocator / adopted_candidate_classifier / candidate_concept_specific_parameter_optimizer_state / initial_statistics_input_features / initial_statistics_observed_class_labels / upload_delay_round_count / candidate_trained_sample_count / candidate_parameter_update_step_count / pending_model_upload_state|採用APIと同じ名前・役割。採用のときだけ使う|
|held_model_training_state_registry / loss_statistics_store / training_sample_store / model_training_and_assignment_counts_store / current_training_model_assignment|採用API・吸収APIと同じ具体型・役割|
|_validate_resolution_inputs|評価結果・現在ID owner・保留標本列と概念ID列を検証。変更なし|
|previous_model_id|確定前の現在の学習帰属ID（読取り値）|
|reusable_reference_model_id|評価結果から読んだ再利用先のモデルID|
|assignment_change|採用または現在ID切替えが返した変更record（局所変数）|
|observed_concept_id|概念ID列の一要素|

## testで使用する名前

|名前|役割|
|---|---|
|build_resolution_oracle|上流の採用oracleへ、指定した旧分岐になる損失列と履歴平均を設定し、同じ入力から新の評価結果を作る|
|legacy_resolution_case|旧分岐のtest軸（create / reuse / maintain / create_rejected）|
|LEGACY_ACTION_BY_RESOLUTION_OUTCOME|新結果種別と旧actionの対応表（test側の明示的な対応）|
|resolution_arguments|新関数のkeyword引数dict|
|evaluation_arguments|移植済み評価関数へ渡すtest-only接続の引数|
|legacy_client / legacy_session / legacy_drift_type|実旧client/実旧session/実旧確定処理の戻り値|
|assert_resolution_matches_legacy|結果record・全owner状態を実旧と対照|
|pending_sample_count|保留標本数のtest軸|
|invalid_case / IntSubclass|拒否入力|
|expected_call_order / actual_call_order / operation_name / original_operation|順序観測|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

共同学習接続と不変確認の名前は上流adoption/absorption specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1・Task2で追加した名前（revision 2）

productionの追加名はない。

|名前|役割|
|---|---|
|LEGACY_DRIFT_TYPE_BY_RESOLUTION_OUTCOME|新結果種別と旧確定処理の戻り値（2/1/0）の対応表（test側）|
|LEGACY_RESOLUTION_CASES|旧分岐のtest軸の値の列（create / reuse / maintain / create_rejected）|
|resolution_module|monkeypatchで採用・吸収の呼出しを観測するための新moduleの別名|
|finalize_forward_validation_in_legacy_client|実旧_finalize_forward_validationを実行し、旧判定acceptedが新の評価結果と同じであることを確かめて戻り値を返すtest helper。手順を再構成しない|
|resolution / assignment_change / previous_model_id|新関数が返した結果record/その変更記録/確定前の現在ID|
|candidate_losses / reference_losses_by_model_id / reference_historical_mean_losses_by_model_id|旧sessionと新評価関数の両方へ与える同じ損失列と履歴平均（既存評価APIの引数名）|
|shared_optimizer_owners / legacy_candidate_model / registry / candidate / candidate_feature_extractor / candidate_optimizer / previous_held_model_training_states / previous_counts / current_counts|上流testと同じ役割の局所名と、採用以外で候補・一覧・計数が変わらないことの比較用|
|valid_resolution_arguments / evaluation / evaluation_subclass / field_name|不正値へ差し替える前の引数一式/評価結果/その派生型/複製するfield名|
|replace_middle_pending_sample|test内closure。保留標本列の中央だけを差し替える|
|record_then_call / record_then_assign / original_assignment|順序観測wrapperと元の現在ID切替え|
|run_joint_update_in_both_implementations とその局所名|上流の採用・吸収testと同じ役割|
|post_alarm_candidate_validation_resolution_cpu_smoke.py / post_alarm_candidate_validation_resolution_red_evidence.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke/実装をstubと誤実装へ一時差し替えてtestの失敗を確かめる検証script。Git管理外|
|test_post_alarm_candidate_validation_resolution_exact_dependency_contract|依存境界testの注入契約|
