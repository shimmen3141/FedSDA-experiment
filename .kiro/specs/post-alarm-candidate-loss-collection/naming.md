# 命名と役割

revision: 1

警報後の有界損失系列の収集だけを担う。candidate/referenceはモデルの実体ではなく外部で算出した損失の対応先。validationは学習に使わない将来観測の評価区間、sample_indexはclientのglobal位置。

| 名前 | 役割・単位・状態 |
|---|---|
| post_alarm_candidate_loss_collection.py | 警報後の候補/参照損失収集と同所有の状態型 |
| PostAlarmCandidateLossCollection | 設定・提案位置・固定参照IDを受け、損失系列を所有 |
| PostAlarmCandidateLossCollectionState | 変更不能な独立snapshot |
| observe_losses_after_label_observation | 観測位置と候補/全参照のlossを検査して一件追加、None返却 |
| validation_sample_count / required_validation_sample_count | 実際の検証件数 / 設定の規定件数、sample/client |
| ready_for_acceptance_evaluation | 規定件数到達、採否自体を意味しないbool |
| get_state_snapshot | 途中/完了のcopy、状態不変 |
| candidate_model_training_and_acceptance_settings / _candidate_model_training_and_acceptance_settings | 規定件数の唯一の所有者、既存設定型 |
| proposal_sample_index / _proposal_sample_index | 警報当日位置、将来観測には含めない |
| sample_index / last_validation_sample_index / _last_validation_sample_index | 今回検証位置 / 最後位置、未観測None |
| reference_model_ids / _reference_model_ids | 開始時の固定参照tuple、反復順を維持 |
| candidate_loss / candidate_losses / _candidate_losses | 一回入力float化前loss / snapshot tuple / 内部list |
| reference_losses_by_model_id / _reference_losses_by_model_id | 一回入力dict・snapshot tuple[(id,tuple)] / 内部dict[id,list]、文脈と型注釈で区別 |
| validated_reference_losses_by_model_id | 更新前に検査・float化を完了した局所dict |
| _validate_bounded_observed_loss | builtin number/有限[0,1]を検査しfloat返却、状態なし |
| specified_value / parameter_name / model_id | 検査入力値 / エラー項目名 / 参照ID |
| collection / reference_collection / legacy_session / legacy_client / legacy_events | testの新収集/別実体/旧oracle/旧stub/観測・finalize capture |
| state_snapshot / state_before_call / inputs_before_call / result / observation | copy・拒否前状態・入力copy・評価結果・public分類出力 |
| sample_count / observation_index / invalid_value / field_name / operation / target_count | test件数・反復・拒否値・署名/field・操作・旧規定件数 |
| global_python_random_state / global_numpy_random_state / global_torch_random_state | 共有乱数のtest診断 |
| global_default_dtype / global_default_device | 共有torchのtest診断、一時device scopeで復元 |
| make_loss_collection / make_legacy_loss_session / make_legacy_observation_client | 明示条件のtest helper |
| reference_historical_mean_losses_by_model_id / available_reference_model_ids / current_training_model_id / maximum_reference_mean_loss_increase / minimum_candidate_mean_loss_improvement | 既存採否APIの上位入力、production収集が所有しない |

## 検証

test_post_alarm_candidate_loss_collection.pyと接頭辞test_candidate_loss_collection_:
matches_legacy_session / invalid_initial_conditions / invalid_observation_is_atomic / readiness_matches_legacy_client / snapshots_and_instances_are_independent / preserves_global_numeric_state / connects_to_acceptance_evaluation / public_arguments_are_keyword_only。
旧callback・属性名target_count/proposal_position/reference_models/candidate_training_examples等とx/y/idx/model/candidate/reference_lossesはtest-only oracle参照。既存AST helper変数・既存数値APIの命名は各上流spec承認を参照する。

