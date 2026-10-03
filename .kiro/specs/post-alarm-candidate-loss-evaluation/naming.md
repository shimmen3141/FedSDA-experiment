# 命名と役割

revision: 1

## 方針

post_alarmは警報後の未学習標本で評価した損失、referenceは警報時点の比較モデル、candidateは新規モデル候補。reusableは履歴平均超過の条件に適合した既存モデルで、ここで実際の再利用操作はしない。
旧refitはこの判定内で再学習しないため新名に含めない。forward_persistentは正式選択肢名に引き継がず、既存固定条件のcurrent_model_first_reuse_then_two_segment_candidate_validationを使用する。

## module・型・関数

| 名前 | 役割・入出力・状態 |
|---|---|
| post_alarm_candidate_loss_evaluation.py | 警報後損失から適合参照選択と候補判定だけを行う数値module |
| PostAlarmCandidateLossEvaluation | immutable診断結果、model/損失列stateなし |
| select_available_reference_within_historical_loss_tolerance | 損失/履歴/利用可能/現行/許容値→適合IDまたはNone、更新なし |
| evaluate_candidate_using_post_alarm_losses | 既存固定条件と収集済みloss→結果、更新なし |
| _validate_nonnegative_finite_number | specified_value/parameter_nameの型・非負・有限性を検査しfloat返却 |
| _validate_model_id | model_id/parameter_nameのexact intを検査 |
| _validate_bounded_loss | specified_value/parameter_nameの有限[0,1]検査 |
| _validate_reference_loss_inputs | 参照/履歴/利用可能/現行/許容値の契約検査、返却なし |
| _validate_loss_sequence | losses/parameter_nameのtuple・最低2件・各値検査 |
| _mean_bounded_loss | losses→CPUfloat32平均Pythonfloat |
| _select_validated_reusable_reference | 検査済み入力から適合IDを選ぶ同module内部計算 |

## 引数・結果・局所名

| 名前 | 意味・型・単位 |
|---|---|
| candidate_model_training_and_acceptance_settings | 既存採否固定条件、既存型が唯一所有 |
| candidate_losses / reference_losses_by_model_id | 観測順candidate tuple / modelID→tuple、値無次元[0,1] |
| reference_historical_mean_losses_by_model_id | 警報時点の履歴平均snapshot、入力のみ |
| available_reference_model_ids / current_training_model_id | 現時点に存在するID列 / 現行学習割当ID、負ID許可 |
| maximum_reference_mean_loss_increase | 履歴平均を超える許容値、無次元、非負 |
| minimum_candidate_mean_loss_improvement | 両区間で必要な改善量、無次元、非負 |
| comparison_reference_model_id / reusable_reference_model_id | 比較に使うID / 適合した既存IDまたはNone |
| candidate_accepted / decision_reason / validation_sample_count | 採用bool / 適合又は区間診断理由 / 観測件数 |
| candidate_full_interval_mean_loss / reference_full_interval_mean_loss | 全体平均、無次元 |
| candidate_second_segment_mean_loss / reference_second_segment_mean_loss | split以降の後半平均、無次元 |
| reference_historical_mean_loss | 比較対象履歴平均またはNone |
| specified_value / parameter_name / model_id / losses | 検査値 / 理由付きエラーの項目名 / 反復ID / 検査列 |
| available_model_ids / fitting_references / mean_loss / historical_mean_loss | 検査済みID集合 / (平均,ID)列 / 算出平均 / 履歴値 |
| candidate_loss_tensor / reference_loss_tensor / segment_split_index | CPUfloat32表現 / 前半floor長 |
| first_segment_margin / second_segment_margin / first_segment_failed / second_segment_failed | float32差のPythonfloat / 閾値以下bool |
| candidate_segment / reference_segment | 判定対象の各非空半区間 |
| reference_losses / validation_sample_count / result | 反復loss / 件数 / immutable出力 |
| field_name / invalid_value / exception_info / constructor_arguments / evaluation_arguments | テストの拒否対象・実行keyword辞書 |

理由値はdesignの結果型が正本。採否とmargin理由は丸めの順序が異なるため独立に記録する。

## 検証名とテスト専用名

test_post_alarm_candidate_loss_evaluation.py、接頭辞test_post_alarm_candidate_:
reference_selection_matches_legacy / invalid_reference_inputs_are_rejected_without_mutation / evaluation_matches_legacy_finalization / rounding_boundary_preserves_decision_reason / invalid_evaluation_inputs_are_rejected_without_mutation / public_loss_connection_matches_legacy / results_inputs_and_random_states_are_independent / public_functions_require_explicit_keyword_arguments。
helper: make_acceptance_settings、capture_legacy_candidate_decision、assert_evaluation_matches_legacy_decision。
test専用LegacyDecisionCaptured（append後中断例外）、LegacyDecisionCapture（decision属性とappend）、legacy_client、legacy_session、legacy_decision、legacy_reference_model_id、legacy_reason_names、reference_selection_arguments、captured_decision、sample_index、sample_count、monkeypatch、legacy_default_dtype、global_python_random_state、global_numpy_random_state、global_torch_random_state、evaluation_before_call、inputs_before_call、model_outputs_by_model_id、prediction_probabilities_by_model_id、observed_losses_by_model_id、observed_class_labels、operation、class_count、observation_index、candidate_loss、reference_loss、threshold。
既存のAST checker名とpublic予測損失名を再利用する。
