# 命名と役割 revision2

## Source

| 名前 | 役割・型・単位・更新 |
| --- | --- |
| alarm_interval_model_reuse_assessment.py | methodsは平均から適合判定、runtimeはモデル評価の組立。同名でも配置が責務を示す |
| AlarmIntervalModelReuseAssessment | 候補初期値用の評価列と適合列を持つfrozen/kw_only record、学習状態なし |
| assess_alarm_interval_model_reuse | 平均と履歴基準から区間の再利用適合を判断、I/O/Tensorなし |
| evaluate_held_models_for_alarm_interval_reuse | 公開保有モデルを区間評価し履歴基準と純粋判定を組み立てる |
| baseline_supported_interval_mean_losses_by_model_id | 公開record field/純粋判定入力。tuple(ID,区間平均)の保有順列で、履歴基準を使えるモデルだけ。全モデルの評価結果ではない。既存selectorのevaluated_mean_losses_by_model_idへ明示対応 |
| reusable_mean_losses_by_model_id | 上の列から増加量条件を満たした順序付き部分列。将来検証の参照選択とは別 |
| reuse_baseline_mean_losses_by_model_id | 評価済みID→非零履歴平均dict。閾値ではなく比較基準 |
| maximum_alarm_interval_mean_loss_increase | 区間mean−履歴meanの許容上限、有限非負数。有界loss単位、標本数/距離ではない |
| selected_reuse_model_id | 適合列mean最小・同率先着のID、Noneなら適合なし。負ID許容 |
| alarm_interval_reuse_assessment | runtime/pure結果の局所値、将来検証結果と区別 |
| per_sample_bounded_losses / interval_mean_loss / reuse_baseline_mean_loss | runtime内だけの一時値。標本損失Tensor/torch平均のfloat/利用可能な履歴floatまたはNone。公開record fieldではない |
| _validate_alarm_interval_reuse_assessment_inputs | 純粋入力tuple/dict/閾値の全検査、状態更新なし |
| _validate_alarm_interval_reuse_evaluation_inputs | runtime owner/閾値/保有非空の検査、batch評価は既存APIへ委譲 |

input_features/observed_class_labels/held_model_training_state_registry/loss_statistics_store、model_id/mean_loss、held_model_training_state/held_model_training_states、model_loss_statistics/overall_loss_moments、parameter_name/specified_value/expected_model_ids/seen_model_ids/parameter_snapshot/current_training_model_id、既存公開APIは同義再利用。現行IDはtest/初期値selectorだけで、区間再利用選択APIへ渡さない。永続する状態変数は追加しない。

## Test / Evidence

| 名前 | 役割 |
| --- | --- |
| test_alarm_interval_model_reuse_assessment.py | 純粋判定/runtime/実旧対照/拒否/接続 |
| build_alarm_interval_reuse_oracle | 同じ実NN/統計/入力を旧新へ準備 |
| assert_alarm_interval_reuse_assessment_matches_legacy | 評価列/適合列/選択と全状態の実旧対照 |
| test_alarm_interval_reuse_assessment_selects_ordered_minimum | 純粋閾値/同率/候補なし/負ID |
| test_alarm_interval_reuse_assessment_is_immutable | record frozen/kw_only |
| test_alarm_interval_reuse_assessment_rejects_invalid_inputs | mean/baseline/ID/閾値/構造の拒否 |
| test_alarm_interval_reuse_evaluation_matches_legacy | 実モデル/履歴/filter/評価順の対照 |
| test_alarm_interval_reuse_evaluation_rejects_without_mutation | owner/batch/閾値等の拒否と全状態保持 |
| test_alarm_interval_reuse_initialization_and_session_start | 3方式の初期値選択→実開始/観測のtest-only接続 |
| test_alarm_interval_reuse_assessment_dependency_contract / test_alarm_interval_reuse_evaluation_dependency_contract | methods/runtimeのexact依存注入 |
| assessment_module / reuse_evaluation_module | testで純粋判定/runtimeを指す別名、旧moduleと混同しない |
| alarm_interval_reuse_cpu_smoke.py / alarm_interval_reuse_mutation_evidence.py | Git管理外のfresh CPU/実source変異・復元証拠 |

class_count/model_count/invalid_case/field_name/field_value/source_text/expected_acceptance/imported_module_name、旧新比較helpers/RNG snapshot/状態照合/実session開始helpersは上流承認済み同義で再利用する。具体taskの新しいfixture/local名は実装前に追加表をレビューする。
