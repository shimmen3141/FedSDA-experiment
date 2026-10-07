# 命名と役割 revision3

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

## 追加 revision3（tasks実装で必要になった名前）

revision2の表は変更していない。以下は主担当Claude Codeがtasks 1〜3のtest/実装を具体化して必要になった名前で、コード追加前に登録する。

| 名前 | 種別 | 役割 |
| --- | --- | --- |
| _validate_bounded_mean_loss | source private（methods） | 区間平均または履歴基準の1値を、bool以外のbuiltin int/float・有限・0以上1以下で検査する。引数は既存同義のparameter_name/specified_value。状態更新なし |
| evaluated_model_loss | source/test local | (ID, 区間平均)の1組。既存の初期値選択selector内の同名localと同義で再利用 |
| TupleSubclass / FloatSubclass | test class | exact tuple/builtin floatだけを受理することを確かめる拒否入力用のsubclass。既存のIntSubclass/DictSubclass（同義再利用）と同じ命名形 |
| valid_assessment_arguments | test local | 純粋判定の正常なkeyword引数dict。既存valid_*_argumentsと同じ形 |
| expected_exception | test parameter | 拒否で期待する例外型。既存同義の再利用 |
| reuse_evaluation_arguments | test local | runtimeの評価関数へ渡すkeyword引数dict（特徴/ラベル/保有状態/統計/閾値） |
| resolve_alarm_interval_in_legacy_client | test helper | 実旧_resolve_driftを、評価より後の副作用（吸収/イベント記録/検出器reset/帰属切替）だけ差し替えて実行し、評価済み候補列・適合列・選択IDを返す。旧の最終状態の対照には使わない |
| start_candidate_validation_session | 上記helperのbool引数 | Trueなら実旧の初期値選択と実旧のsession開始まで実行する（task3の接続用）。Falseなら開始を差し替えて評価だけ観測する |
| legacy_evaluated_candidates / legacy_valid_candidates / legacy_selected_reuse_model_id | test local | 実旧のevaluated_candidates/valid_candidatesと、旧の選択関数が返したID（適合なしはNone）。旧のlocal名に対応 |
| legacy_initialization_parameters | test local | 実旧_select_initialization_paramsが返した初期parameter |
| legacy_interval_mean_losses_by_model_id | test local | 実旧モデルのper_sample_errorから直接求めた区間平均。履歴基準と閾値を条件どおりに組むための測定値で、判定の期待値には使わない |
| HISTORY_AND_FIT_CASES / history_and_fit_case / history_and_fit_roles | test定数/parameter/local | 保有順のモデルごとの役割列（履歴未登録・0件・1件・零平均・履歴と同値で適合・履歴より低く適合・履歴を超えて不適合）と、そのcase名 |
| threshold_boundary_case | test parameter | 許容損失増加量を実測の差の直前/等値/直後に置く3条件の名 |
| loss_evaluation_calls | test local | 新runtimeが既存の損失評価を呼んだ記録（patchのwraps）。基準を使えないモデルもforwardされること、保有順を確かめる |
| mismatched_feature_count_classifier | test local | 特徴数が異なる保有モデル。後続モデルでの形状拒否と状態保持の確認用 |
| snapshot_reuse_evaluation_state / assert_reuse_evaluation_state_unchanged | test helper | 保有順・owner同一性・全parameter/grad・optimizer state・履歴統計・入力Tensor・3乱数状態の記録と不変の照合 |
| test_alarm_interval_reuse_evaluation_threshold_boundary_matches_legacy | test | 実NNの差に対する閾値の直前/等値/直後を実旧と対照 |
| test_alarm_interval_reuse_evaluation_tie_follows_held_order | test | 同じ値の実モデルによる同率で、現行やIDの大小でなく保有順で選ぶことを実旧と対照 |

上流testから同義で再利用するhelper: build_session_start_oracle、begin_candidate_validation_session_in_legacy_client、assert_started_session_matches_legacy、set_overall_loss_statistics_in_both_implementations（引数statistics_by_model_id）、snapshot_parameter_values_and_gradients、assert_parameter_values_and_gradients_unchanged、assert_nested_state_equal、assert_collected_losses_match_legacy。held_model_ids/optimizer_variant/session_start_arguments/started_session/legacy_session/legacy_client/shared_optimizer_ownersも同義。
