# 命名と役割 revision3

## Source

| 名前 | 役割・型・単位・更新 |
| --- | --- |
| alarm_change_interval_resolution.py | runtime。警報の変化区間へ区間評価の結果を適用する組立 |
| ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES | 結果種別の正式値tuple |
| alarm_interval_held_model_reused | 結果種別。警報区間の評価で、現行と別の保有モデルを再利用し、学習帰属を切り替えた。旧action reuse。候補検証の確定処理のheld_reference_model_reused（検証後に参照モデルを再利用）と区別するため、警報区間の結果であることを接頭辞で示す |
| alarm_interval_current_model_maintained | 結果種別。警報区間の評価で現行モデルが選ばれ、維持した。旧action maintain。確定処理のcurrent_model_maintained（検証後の維持）とは別の値 |
| alarm_interval_candidate_validation_started | 結果種別。警報区間で適合なしのため候補検証sessionを開始した。旧action create_pending |
| AlarmChangeIntervalResolution | frozen/kw_onlyの結果record |
| resolution_outcome | 結果種別。既存の確定recordのfieldと同名同義 |
| alarm_interval_reuse_assessment | 区間評価の情報。前specのlocal名をfield名として使う |
| assigned_model_id | 区間の標本を吸収したモデルID。候補検証開始ではNone。既存の確定recordのfieldと同名で意味も同じ（標本の帰属先）。型だけ違い、既存はint、本recordはNoneを許す |
| training_model_assignment_change | 学習帰属の変更記録。変更なしはNone。既存と同名同義 |
| started_validation_session | 開始した候補検証session。開始以外はNone |
| resolve_alarm_change_interval | 公開関数。評価→分岐→適用→結果 |
| change_interval_training_samples | 変化区間の標本列（1標本ずつ、観測順）。候補検証開始時は保留標本になる |
| change_interval_sample_concept_ids | 上の標本列と同じ長さの概念ID列。Noneは概念なし |
| change_interval_input_features / change_interval_observed_class_labels | 標本列を連結した区間全体の特徴[N, F]とラベル[N, 1]。関数内の一時値 |
| _validate_alarm_change_interval_resolution_inputs | 分岐に依らない入力の事前検査。状態更新なし |
| input_feature_count | 先頭の標本の特徴数。既存同義 |
| selected_reuse_model_id | 評価情報の選択ID。前specと同名同義 |
| available_parameter_snapshots_by_model_id / initial_candidate_parameter_snapshot | 初期値選択へ渡す全保有モデルのsnapshotと、選ばれた初期値。上流と同名同義 |
| held_model_training_states | 保有順の保有状態のsnapshot（registryの公開取得の戻り値）。事前検査での保有なし・帰属の確認と、全モデルのsnapshot取得に使う関数内の一時値。上流と同名同義 |
| current_training_model_id | 事前検査と初期値選択へ渡す、処理開始時点の現在の学習帰属ID。上流の引数と同名同義 |
| training_model_assignment_change（local） | 吸収後の`assign_model_for_training`の戻り値。Noneでなければ再利用、Noneなら維持と判定し、そのままrecordの同名fieldへ入れる |
| started_validation_session（local） | session開始の戻り値。そのままrecordの同名fieldへ入れる |
| first_training_sample | 特徴数の基準にする先頭の標本。事前検査内の一時値 |
| parameter_name / specified_value | 事前検査で検査対象の入力名と値を受ける引数。上流と同名同義（使う場合） |

maximum_alarm_interval_mean_loss_increase、held_model_training_state_registry、loss_statistics_store、training_sample_store、model_training_and_assignment_counts_store、current_training_model_assignment、candidate_parameter_initialization_settings、architecture_reference_classifier、parameter_optimizer_settings、candidate_epoch_training_settings、candidate_model_training_and_acceptance_settings、proposal_sample_index、estimated_change_point_sample_index、detection_episode_id、detector_name、training_sample、observed_concept_id、held_model_training_state/held_model_training_states、model_idは上流の公開APIと同名同義で再利用する。永続する状態変数は追加しない。

## Test / Evidence

| 名前 | 役割 |
| --- | --- |
| test_alarm_change_interval_resolution.py | 実旧対照、record、拒否、呼出し順、後続学習 |
| resolution_module | testで対象runtime moduleを指す別名（既存の確定testの同名別名と同じ形） |
| build_alarm_change_interval_resolution_oracle | 同じ実NN・統計・標本・計数・区間を持つ新owner群と実旧clientを準備する。戻り値はresolution_arguments、shared_optimizer_owners、legacy_client、legacy_interval_mean_losses_by_model_id |
| resolution_arguments | 対象関数のkeyword引数dict |
| resolve_alarm_change_interval_in_legacy_client | 実旧_resolve_driftを、吸収・帰属切替・初期値選択・session開始を差し替えずに実行する。イベント記録は記録用に、検出器resetと推定区間長だけ差し替える。戻り値はlegacy_drift_type、legacy_adaptation_events、legacy_epoch_training_calls |
| legacy_drift_type | 実旧_resolve_driftの戻り値（再利用1、他0） |
| legacy_adaptation_events | 実旧がイベント記録へ渡した引数dictの一覧。actionを結果種別と対応させる |
| record_legacy_adaptation_event | 上の一覧へ追加するだけの差し替え関数（選択や判定を行わない） |
| legacy_epoch_training_calls | 実旧の候補学習の呼出し記録。上流と同名同義 |
| legacy_interval_mean_losses_by_model_id | 実旧モデルで直接求めた区間平均。履歴と閾値を条件どおりに組むための測定値。前specと同名同義 |
| LEGACY_ACTION_BY_ALARM_INTERVAL_RESOLUTION_OUTCOME | 警報区間の結果種別→旧actionの対応（test内だけ）。既存の確定testのLEGACY_ACTION_BY_RESOLUTION_OUTCOME（候補検証の確定の結果種別→旧action）と同じ形で、対象が警報区間であることを名前で区別する |
| ALARM_INTERVAL_RESOLUTION_CASES / alarm_interval_resolution_case | 履歴の与え方と期待する分岐の条件名（別モデルの再利用、現行の維持、現行が適合しても別モデル、同率、負ID、適合なし等）とそのparameter。既存の確定testのLEGACY_RESOLUTION_CASES（旧の確定分岐の一覧）とは別で、新旧共通の条件を表すためLEGACY_を付けない |
| statistics_by_model_id | 上流helperの引数と同名同義 |
| expected_resolution_outcome / expected_assigned_model_id | 条件から決まる期待値 |
| candidate_parameter_initialization_source / LEGACY_INITIALIZATION_NAMES | 前specのtestと同名同義（定数は前specのtestからimportして再利用） |
| alarm_change_interval_resolution | 対象関数の戻り値 |
| assert_alarm_change_interval_resolution_matches_legacy | 結果recordと全状態（標本・計数・統計・現行ID・開始session・乱数）を実旧と照合する |
| snapshot_alarm_change_interval_resolution_state / assert_alarm_change_interval_resolution_state_unchanged | 保有順・owner同一性・全parameter/grad・optimizer state・学習mode・統計・標本・計数・現行ID・3乱数の記録と不変の照合 |
| state_snapshot / valid_state_snapshot | 上の記録。後者は不正入力へ差し替える前の記録（前specと同名同義） |
| operation_calls | 既存部品（区間評価・吸収・初期値選択・session開始）と帰属切替が呼ばれた順の記録 |
| record_operation_call | operation_callsへ名前を追加してから元の部品を呼ぶ記録用wrapperを作る関数（選択や判定を行わない） |
| operation_name | 上で記録する部品名 |
| initial_rng_state / expected_rng_state | 上流と同名同義 |
| invalid_case / field_name / field_value / expected_exception / exception_info | 拒否testのparameterと局所名。上流と同名同義 |
| TupleSubclass / IntSubclass | exact型の拒否入力用subclass。前specと同名同義 |
| InputSubclass | 所有者のsubclass instanceを作る動的type名。上流と同名同義 |
| sample_index / sample_count / class_count / held_model_ids / optimizer_variant / registry / held_model_training_states | 上流と同名同義 |
| test_alarm_change_interval_resolution_matches_legacy | 分岐ごとの実旧対照 |
| test_alarm_change_interval_resolution_starts_candidate_validation_like_legacy | 3初期化方式×履歴条件での候補検証開始の実旧対照 |
| test_alarm_change_interval_resolution_is_immutable | recordのfrozen/kw_onlyと結果種別の検査 |
| test_alarm_change_interval_resolution_rejects_common_inputs_without_mutation | 共通入力の拒否と不変 |
| test_alarm_change_interval_resolution_rejects_start_only_inputs_only_without_reuse | 開始だけの入力の拒否と、再利用時は検査しないこと |
| test_alarm_change_interval_resolution_calls_only_selected_operations_in_order | 分岐ごとの呼出しと順序 |
| test_resolved_alarm_change_interval_continues_joint_training | 解決後の共同学習が実旧と一致 |
| test_alarm_change_interval_resolution_dependency_contract | exact依存の注入契約 |
| alarm_change_interval_resolution_cpu_smoke.py / alarm_change_interval_resolution_mutation_evidence.py | Git管理外のfresh CPUと実source変異・復元の証拠 |

上流testから同義で再利用するhelper: build_fixation_oracle、set_overall_loss_statistics_in_both_implementations、assert_started_session_matches_legacy、assert_training_samples_match_legacy、assert_model_counts_match_legacy、assert_store_statistics_match_legacy、snapshot_parameter_values_and_gradients、assert_parameter_values_and_gradients_unchanged、assert_nested_state_equal、run_legacy_joint_update、convert_legacy_parameter_name。testを書く過程でこの表にない名前が必要になったら、コードへ書く前に追加表を登録して独立レビューを受ける。
