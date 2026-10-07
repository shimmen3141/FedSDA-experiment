# 命名と役割 revision5

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

## 追加 revision4（Task 2/3のtestで使う名前。testをリポジトリへ置く前の登録）

revision3までの表は変更していない。主担当がTask 2のtestをリポジトリ外で下書きし、表にない名前を洗い出した。sourceに新しい名前はない。

| 名前 | 種別 | 役割 |
| --- | --- | --- |
| build_alarm_change_interval_resolution_oracleの引数alarm_interval_resolution_case | helper引数 | r3では戻り値だけを書いた。条件名を受け取り、保有ID・同率用の値の複製・履歴統計を条件どおりに両実装へ設定する。ほかの引数はclass_count、monkeypatch、optimizer_variant、candidate_parameter_initialization_source |
| ALARM_INTERVAL_RESOLUTION_CASESの各値のkey: held_model_ids / statistics_by_model_id / expected_resolution_outcome / expected_assigned_model_id | dictのkey | 条件ごとの保有ID、実測の区間平均から履歴統計を作る関数、期待する結果種別と吸収先ID。keyは登録済みの同名と同義 |
| INVALID_COMMON_INPUT_CASES | test定数 | 共通入力の不正の条件名→（引数名、正常なresolution_argumentsから不正値を作る関数または"subclass"、期待する例外型） |
| INVALID_START_ONLY_INPUT_CASES | test定数 | 候補検証の開始だけに使う入力の不正の条件名→（引数名、不正値、期待する例外型） |
| replace_change_interval_sample | test helper | 標本列のsample_index番目だけを、指定のfieldを差し替えた標本へ置き換えた標本列を返す。拒否入力の作成用 |
| replaced_fields | 上のhelperの可変keyword引数 | 差し替えるfield名→値（input_features、observed_class_labels） |
| event_fields | record_legacy_adaptation_eventの可変keyword引数 | 実旧がイベント記録へ渡したfield。前specと同名同義 |
| initial_training_model_id | test local / 照合helperの引数 | 処理前の現在の学習帰属ID。上流sessionのfieldと同名同義 |
| current_state_snapshot | test local | 不変の照合時に取り直した状態の記録 |
| state_snapshotのkey: resolution_arguments / held_model_training_states / parameter_snapshot / training_modes / optimizer_state_snapshots / loss_statistics_snapshot / training_sample_snapshot / counts_snapshot / current_training_model_id / random_states | dictのkey | 記録時の引数dictのcopy、保有状態、parameter/grad、学習mode、optimizer state、履歴統計、標本（モデルIDとTensorの同一性）、計数、現行ID、3乱数。held_model_training_states、parameter_snapshot、training_modes、optimizer_state_snapshots、loss_statistics_snapshot、random_statesは前specの同名keyと同義。resolution_arguments、training_sample_snapshot、counts_snapshot、current_training_model_idは今回追加 |
| torch_random_state / python_random_state / numpy_random_state / numpy_state | test local | 前specと同名同義 |
| counts_store | test local | 割当概念計数と学習計数の所有者。上流testと同名同義 |
| legacy_training_samples / legacy_training_sample | test local | 実旧の標本一覧とその1件（特徴, ラベル, 概念）。上流testと同名同義 |
| concept_counts | test local | 実旧の1モデルぶんの概念ID→件数 |
| legacy_session | test local | 実旧が開始した警報後検証session。上流と同名同義 |
| legacy_epoch_training_call | test local | legacy_epoch_training_callsの1件 |
| candidate_epoch_training_result / collection_state | test local | 開始sessionの学習結果と損失収集の状態。上流testと同名同義 |
| held_parameter_storage_addresses | test local | 保有モデルのparameter実体のアドレス集合。上流testと同名同義 |
| collection / expected_training_sample | test local | 標本storeの1モデルぶんの標本集合、区間の標本列の対応する1件 |
| loss_statistics / previous_loss_statistics | test local | 1モデルの現在の統計と、処理前の統計 |
| assigned_concept_counts / previous_concept_counts | test local | 吸収先モデルの処理後・処理前の概念ID→件数 |
| owner / previous_optimizer / previous_optimizer_state | test local | optimizer stateの不変照合用。上流testと同名同義 |
| operation_calls（Task 2での使い方） | test local | r3の役割に加え、Task 2の拒否testでは区間評価だけを包み、評価より前に拒否されたことを呼出し回数で確かめる |
| run_joint_update_in_both_implementations | test内の関数（Task 3） | 両実装で、各自の標本storeにある全標本をモデルごとの固定batchにして共同学習を1回行い、損失と全状態を照合する。上流の吸収testの同名関数と同義 |
| legacy_training_batches / participating_training_batches / training_bindings / model_training_sample_collections / local_training_settings / update_shared_features / expected_joint_loss / actual_joint_loss | test local（Task 3） | 上流の吸収testと同名同義 |
| validation_samples / sample_offset / legacy_reference_model | test local（Task 3） | 開始後の実観測に使う標本（位置, 特徴, ラベル）と番号、実旧の参照モデル。前specのtestと同名同義 |

上流testから同義で再利用するhelperの追加: build_session_start_oracle（session開始の設定と旧configの差し替えを含む実NN oracle。標本と計数の新ownerは、このoracleが実旧clientへ置いた内容を同じobject・同じ順で写して作る）、assert_candidate_epoch_training_matches_legacy、assert_fixed_references_match_legacy、assert_held_model_states_match_legacy、assert_collected_losses_match_legacy。r3で挙げたbuild_fixation_oracleとassert_started_session_matches_legacyは使わない（後者は区間のTensorが旧と同一objectであることを要求するが、本specは区間を新しく連結するため、同じ内容を個別に照合する）。

## 追加 revision5（Task 1/2のtestの未登録名の事後登録と、Task 3のtestで使う名前の事前登録）

revision4までの表は変更していない。sourceに新しい名前はない。

### 事後登録（Task 1/2のtestに既にある名前。独立レビューが未登録と指摘した）

commit 1f6fd4a・55a3b92のtestに、登録していない局所名が6つあった。Task 2のtestは下書きから名前を洗い出して登録したが、次の名前を見落とした。登録が後になった手順逸脱として記録する。

| 名前 | 種別 | 役割 |
| --- | --- | --- |
| session_start_arguments | test local | 上流build_session_start_oracleの戻り値（session開始の引数dict）。oracle内で、開始だけに使う入力と所有者を取り出す。上流と同名同義 |
| state / previous_state | test local | 保有状態の1件と、記録時の対応する1件。上流・前specのtestと同名同義 |
| mean_loss | test local | 実測の区間平均の1値。前specと同名同義 |
| legacy_model | test local | 実旧の保有モデルの1件。前specのtestと同名同義 |
| expected_started_validation_session | test local | record testで、結果種別ごとにstarted_validation_sessionへ入れて保持を確かめる値（開始以外ではNone） |
| invalid_case（record testでの使い方） | test local | 登録済みの「拒否入力の条件名」に加え、record testでは正式値以外の結果種別そのものを指す |
| field_value（record testでの使い方） | test local | 登録済みの「不正値」に加え、record testではtraining_model_assignment_changeへ入れる値を指す |

### 事前登録（Task 3のtest。リポジトリ外の下書きから洗い出した）

| 名前 | 種別 | 役割 |
| --- | --- | --- |
| record_operation_callの引数 operation_name / operation | 関数の引数 | 記録する部品名と、包む元の部品（関数またはmethod） |
| recorded_operation | record_operation_callが返す関数 | 呼出しをoperation_callsへ（部品名, keyword引数）として記録してから、元の部品をそのまま呼んで戻り値を返すwrapper。選択や判定を行わない |
| operation_arguments / operation_keyword_arguments | recorded_operationの可変引数 | 元の部品へそのまま渡す位置引数とkeyword引数。後者をoperation_callsへ記録する |
| training_batch | test local | participating_training_batchesの1件。上流の吸収testと同名同義 |
| random_states（local） | test local | 解決直前の3乱数状態。state_snapshotの同名keyと同義 |
| test_alarm_change_interval_resolution_observes_samples_after_started_validation_like_legacy | test | 候補検証開始の後、実観測の損失と到達判定が実旧と一致することを確かめる（tasksのTask 3「候補検証開始の後は、実観測の損失が実旧と一致する」）。共同学習の継続testとは別関数にする。独立レビューの提案で、他のtest名と同じ接頭辞にした |
| losses | test local | 実旧の1参照モデルぶんの損失の列。上流のsession開始testと同名同義（事後登録） |

条件名（pytestのparametrize idになる文字列）は、ALARM_INTERVAL_RESOLUTION_CASES、INVALID_COMMON_INPUT_CASES、INVALID_START_ONLY_INPUT_CASESの各dictのkeyとして定義する。test側のdictが正本で、keyの追加は同じファイル内で完結するので、個別には本表へ列挙しない。旧の再利用計数のkey（alternative_fit、current_fit）は旧実装側の名前で、testの照合にだけ使う。
