# 単一run specの命名

- revision: 3。task 1の依存検査名を追加。承認状態はspec.jsonを確認する。
- 新しい名前の承認はgpt-6-lunaレビューと主担当の有用な指摘反映による。ユーザーへ命名だけの承認を再要求しない。
- 現段階では新src・実装を先取りするテストを作らない。

## 引き継ぐ正式名と利用範囲

| 名前 | この段階での扱い |
|---|---|
| `federated_learning_experiments` | 承認済みの新パッケージ |
| `sine2` | 既存ベンチマーク識別子。変更しない |
| `ExperimentRunConditions` | dataset・seed・規模・同期区間の固定条件 |
| `ValidatedExperimentRunSettingsSubset` | 機能の一部を束ねた検証済み部分型。完全な実験実行設定へ昇格させない |

既存の名称・値域の正本は`../configuration-foundation/naming.md`。
調査表の旧名は移植元の位置を示すための表記であり、新APIのaliasではない。

## ファイル・型・所有状態

パスはsrc/federated_learning_experiments以下。型名はこの表を正本とする。

| ファイル | 型 | 役割・所有状態 |
|---|---|---|
| data/concept_schedules/random_concept_schedule_settings.py | RandomConceptScheduleSettings | 変更試行の方式・最小位置差・確率。不変の固定条件 |
| data/concept_schedules/random_concept_schedule_generation.py | 型の追加なし | 全clientの概念系列生成。モデル状態を持たない |
| data/observed_streams.py | ObservedSample | 長さ2の特徴とクラスラベル。不変、真の概念なし |
| 同上 | ClientObservedStream | client IDと標本列。不変 |
| 同上 | ClientConceptTrace | 評価用の真の概念列。不変、学習入力と別 |
| data/sine/sine_sample_generation.py | SineSampleGenerator | runのNumPy乱数を借りて1標本を生成。乱数は消費するが設定を変更しない |
| execution/stream_protocol_execution_settings.py | StreamProtocolExecutionSettings | データ・処理順だけの設定集約。完全な手法設定とは異なる |
| execution/run_participant_contracts.py | RunParticipantFactory | 自身の固定条件検証と、新しいrun参加者の初期準備のProtocol |
| 同上 | RunClientOperations | 標本処理・区間末・候補終端のProtocol。学習の計算を定義しない |
| 同上 | RunServerOperations | 状態記録・同期・通信終端のProtocol |
| 同上 | RunParticipants | client操作のtupleとserver操作の集合。各状態は接続先が所有 |
| execution/run_execution_records.py | RunExecutionEvent | 成功した段階と位置。不変 |
| 同上 | StreamProtocolRunResult | データ供給・順序の成功記録。研究指標や参加者参照を含めない |
| execution/run_execution_errors.py | RunExecutionError | 失敗段階・位置・理由。元例外は標準causeで保持 |
| execution/stream_protocol_execution_loop.py | 型の追加なし | 完全区間と終端の逐次呼出順 |
| execution/run_random_sources.py | RunRandomSources | run専用Python/NumPy乱数の実体。設定ではなく可変の実行状態 |
| learning/models/torch_random_state_scope.py | 型の追加なし | 呼出元のCPU torch乱数をcontext出口で復元 |
| runtime/single_run_execution.py | 型の追加なし | 検証・初期準備・生成・進行の組立 |

## フィールド・引数と意味

| 所属 | 名前 | 型・単位・意味 |
|---|---|---|
| RandomConceptScheduleSettings | concept_schedule_strategy | str。random_changes_after_minimum_index_gapだけ受理 |
| 同上 | minimum_sample_index_gap_before_change_trial | int >=0、sample/client。現在位置と直近変化位置の差がこの値より大きい時に試行。100なら初回101 |
| 同上 | per_eligible_sample_concept_change_probability | float [0,1]、無次元。試行可能な各標本での変更確率 |
| StreamProtocolExecutionSettings | experiment_run_conditions | 承認済みExperimentRunConditions |
| 同上 | concept_schedule_settings | RandomConceptScheduleSettings |
| 同上 | execution_strategy | str。sample_index_then_client_order_with_interval_synchronizationだけ受理 |
| ObservedSample | feature_values | tuple[float, float]。float32に丸めた2特徴の値 |
| 同上 | class_label | int 0/1。クラス。真の概念IDではない |
| ClientObservedStream | client_id, observed_samples | int、tuple[ObservedSample, ...]。位置はtupleのindex |
| ClientConceptTrace | client_id, concept_ids_by_sample_index | int、tuple[int, ...]。真の概念ID0/1 |
| RunRandomSources | python_random_generator | random.Random。系列生成・shuffle用の可変乱数源 |
| 同上・SineSampleGenerator | numpy_random_generator | numpy.random.RandomState。旧NumPy乱数方式に対応する可変乱数源 |
| RunParticipants | client_operations, server_operations | tuple[RunClientOperations, ...]、RunServerOperations |
| RunExecutionEvent・RunExecutionError | stage_name | 正式ステージ名。下表 |
| 同上 | client_id, sample_index, round_index | intまたはNone。対象がある段階だけ位置を設定。全て0始まり |
| RunExecutionError | failure_reason | str。日本語の説明。設定値域のvalidation_failure_reasonとは別 |
| StreamProtocolRunResult | observed_client_streams | tuple[ClientObservedStream, ...]。不変の全観測列 |
| 同上 | evaluation_concept_traces | tuple[ClientConceptTrace, ...]。評価用。処理部に渡さない |
| 同上 | generated_sample_count_per_client | int、sample/client。指定T |
| 同上 | processed_sample_count_per_client | int、sample/client。A*(T//A) |
| 同上 | unprocessed_tail_sample_count_per_client | int、sample/client。T%A |
| 同上 | synchronization_interval_count | int、区間/run。T//A。終端呼出数とは別 |
| 同上 | execution_events | tuple[RunExecutionEvent, ...]。実際の成功順 |

## 関数・操作と入出力

公開呼出はkeyword-only。生成物と乱数の受渡しは明示し、旧configを読む引数なしの生成器を作らない。

| 名前 | 引数 → 戻り値 | 状態更新・役割 |
|---|---|---|
| validate_stream_protocol_execution_settings | execution_settings → None | 狭い設定型・方式・dataset・seed範囲を検証。入力更新なし |
| generate_random_client_concept_traces | experiment_run_conditions, concept_schedule_settings, python_random_generator → tuple[ClientConceptTrace, ...] | Python乱数を消費してclient順に全系列を作る |
| SineSampleGenerator.generate_sample | concept_id → ObservedSample | 概念ID0/1から1標本。NumPy乱数を消費 |
| build_sine_client_observed_streams | evaluation_concept_traces, sample_generator → tuple[ClientObservedStream, ...] | 全系列の後、client順に標本生成 |
| create_run_random_sources | random_seed → RunRandomSources | 新しい乱数源を作る。呼出元のglobal乱数を変更しない |
| isolated_cpu_torch_random_state | random_seed → ContextManager[None] | CPU torch乱数を保存・seed設定・全出口で復元 |
| execute_stream_protocol_run | execution_settings, participant_factory → StreamProtocolRunResult | 組立。完全なFedSDA実験の入口ではない |
| run_stream_protocol_intervals | participants, observed_client_streams, server_aggregation_interval_per_client_samples → tuple[RunExecutionEvent, ...] | 区間・終端を進める。真の概念列を受け取らない |
| RunParticipantFactory.validate_configuration | 引数なし → None | factory自身の条件を副作用なく確認 |
| RunParticipantFactory.prepare_run | experiment_run_conditions, run_random_sources, sample_generator → RunParticipants | 初期準備。参加者を新規に構築。学習内容は接続先の責務 |
| RunClientOperations.process_observed_sample | observed_sample, sample_index → None | 観測値だけを実処理へ渡す |
| RunClientOperations.flush_pending_local_updates | round_index → None | 区間末の保留更新確定 |
| RunClientOperations.has_model_ready_for_server_registration | 引数なし → bool | モデル存在かつ送信可能。単なる候補存在ではない |
| RunClientOperations.advance_new_model_upload_wait_after_synchronization | round_index → None | 同期後の待ち期間進行。次回送信可能な状態への移行はclientが所有 |
| RunClientOperations.finalize_incomplete_candidate_validation | 引数なし → None | 未完了の候補検証を終端確定 |
| RunServerOperations.record_client_states_before_synchronization | round_index → None | 同期前の状態記録 |
| RunServerOperations.synchronize_models | round_index, new_model_registration_available → None | boolの時点情報を受けて同期。統合判断・計算はserverが所有 |
| RunServerOperations.finalize_started_communications | completed_round_count → None | 開始済み通信の確定。通常同期の追加ではない |

型コンストラクタの引数はフィールドと同名。上表の引数名と同じ意味の一時値は同名を使う。
実装の補助名・テスト名は担当task開始前に追加レビューする。Python慣用のself・__post_init__は維持する。

## ステージの正式値

| stage_name | 成功イベントまたは失敗位置の意味 |
|---|---|
| configuration_validation | 入力とfactoryの事前検証 |
| initial_preparation | factoryの初期準備 |
| participant_validation | 初期準備後の参加者契約の確認 |
| concept_trace_generation | 全clientの概念列生成 |
| observed_stream_generation | 全clientの標本生成 |
| sample_processing | 指定client・標本位置の処理 |
| pending_update_flush | 指定client・区間末の更新確定 |
| pre_sync_recording | 指定区間の同期前状態記録 |
| registration_readiness_check | 指定clientの送信可能モデル照会 |
| server_synchronization | 指定区間の同期 |
| upload_wait_advance | 指定clientの同期後の待ち期間進行 |
| incomplete_candidate_validation_finalization | 指定clientの候補検証終端 |
| started_communication_finalization | 通信の終端確定 |

## task 1: package境界と依存検査

productionへの追加はdata、data/concept_schedules、data/sine、execution、runtimeの各__init__.pyのみ。日本語の境界説明を持ち、再exportしない。
新テストファイルはtests/refactoring/test_single_run_dependency_boundaries.py。

| 名前 | 役割・入出力・状態 |
|---|---|
| test_configuration_foundation_imports_only_allowed_dependencies | 既存の全package検査を設定基盤対象へ限定したテスト名。標準ライブラリ制約は保持 |
| test_single_run_package_boundaries_have_no_exports | package境界の存在と再exportがないことを確認 |
| test_single_run_layers_import_only_allowed_dependencies | 新srcをAST走査し、各層の許可依存を確認 |
| test_single_run_dependency_checker_rejects_forbidden_imports | 禁止import fixtureを実際の検査関数へ渡して拒否を確認 |
| test_single_run_dependency_checker_accepts_allowed_imports | NumPy例外・専用torch・相対importの許可を確認 |
| collect_dependency_boundary_violations | source_module_path, source_text → tuple[tuple[str, str], ...]。禁止依存と日本語の理由を返す。状態変更なし |
| resolve_imported_module_names | import_statement, importing_package_name → tuple[str, ...]。相対importを絶対名へ解決 |
| dependency_is_allowed | source_module_path, imported_module_name → bool。設計の層別許可条件を確認 |
| is_configuration_foundation_module | source_module_path → bool。core・configuration・機能別設定を判別 |

既存局所名source_module_path、source_file_path、package_source_directory、importing_package_name、parsed_source_module、import_statement、imported_module_names、imported_module_name、imported_module_aliasは同じAST/パスの意味で使う。

| 新しい局所名・テスト引数 | 意味・型 |
|---|---|
| source_text | str。読込ソースまたはfixtureのPython文字列 |
| dependency_boundary_violations | tuple[tuple[str, str], ...]。禁止モジュールと理由 |
| dependency_boundary_violation_reason | str。違反理由の日本語説明 |
| allowed_internal_module_prefixes | tuple[str, ...]。その層の許可内部モジュールprefix |
| source_module_layer | str。相対パスの第1成分 |
| package_boundary_paths, package_boundary_path | tuple[str, ...]とstr。必要なpackage境界とその1件 |
| relative_import_prefix | str。相対importのlevelに対応するドット列 |
| imported_base_module_name | str。from節の絶対モジュール名 |
| expected_imported_module_name | str。違反として確認するモジュール名 |
| configuration_foundation_module | bool。設定基盤の対象か |

全てテスト内の一時状態で、productionの実行状態を更新しない。
