# 単一run specの命名

- revision: 10。task 4.3の接続・記録契約名を追加。承認状態はspec.jsonを確認する。
- 新しい名前の承認はgpt-6-lunaレビューと主担当の有用な指摘反映による。ユーザーへ命名だけの承認を再要求しない。
- 現段階では新src・実装を先取りするテストを作らない。

## task 4.3: 接続・記録契約の追加名

公開型・操作・フィールドは下の既承認表を使う。参加者の件数・ID・必要操作検査はruntimeが所有する。
RunParticipantsは不変の容器でclient tupleと処理部参照を保持する。成功結果は参加者参照を保持しない。
イベント位置はNoneを既定とし、結果の全フィールドは必須。正式stage値と位置はイベント・例外で同じ検査を使う。

| 名前 | 役割・入出力・状態 |
|---|---|
| RUN_EXECUTION_STAGE_NAMES | tuple[str, ...]。下表の13正式stage値。recordsで宣言しerrorsから共有 |
| validate_run_execution_stage_and_positions | keyword-only stage_name/client_id/sample_index/round_index → None。正式stageとNoneまたは非負int位置を検証。状態変更なし |
| position_parameter_name, position_parameter_value | strとintまたはNone。位置の検証対象項目・値 |
| sample_count_parameter_name, sample_count_parameter_value | strとint。結果件数を非負intとして検証する局所値 |
| observed_client_stream, evaluation_concept_trace, execution_event | 結果tupleの要素検証で使用する各記録 |
| test_run_participant_protocols_expose_only_declared_observation_operations | Protocolの公開member・keyword-only・真の概念入力なしを確認 |
| test_run_participants_preserve_operation_references_and_reject_mutable_client_collections | 参照保持・容器不変性・list拒否を確認 |
| test_run_execution_events_preserve_stage_and_optional_positions | 13stageと任意位置の保持を確認 |
| test_run_execution_records_are_frozen_and_keyword_only | イベント・成功結果の変更/削除/位置引数拒否を確認 |
| test_stream_protocol_results_preserve_observations_truth_counts_and_event_order | 観測・真値・件数・順序を保持し参加者・乱数・研究指標fieldがないことを確認 |
| test_stream_protocol_results_reject_mutable_or_invalid_records | list・不正要素・件数の拒否を確認 |
| test_run_execution_errors_preserve_stage_positions_reason_and_cause | 実raise-fromの元causeと日本語位置表示を確認 |
| test_run_execution_records_reject_unknown_stages_and_invalid_positions | 不明stage・負値・bool・非整数位置の拒否を確認 |
| protocol_type, protocol_member_names, expected_protocol_member_names | Protocol型・宣言member集合・期待集合 |
| operation_name, operation_parameters | 検査するmethod名とinspectの引数情報 |
| participants, operation_reference | RunParticipantsとテスト専用の処理部参照 |
| run_result, run_execution_error, original_exception | 成功記録・位置付き例外・元例外 |
| valid_run_result_field_values | fixture。各テストへ正常値dictを独立供給 |
| stage_names | テスト側の13stage期待値tuple。production定数を期待値に流用しない |

記録の既承認局所名は同じ役割で再利用する。真の概念列は処理部Protocolへ渡さない。

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

## task 2.1: 概念系列の設定検証

production名・フィールドは既存表のRandomConceptScheduleSettingsを用いる。__post_init__は既存metadata検証を呼び、追加のproduction補助名は作らない。
tests/refactoring/test_sine_stream_generation.pyへ以下を追加する。

| 名前 | 役割・入出力・状態 |
|---|---|
| test_random_concept_schedule_settings_accept_valid_values | 方式・位置差・確率の正常値と境界値を受理することを確認 |
| test_random_concept_schedule_settings_reject_invalid_values | 不正値で項目・元の値・理由が報告されることを確認 |
| test_random_concept_schedule_settings_require_all_fields | 必須項目不足の拒否を確認 |
| test_random_concept_schedule_settings_are_frozen_and_keyword_only | 不変性と位置引数の拒否を確認 |
| valid_concept_schedule_values | dict[str, object]。テスト用の正常設定値。各テストへ独立したコピーを渡すfixture |
| settings_instance | RandomConceptScheduleSettings。テストで構築した不変条件 |
| configuration_parameter_name, specified_parameter_value | strとobject。不正にしたフィールドと元の指定値。既存例外と同じ意味 |
| expected_validation_failure_reason | str。確認する例外理由の表記 |
| exception_info | pytestの捕捉例外記録 |
| missing_parameter_name | str。必須項目不足を確認するため除く項目 |

パラメータ化には承認済みのフィールド名をそのまま使用する。テストの設定dictだけを変更し、production・global設定は変更しない。

## task 2.2: 不変の観測値と評価情報

productionの型・フィールドは既存表を使い、補助関数は追加しない。各__post_init__でtuple・要素型・ラベル/概念/IDを検証して可変参照を拒否する。float32丸めはtask 3.2の生成側の責務。
テストはtests/refactoring/test_sine_stream_generation.pyに追加する。

| 名前 | 役割・入出力・状態 |
|---|---|
| test_observed_sample_preserves_features_and_binary_label | 2特徴・0/1ラベルを保持し真の概念フィールドを持たないことを確認 |
| test_client_data_records_preserve_ids_positions_and_counts | 観測列・概念列のclient ID・順序・件数・位置の対応を確認 |
| test_observed_data_records_are_frozen_and_keyword_only | 3型の変更・削除・位置引数の拒否を確認 |
| test_observed_data_records_reject_mutable_collections | リストや可変の要素を保持しないことを確認 |
| test_observed_sample_rejects_invalid_values | 特徴長・float型と、boolを除くint 0/1ラベルを確認 |
| test_client_data_records_reject_invalid_ids_and_items | 非負intのIDと標本・概念の要素を確認 |
| test_observed_data_records_require_all_fields | 3型の必須項目を確認 |
| observed_sample, observed_samples, client_observed_stream, client_concept_trace | 対応する型の実体、標本tuple。テスト内で構築しglobal状態へ残さない |
| feature_value, concept_id | floatとint。特徴・概念の検証ループの一時値 |
| record_type, record_field_values, record_instance | type、dict[str, object]、不変の記録。パラメータ化の型・入力・結果 |
| record_field_name, invalid_field_value, expected_exception_type | str、object、例外type。対象項目・不正入力・期待例外 |

例外捕捉は承認済みexception_infoを使う。設定値ではない不正な記録はTypeError/ValueErrorで報告し、既存coreを変更しない。

## task 3.1: 概念列と乱数消費

公開関数・引数は既存表を使用する。production補助関数は追加しない。

| 名前 | 役割・入出力・更新 |
|---|---|
| evaluation_concept_traces | list[ClientConceptTrace]。全clientの結果を蓄積してtupleで返す |
| concept_ids_by_sample_index | list[int]。当該clientの位置順の概念を蓄積してtupleにする |
| current_concept_id | int。現在概念0/1、各clientで0に初期化 |
| last_concept_change_sample_index | int。直近の実変化位置、初期値0。変化時だけ更新 |
| alternative_concept_ids | list[int]。現在概念以外の候補。一つでもchoiceで選び乱数消費を維持 |
| client_id, sample_index, concept_id | int。承認済みのclient ID・位置・候補概念IDと同じ意味 |
| test_random_client_concept_traces_match_reference_and_random_state | 旧helperと全系列・順序・件数・生成後乱数状態を照合 |
| test_random_client_concept_traces_use_strict_minimum_index_gap | gap100で101に初回変化、202に次回変化となることを確認 |
| test_random_client_concept_traces_consume_eligible_trials_with_zero_probability | 確率0でも対象位置ごとに試行乱数を消費しglobal乱数は不変と確認 |
| test_random_client_concept_trace_generation_requires_keyword_arguments | 公開keyword-only契約を確認 |
| reference_schedules_module | 旧data.schedulesのmodule。テスト側の参照専用 |
| reference_python_random_generator | Random。旧moduleへ一時注入する独立乱数源 |
| reference_concept_schedules | list[list[int]]。旧helperの結果 |
| initial_python_random_state, global_python_random_state | Python乱数の状態tuple。照合の開始状態・呼出元の保持状態 |
| expected_python_random_generator | Random。確率0での試行消費数の照合用 |
| eligible_sample_trial_count | int。変更試行可能な標本数 |
| preparation_random_draw_count, random_draw_index | int。系列前に消費する試行数と、消費ループの一時index |
| monkeypatch | pytest fixture。テスト中だけ旧moduleの乱数源を差し替えて復元 |

実験条件・設定・seed・client数・標本数には既承認名を使う。テスト引数の位置差と確率もminimum_sample_index_gap_before_change_trial、per_eligible_sample_concept_change_probabilityとし、gap/probabilityという別名は導入しない。

## task 3.2: SINE標本の生成と精度

公開型・関数・引数は既存表を使用する。SineSampleGenerator.__init__はnumpy_random_generatorを借用して同名フィールドに保持し、global乱数を参照しない。

| 名前 | 役割・入出力・状態 |
|---|---|
| float64_feature_values | ndarray。uniformから得た2特徴。ラベル判定前のfloat64値 |
| float32_feature_values | ndarray。ラベル判定後にfloat32へ丸めた2特徴 |
| below_sine_boundary | bool/np.bool_。第2特徴がsin(第1特徴)以下か |
| observed_client_streams | list[ClientObservedStream]。client順の結果を蓄積してtupleで返す |
| test_sine_samples_match_reference_and_random_state | 旧SINEとtorch float32特徴・ラベル・生成後乱数状態を照合 |
| test_sine_samples_reject_invalid_concepts_without_consuming_randomness | 不正概念IDを乱数未消費で拒否することを確認 |
| test_sine_labels_are_decided_before_float32_rounding | 丸めによって境界判定が反転する特徴で、ラベルを先に判定することを確認 |
| test_sine_client_observed_streams_preserve_client_order_positions_and_counts | 全概念列後のclient順標本供給とID・位置・件数を旧基準と比較 |
| test_sine_client_observed_streams_accept_empty_traces | 空入力/空概念列で追加乱数を消費しないことを確認 |
| test_sine_sample_generation_requires_keyword_arguments | 生成器構築・単一生成・全列構築のkeyword-onlyを確認 |
| assert_numpy_random_states_equal | actual_numpy_random_state, expected_numpy_random_state → None。NumPy状態の方式・key配列・位置・cacheを完全比較するテスト補助 |
| BoundaryFeatureRandomState | テスト専用のRandomState派生型。丸めでSINE境界が反転する固定float64特徴を返す |
| uniform | lower_bound, upper_bound, size → ndarray。上記fixtureの固定特徴供給。乱数を消費しない |
| lower_bound, upper_bound, size | float、float、int。既存RandomState.uniformと対応するfixture引数 |
| reference_synthetic_module | 旧SINE生成器のmodule、テスト参照専用 |
| reference_numpy_random_generator | RandomState。旧基準へ一時接続する独立乱数 |
| reference_feature_values, reference_class_labels | 旧helperが返した特徴list・ラベルlist |
| initial_numpy_random_state, global_numpy_random_state | NumPy状態tuple。照合の開始状態と呼出元の保持状態 |
| actual_numpy_random_state, expected_numpy_random_state | NumPy状態tuple。テスト補助へ渡す実際/期待状態 |
| invalid_concept_id, expected_class_label | objectとint。不正な概念入力と境界fixtureの期待ラベル |
| reference_float32_feature_values | list[float]。旧torch FloatTensorによる丸め後の特徴 |
| reference_observed_samples | list。旧標本の照合用蓄積 |
| sample_count | int。単一標本生成を繰り返すテスト内の回数 |

既承認の標本・概念・client位置名を同じ意味で使う。NumPyの慣用aliasはnp。テストでのみtorchとSimpleNamespaceを使い、旧moduleのnp参照を局所的に差し替えて復元する。

## task 4.1: run専用乱数とCPU状態の復元

productionは既存表のRunRandomSources・create_run_random_sources・isolated_cpu_torch_random_stateを使う。追加補助関数・production局所名は導入しない。
torch.manual_seedは全デバイスを対象とするため、CPUのdefault_generatorへ直接seedを設定し、CPUだけのcontextで全出口を復元する。
tests/refactoring/test_single_run_execution.pyへ以下を追加する。

| 名前 | 役割・入出力・状態 |
|---|---|
| test_run_random_sources_are_independent_and_reproduce_legacy_sequences | 各runの新規実体、旧乱数方式の値列と呼出元のglobal保持を確認 |
| test_cpu_torch_random_scope_restores_state_on_success_and_failure | 同じCPU seedの値列と、成功・例外出口の状態復元を確認 |
| test_cpu_torch_random_scope_does_not_seed_other_devices_or_change_runtime_defaults | 全device seed APIを呼ばず、thread数・dtypeを変更しないことを確認 |
| run_random_sources, repeated_run_random_sources | RunRandomSources。独立して作成する同条件の実行乱数 |
| expected_python_random_generator, expected_numpy_random_generator | RandomとRandomState。旧方式の期待値列を作る独立実体 |
| original_cpu_torch_random_state | torch.Tensor。context前のCPU状態、復元照合用 |
| expected_cpu_random_generator | torch.Generator。CPU値列の独立した参照 |
| observed_cpu_random_values | torch.Tensor。context内で消費した実際のCPU値列 |
| fail_inside_scope | bool。正常出口/例外出口を選ぶテスト引数 |
| non_cpu_seed_calls | list[int]。全device/他device seed APIへの呼出を記録するspy |
| original_torch_thread_count, original_torch_default_dtype | intとtorch.dtype。呼出前の実行既定値 |

global Python/NumPy状態とmonkeypatchは既承認名を再利用する。乱数容器は可変な実行状態であり、固定条件や成功結果として保存しない。

## task 4.2: 狭い実行条件の検証

productionは既存表のStreamProtocolExecutionSettingsとvalidate_stream_protocol_execution_settingsを使用し、補助関数・局所名を追加しない。
複合型の検査をこのモジュールで行い、既存の単純値検証は入れ子の固定条件へ適用する。

| 名前 | 役割・入出力・状態 |
|---|---|
| valid_stream_protocol_execution_settings | pytest fixture。SINE基準規模と概念系列条件の正常な不変集約 |
| test_stream_protocol_execution_settings_accept_seed_and_interval_boundaries | NumPyのseed上下限と、端数・区間長超過を受理することを確認 |
| test_stream_protocol_execution_settings_reject_invalid_components_and_choices | 入れ子の型・方式・dataset・seed範囲を拒否し、元の項目と値を確認 |
| test_stream_protocol_execution_validation_rejects_partial_settings | 設定基盤の部分集約を完全な実行条件として受理しないことを確認 |
| test_stream_protocol_execution_settings_are_required_frozen_and_keyword_only | 必須項目・変更/削除・位置引数の拒否を確認 |
| valid_run_settings_mapping | 設定基盤テストの既存fixture。部分集約の正常値を再利用する |
| partial_run_settings | ValidatedExperimentRunSettingsSubset。拒否を確認する既存部分集約の実体 |

execution_settings、experiment_run_conditions、concept_schedule_settings、configuration_parameter_name、specified_parameter_value、exception_info、missing_parameter_nameは既承認の役割で再利用する。テストの変更にはdataclasses.replaceを使い、固定条件の実体を書き換えない。
