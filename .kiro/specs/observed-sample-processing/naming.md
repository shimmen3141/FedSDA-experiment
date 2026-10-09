# 標本1件の処理 — 命名 revision6

sourceの名前と、testのmodule直下の名前（test関数の名前を除く）の実装前一覧（範囲は共通引継ぎ手順）。リポジトリ外の下書きを作業ツリーの複製へ置いて`spec_checks.py names`で照合した。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| observed_sample_processing.py | runtimeの新module。観測した標本1件に対する学習側の処理をつなぐ。既存`alarm_occurrence_handling.py`（警報1回ぶん）や`released_pending_sample_assignment.py`（警報のない標本での確定）を呼ぶ側。既存`post_alarm_candidate_validation_sample_observation.py`（候補検証が標本を観測する1段）とは別で、標本1件に対する全段を扱う。 |
| `process_observed_sample` | 標本1件を処理する関数（旧`process_one_step`から予測を除いたもの）。「observed」は、ラベルを観測済みの標本であることを表す。 |
| `ObservedSampleProcessing` | 上の関数の結果record。既存`AlarmOccurrenceHandling`（警報1回ぶんの結果）と同じく、処理の名詞形をrecord名にする。 |
| `held_validation_advance` / `loss_monitoring_observation` / `alarm_occurrence_handling` / `released_sample_observations` / `completed_joint_update_losses` | 結果recordのfield。順に、候補検証の進行の結果（既存`HeldCandidateValidationAdvance`）、監視の観測（既存`LossMonitoringObservation`）、警報の処理の結果（既存`AlarmOccurrenceHandling`。警報がなければNone）、確定した標本、完了した共同更新の損失。後の2つは、既存の関数の戻り値の同名と同じ役割。 |
| `_validate_observed_sample_processing_inputs` | 上の関数の、最初の状態更新より前の検査をまとめた非公開の関数。 |
| `last_monitoring_observation` | 局所名。監視の最後の観測（位置の連続性の検査に使う）。field名`loss_monitoring_observation`（今回の観測）と区別する。 |
| `pending_observation` | 局所名。保留標本1件。引数`indexed_observation`（今回の標本）と区別する。 |
| `observed_loss` / `current_model_loss_statistics` | 局所名。現在のモデルでのこの標本の損失／現在のモデルの履歴統計。 |
| `_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME` / `owners_by_argument_name` | 引数名から、本処理が受け取るownerのexact型への対応（module直下の定数）と、引数名から渡されたownerへの対応（検査の関数の引数）。 |
| `_select_validation_assignment_sample_concept_ids` / `validation_assignment_training_samples` / `latest_pending_observations` | 開始した候補検証へ渡された標本の概念IDを、警報時の保留標本の末尾から返す非公開の関数と、その引数（sessionのfield `pending_assignment_training_samples`の値。保持する側の名前`validation_assignment_sample_concept_ids`にそろえる）、保留標本の末尾（局所名。既存`retain_latest_pending_sample_observations`の「latest」と同じ意味）。revision2で追加し、revision3で対応づけの方法を末尾の一致へ変えた（仕様レビューの指摘）。 |
| pending_sample_observation_store.py / `PendingSampleObservationStore` | 保留中の標本そのもの（位置・標本・概念ID）を持つowner。既存`PendingTrainingAssignmentBuffer`（保留の位置だけを持ち、容量と解放を決める）と対で使う。既存`ModelTrainingSampleStore`（帰属が確定した標本をモデル別に持つ）とは別で、帰属が未確定の標本を持つ。 |
| `append_pending_sample_observation` / `snapshot_pending_sample_observations` | 標本を末尾へ足す／保持中の標本を古い順のtupleで返す。 |
| `retain_latest_pending_sample_observations` / `retained_sample_indices` | 新しい側の標本だけを残す操作と、残す位置の引数。「latest」は、残す位置が保持中の並びの末尾であることを表す（どの位置を残すかは保留位置のownerが決める）。 |
| `validation_assignment_sample_concept_ids` | property。開始した候補検証へ渡した標本（確定のときに帰属が決まる標本）の概念ID。保持していなければNone。既存の引数`pending_assignment_sample_concept_ids`（候補検証の進行が受け取る同じ値）の、保持する側の名前。 |
| `hold_validation_assignment_sample_concept_ids` / `release_validation_assignment_sample_concept_ids` / `sample_concept_ids` | 上の値を保持する／外して返す操作と、その引数。既存`CandidateValidationSessionHolder`の`hold_validation_session`・`release_validation_session`と同じ語を使う。 |
| `_validate_concept_ids` | 概念IDの組の型を確かめる非公開の関数。 |
| `retained_sample_count` / `pending_sample_count` | 局所名。残す件数／保持中の件数。 |
| loss_change_alarm_record_store.py / `LossChangeAlarmRecordStore` / `LossChangeAlarmRecordSnapshot` | 損失の監視の値の列と、警報ごとの位置を記録するownerと、その読取り。既存`AdaptationRecordStore`（警報や確定に対して行った適応の結果を記録する）とは別で、検出の側の記録を持つ。 |
| `append_monitored_log_e_value` / `monitored_log_e_values` | 標本1件の監視の値（混合したe値の対数。既存`LossMonitoringObservation.log_e_value`）を足す操作と、その列（旧`history_detector_log_e`）。 |
| `append_alarm_record` / `alarm_sample_indices` / `estimated_change_point_sample_indices` / `detector_candidate_start_sample_indices` | 警報1回の位置を足す操作と、3つの列（旧`detected_event_positions`・`estimated_drift_start_positions`・`detector_candidate_start_positions`）。単数形の引数名は、既存の同名（警報の処理の`alarm_sample_index`・`estimated_change_point_sample_index`、監視の観測の`detector_candidate_start_sample_index`）と同じ役割。 |

引数`loss_change_alarm_record_store`・`pending_sample_observation_store`は、上のownerを受け取る。そのほかの引数と、局所名`owner_argument_name`・`owner`・`required_owner_type`・`sample_index`・`pending_assignment_state`・`input_features`・`observed_class_labels`・`current_training_model_id`・`pending_sample_observations`・`estimated_change_span_sample_count`・`estimated_change_point_sample_index`・`alarm_buffer_response`・`active_validation_session`・`training_sample`・`observed_concept_id`・`log_e_value`は、既存の部品の同名と同じ役割。非公開の属性`_pending_sample_observations`・`_validation_assignment_sample_concept_ids`・`_monitored_log_e_values`・`_alarm_sample_indices`・`_estimated_change_point_sample_indices`・`_detector_candidate_start_sample_indices`は、対応する公開の読取りと同じ値を持つ。importする既存symbol（設計5節）も定義元と同じ役割。

## test（module直下の名前。test関数の名前は対象外）

| 名前 | 役割 |
| --- | --- |
| test_observed_sample_processing.py | 新test module。標本1件の処理を、実旧の標本処理と標本ごとに照合する。 |
| `sample_processing_module` | 新moduleのimport別名（test専用）。各段の呼出しを記録するために使う。 |
| `LEGACY_ACTION_BY_ADAPTATION_OUTCOME` | 適応記録の結果種別から旧イベントのactionへの対応（既存の警報の対応と候補検証の対応を合わせたもの）。 |
| `UPDATE_INTERVAL` / `ITERATIONS_PER_REQUEST` / `BATCH_SAMPLE_COUNT` / `UPLOAD_DELAY_ROUND_COUNT` / `VALIDATION_SAMPLE_COUNT` | 両実装へ与える設定値。 |
| `STREAM_NAMES` / `HISTORICAL_MEAN_LOSS_PROFILES` / `TRAJECTORY_ORACLE_CASES` / `PROCESSED_SAMPLE_COUNT` | 対照の条件: 標本列の種類、保有モデルの履歴統計の平均、最初の警報の結果、処理する標本数。 |
| `EXPECTED_TRAJECTORY_CONDITION_COUNT` / `OBSERVED_OUTCOMES_BY_CONDITION` | 対照の条件数と、条件ごとに観測した適応結果（全結果を通ったことの確認に使う）。 |
| `build_sample_processing_oracle` / `build_default_sample_processing_oracle` | 最初の警報を両実装に処理させ、その直後から標本処理を続けられる状態を作る。後者は、候補検証を保持して始まる既定の条件（保持なしで始まる条件も選べる）。 |
| `make_stream_observation` / `make_next_observation` | 標本列の標本／保留位置の最終観測位置の次の位置の標本を作る。 |
| `run_legacy_sample_processing` | 実旧の標本処理を、指定の乱数状態から実行し、実行後の乱数状態を返す。 |
| `assert_sample_processing_state_matches_legacy` | 全ownerの状態を、実旧clientの対応する属性と照合する。 |
| `snapshot_sample_processing_state` / `assert_sample_processing_state_unchanged` | 標本1件の処理が触れうる全ownerの読取りと、不変の確認。 |
| `make_store_with_other_pending_observations` / `make_store_with_inverted_concept_id_holding` | 拒否条件に使う、保留標本だけ／概念IDの保持の有無だけが違う保留標本のowner。 |
| `REJECTION_MESSAGES` / `make_monitor_observed_at` | 拒否条件名から、例外の文言の一部への対応（どの検査が拒否したかを確かめる）／渡された監視の複製へ、指定の位置で1件観測させたものを作る（監視の位置の連続性の拒否条件に使う）。revision6で追加（変異toolで見つかったtestの穴へtestを足したときの名前。実装のレビューで登録漏れを指摘された）。 |
| `OWNER_ARGUMENT_NAMES_VALIDATED_FIRST` / `INVALID_SAMPLE_PROCESSING_INPUT_CASES` | 最初に型を確かめる引数名（本処理が受け取るownerすべてと、乱数生成器）／拒否条件名から（差し替える引数を作る操作、期待する例外）への対応。 |
| `SAMPLE_PROCESSING_STEP_NAMES` / `record_sample_processing_steps` | 標本1件の処理が呼ぶ段の名前／各段の呼出しを、順と、その時点の状態つきで記録する。 |
| `make_owner_subclass_instance` | exact型の検査が拒否するべき、ownerの派生型の値を作る（乱数生成器は派生型を新しく作り、ほかは既存`make_subclass_copy`を使う）。revision2で追加。 |
| test_pending_observation_and_alarm_record_stores.py | 新test module。2つのownerの単独の動作。 |
| `make_pending_observation` / `make_store_with_pending_observations` / `TupleSubclass` | 指定の位置の保留標本／それらを持つ保留標本のownerを作る／exact tupleの検査が拒否するべきtupleの派生型。 |
| `PROCESSED_SAMPLE_COUNT` / `PROCESSED_ALARM_COUNTS`（fresh_process_smoke.py） | 共用scriptで、各流れの最後に標本1件の処理を呼ぶ回数／流れごとの、その中で起きた警報の回数。 |

派生型の値は、保持と進行のtestの既存helper `make_subclass_copy`をimportして使う（同じ役割）。
