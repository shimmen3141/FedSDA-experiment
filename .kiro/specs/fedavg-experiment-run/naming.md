# Naming Review

正式手法名は`fedavg`。旧Obliviousはtestの対応説明だけで、新API aliasは作らない。型の単語規則は既存Feddriftと揃える。共通executorの公開操作名は既存契約を同じ役割で再利用する。

| 公開名 | 役割・既存名との違い |
| --- | --- |
| `fedavg_client` | 単一モデルclientの学習・記録 |
| `FedavgSamplePrediction` | 学習前の単一モデル予測記録 |
| `FedavgClient` | 全到着標本と単一モデル/optimizerを所有 |
| `process_observed_sample` | 標本の逐次処理、共通executor契約 |
| `flush_pending_local_updates` | 区間末の保留予算消化 |
| `has_model_ready_for_server_registration` | 新モデルを作らないため常にfalse |
| `advance_new_model_upload_wait_after_synchronization` | FedAvgには待機なし |
| `finalize_incomplete_candidate_validation` | FedAvgには候補なし |
| `apply_global_parameters` | 新モデル生成とoptimizer初期化による配布適用 |
| `fedavg_run_settings` | FedAvgの完全設定 |
| `FedavgRunParticipantSettings` | モデル準備とclient学習の設定 |
| `FedavgRunSettings` | 実行/参加者/評価の完全設定 |
| `build_default_fedavg_run_settings` | 旧baseline条件から既定の完全設定 |
| `fedavg_run_server` | 単一global modelの同期 |
| `FedavgRunServer` | FedAvgと通信/集約計数のowner |
| `record_client_states_before_synchronization` | 軽量通知なし、同期前checkpoint |
| `synchronize_models` | 件数重み集約と全client配布 |
| `finalize_started_communications` | FedAvg終端に追加処理なし |
| `fedavg_run_participant_factory` | 初期準備 |
| `FedavgRunParticipantFactory` | 事前学習と参加者組立て |
| `validate_configuration` | 乱数消費前の設定検査 |
| `prepare_run` | 事前学習後のclient/server生成 |
| `prepared_run_participants` | 直前の成功済み参照 |
| `fedavg_experiment_run` | 全体run・計測・評価接続 |
| `FedavgRoundComputationCounts` | 同期区間ごとのモデル計算 |
| `FedavgExperimentRun` | 完全条件/実run/参加者/準備・本計数/指標 |
| `FedavgRunMetrics` | 共通基本指標、存在しない候補等の指標は含めない |
| `execute_fedavg_experiment_run` | 完全設定から計測つき実run |
| `derive_fedavg_run_metrics` | 不変記録から共通指標を導出 |
| `fedavg_run_record_tables` | 予測/同期/診断の配列変換 |
| `build_fedavg_run_record_tables` | 対応検査後の記録表生成 |
| `fedavg_run_artifact_files` | FedAvg固有formatの保存/読込 |
| `FedavgRunArtifactPaths` | JSON/NPZ保存先 |
| `save_fedavg_experiment_run` | 既存上書きなしの2ファイル保存 |
| `load_fedavg_run_artifacts` | FedAvg識別と安全NPZ読込 |
