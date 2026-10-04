# 命名と役割
revision: 1

| 名前 | 役割 |
|---|---|
| methods/fedsda/loss_statistics / loss_baseline_selection.py | 手法固有の損失基準値方針。learningの数値集計から区別 |
| select_loss_monitoring_baseline_mean_loss | 監視の基準平均を選択、loss単位、float |
| select_alarm_interval_reuse_baseline_mean_loss | 警報時に保持区間で再利用比較するための平均、floatまたはNone |
| select_post_alarm_reference_historical_mean_loss | 警報後比較に使う参照の履歴平均、0も有効 |
| _validate_optional_loss_moments | 任意集計の型とfieldを検査し独立した検証済みコピーを返す |
| loss_moments / validated_loss_moments | 不変入力 / 検証済み独立コピー、状態変更なし |
| legacy_client / legacy_stats / result / state_before_call / estimate | test旧oracle・戻り値・入力状態・集計推定 |
| model_id / sample_index / field_name / invalid_value / operation / loss_sequence | test対象ID・位置・拒否field・操作・観測列 |
| monitor / observation / reference_historical_mean_losses_by_model_id / reference_losses_by_model_id | 既存公開APIへ明示するtest局所値 |
| global_python_random_state / global_numpy_random_state / global_torch_random_state / global_default_dtype / global_default_device | test共有状態診断 |
| candidate / reference_models / captured_session / evaluated_candidates / valid_candidates / baseline_mean_loss / observed_loss_count / mean_loss / sum_squared_loss_deviations | 旧実装oracleのstub/捕捉結果/入力数値。旧引数名・既存upstream fixture名はtest-onlyで維持 |
| test_loss_baseline_selection.py / test_loss_baseline_ | testfile/接頭辞。末尾matches_legacy_monitor, matches_legacy_reuse, matches_legacy_history, rejects_invalid_input, connects_to_monitor_and_reference, preserves_shared_state, uses_keyword_arguments |

公開関数はモデル名/IDを受けず、一系列の意味と利用時機を名前で区別する。初回/initialという語は初期学習と混同するため避け、alarm_intervalを使う。
