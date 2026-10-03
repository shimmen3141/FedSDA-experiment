# 命名と役割

revision: 1

保留位置を管理する部品で、モデルへの帰属を実行しない。sample_indexはclient内のglobal観測順、spanは標本件数、change intervalは検出器から推定された区間で真のconceptを意味しない。

| 名前 | 役割・入出力・状態 |
|---|---|
| pending_training_assignment_buffer.py | 帰属確定前の標本位置FIFOと同所有の結果型 |
| PendingTrainingAssignmentBuffer | 設定+位置を入力しFIFO/last位置を所有 |
| PendingTrainingAssignmentState | pending_sample_indices tuple、last_observed_sample_index int又はNoneのfrozen診断copy |
| BufferedChangeIntervalPartition | earlier_sample_indices tuple、change_interval_sample_indices tuple、change_interval_start_sample_index int又はNoneのfrozen分割 |
| append_observed_sample_index | 明示sample_indexを末尾追加、状態更新、返却None |
| release_sample_indices_exceeding_capacity | 超過分だけ最古順に返す、dequeのみ更新 |
| get_change_interval_partition | 正span→変更不能な分割結果、状態不変 |
| drain_pending_sample_indices | 全件を順に返しdequeだけ空、last継続 |
| get_state_snapshot | 独立copyの診断state、状態不変 |
| training_data_assignment_settings / _training_data_assignment_settings | 既存immutable容量条件、設定が唯一所有者 |
| sample_index / _last_observed_sample_index | 今回位置 / 最終追加位置、nonnegative int、未観測None |
| estimated_change_span_sample_count | 推定区間件数、int>=1 |
| _pending_sample_indices | 非公開deque、同実体だけが更新 |
| released_sample_indices / pending_sample_indices | 今回解放位置の局所list / snapshot tuple |
| change_interval_sample_count / partition_split_index | clip後件数 / 前区間長 |
| earlier_sample_indices / change_interval_sample_indices / change_interval_start_sample_index | 分割した順序tupleと先頭位置 |
| _validate_nonnegative_sample_index | sample_index型/非負検査、変換なし |
| state_snapshot / state_before_call / inputs_before_call / result | テスト・診断のcopy/期待結果 |
| buffer / reference_buffer / legacy_client / legacy_events / legacy_sample_indices | 新buffer / 独立実体 / 旧stub / 割当/イベント記録 / 旧位置tuple |
| sample_count / capacity / observation_index / invalid_value / field_name / exception_info / operation | 件数/容量/反復/拒否/署名のtest値 |
| global_python_random_state / global_numpy_random_state / global_torch_random_state | testでだけ保持する共有乱数状態 |
| monitor / observation / loss_change_detection_settings | public監視接続のtest値、productionへ所有しない |

## 検証名

test_pending_training_assignment_buffer.py、接頭辞test_pending_assignment_:
append_and_release_match_legacy_processing / invalid_inputs_preserve_state / partition_matches_legacy_positions / legacy_alarm_branches_preserve_consumption / copies_instances_and_random_states_are_independent / monitoring_span_connects_to_buffer_partition / public_functions_require_explicit_keyword_arguments。
helper: make_assignment_buffer、make_legacy_processing_client、make_legacy_alarm_client。test専用x/y/concept_id、old_model_id、model_id、model、idx、monkeypatch、legacy_span、pending_validation、minimum_drift_sample_count。旧callback/API/既存AST名はtest-only参照として使用する。

