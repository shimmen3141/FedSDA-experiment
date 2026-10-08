# 警報応答の完了処理 — 調査記録

2026-10-08、主担当Claude Code。基点commit `3f1202c`。固定旧基準`748c3aa`。

## 旧処理の事実（`federated_drift_experiment/clients/fedsda.py`）

- `_resolve_drift`の4つの終了経路（候補検証中、不足、再利用・維持、新規作成）は、いずれも`_record_adaptation_event`→`_reset_drift_detectors`の順に実行する。`buffer.clear()`は不足の経路だけ実行しない（LEGACY-002として記録済み）。
- 再利用では、`reuse_selection_counts["alternative_fit"]`の加算→`local_switch_positions.append`→`_set_local_current_model`（内部で`_on_local_model_change`）→変化区間の吸収→イベント記録→reset→clearの順。維持では`current_fit`の加算→吸収→イベント→reset→clear。
- 最終構成の検出器は`ClassConditionalESRFedSDAClient`。`_reset_drift_detectors`は`e_detector.reset(_e_detector_baseline())`の後、`class_e_detectors`と`class_e_positions`をclearし、`_class_drift_start`をNoneにする。`_e_detector_baseline`は`model_stats[current_model_id]`が無いか`n < 1`なら0.01、それ以外は平均を0.01以上1-1e-6以下へ丸める。resetの時点で`current_model_id`は切替後、統計は吸収後。
- 新`OverallAndTrueClassLossMonitor.reset`と`select_loss_monitoring_baseline_mean_loss`は、それぞれ完了済みspecで実旧と照合済み（`tests/refactoring/test_loss_change_monitoring.py`、`test_loss_baseline_selection.py`）。本specは両者を警報応答の後に正しい時点の統計でつなぐことを照合する。
- 戻り値は再利用だけ1、その他は0（最終構成では新規作成が候補検証の開始になるため2を返さない）。呼出元`process_one_step`は1または2のとき`detection_episodes.mark_operation()`を呼ぶ。`FEDSDA_DETECTION_EPISODES_ENABLED`の既定はFalse。
- `_on_local_model_change`は最終構成（Restarting mixin）でAdaHedge系の`restart_for_concept`を呼ぶ。新実装の`FixedSharePredictionWeightController`との接続は後続specで扱う。
- 警報標本は`process_one_step`で`buffer.append`された後に警報処理へ入るので、警報位置はFIFOの最終観測位置と等しい。

## oracleの実行可能性（REDより前に確認）

既存`build_buffer_response_oracle`の実旧client（`__new__`で作った`SharedBackboneClassConditionalESRFedSDAClient`）へ、`e_detector`（実`BoundedMeanEDetector`）、`overall_component_weight`、`class_component_weight`、`class_e_detectors`、`class_e_positions`、`_class_drift_start`、`history_detector_log_e`、`compute_counters`、`adaptation_events`、`local_switch_positions`、`reuse_selection_counts`を与え、上流oracleがinstance属性で差し替えた`_record_adaptation_event`と`_reset_drift_detectors`を外すと、実旧`_resolve_drift`がイベント記録・reset・clearを含めて実行できることを確認した。`_update_drift_detectors`と`_estimated_new_concept_span`（classの実メソッド）も同じclientで実行できる。推定区間長だけはinstance属性で供給する（上流oracleの区間設計を保つため）。

上流oracleの解決条件（再利用・維持・適合なし）がどの応答結果になるかは、乱数seedと前区間の有無で変わる。seed 823では、2/4class×3解決条件×4組の24条件が不足・再利用・維持・候補検証開始の4結果をすべて含み、前区間0件では解決条件どおりの結果になることを確認した。単一条件のtestは前区間0件・seed 823を使い、期待する応答結果をtest内でassertする。

## 手順上の事実

命名を事前登録するため、runtimeとtestをリポジトリ外で下書きし、リポジトリ外で一度実行して上記の実行可能性とoracleの条件を確かめた（51件が成功）。worktreeには新src・新testをまだ置いていない。実装taskでは、testを先にworktreeへ追加してREDを記録してからsrcを追加する。

## 判断

- 応答の呼出しを本処理に含めない。含めると22引数の応答の署名を再掲することになり、完了処理の検査対象（監視・保留位置）と混ざる。呼出側（後続のclient進行）が応答→完了の順に呼ぶ。応答と入力の構造上の対応は本処理が検査する。
- 検出器名は扱わない。既存の候補検証の完了情報と同じく、呼出側が持つ。
- 再利用計数の種別はrecordへ持たせない。応答結果から一意に決まるので、名前を増やさず設計の対応表で示す。
- イベント一覧などのownerは、候補検証の到達時・未完了回収・警報応答の3つの完了情報を受ける形で後続specが設計する。
