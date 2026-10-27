# Naming

| 名前 | 意味 |
|---|---|
| `runtime/alarm_interval_candidate_training.py` | 警報の変化区間で、独立候補の生成・学習とその前の入力検査を組み立てる。候補検証sessionの開始・参照固定は扱わない |
| `TrainedAlarmIntervalCandidate` | 変化区間で学習した独立候補とepoch学習の結果。既存と同じ役割のまま新しいモジュールへ移す |
| `train_candidate_on_alarm_change_interval` | 検査→候補生成→epoch学習の順で候補を返す。既存と同じ役割・引数・処理順で移す |
| `EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION` | goldenの数値照合から外す、候補検証ありの方針のcaseの版別一覧。保存済みJSONのcase一覧自体は変えない |
| `REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS` | goldenの照合で除く項目の、snapshotの根からの完全な鍵の道。task 1では空 |
| `without_removed_candidate_validation_items` | 明示した鍵の道だけを除いた比較用copyを返す。保存済み設定の照合にも同じ一覧を使う |
| `behavior_random_state_cases.py` | 製品コードを変更せず、runの乱数源と終端の地点を観測するtest helper |
| `RANDOM_STATE_CASE_NAMES` | 版14から候補検証ありの4 caseとFedDrift・FedAvgを除いたFedSDAの14 case |
| `RANDOM_STATE_FIXTURE_PATH` | task 1開始時のHEADで採取した、更新しない乱数状態fixtureの保存先 |
| `capture_run_end_random_states` | 全client・通信の終端処理後、torch状態の復元前に、借用Python・NumPyとtorch CPUの状態全体を採取する |
| `generate_random_state_fixture` | 開始時commitとsrc無変更を検査し、既存fixtureを上書きせずに生成する。移設後は再生成を拒否する |
