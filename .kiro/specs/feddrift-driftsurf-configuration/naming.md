# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_DRIFTSURF_CONFIGURATION_NAME`（値`feddrift-driftsurf`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、サーバが役割`pred`・`stab`・`reac`の大域モデルを、全clientの正解率から公式のDriftSurfの状態遷移で切り替える構成 | FedDrift（`feddrift`）はclientごとに損失の増加でモデルを作成・統合する。Adaptive-FedAvgは単一のモデルで学習率を変える |
| `--driftsurf-accuracy-drop-thresholds` | 掃引のオプション | DriftSurfのδの値を並べる | `--feddrift-loss-increase-thresholds`はFedDriftのδ（損失の増加）。DriftSurfのδは正解率の低下 |

| `driftsurf_state` / `DriftSurfSettings` / `DriftSurfState` / `DriftSurfTransition` / `advance` / `scoring_roles` | `methods/feddrift/` | 正解率から役割・窓を決め、作り直しと昇格を返す独立部品。`advance`は時刻1からの遷移、`scoring_roles`は必要な評価役割 | runtimeのモデル実体・IDを所有しない |
| `TimestepTrainingSampleStore` / `append_timestep` / `samples_for_window` / `timestep_indices` | `learning/training/timestep_training_sample_store` | 直近10時刻の標本を保管し、指定窓を時刻昇順に連結 | `ModelTrainingSampleStore`は学習直前のモデル別集合 |
| `prepare_driftsurf_timestep` / `count_driftsurf_role_correct_predictions` | `runtime/driftsurf_timestep_preparation` | clientの正解数の合計、役割の登録・除去、窓とstoreの接続 | 状態遷移の判断は独立部品に委ねる |
| `driftsurf_settings` / `accuracy_drop_threshold` | `FeddriftRunClientSettings` / `DriftSurfSettings` | 既定None / 0.10。反応3時刻・窓10時刻は固定 | 既定値は保存設定から省き、既存runの識別子を維持 |
| `driftsurf_global_role_assignment` / `role_timestep_window` | `model_assignment_policy` / `training_sample_window`の値 | 共通の予測役割と役割ごとの時刻窓 | clientごとの最小損失選択や窓1時刻とは異なる |
| `driftsurf_accuracy_drop_thresholds` | `ExperimentSweepPlan` | DriftSurfのδだけを増やす掃引軸。既定None（δ0.10） | FedDriftの損失増分とは別の閾値 |
| `driftsurf` | 保存形式版6の表 | 時刻、遷移前後の状態、役割の正解率・最高値、学習役割・予測役割、反応回数、反応への進入・昇格、役割のIDと窓の時刻を記録 | 未計測正解率NaN、未登録ID/反応回数−1、窓はカンマ区切り |
