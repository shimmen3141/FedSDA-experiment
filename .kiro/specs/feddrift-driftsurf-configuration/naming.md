# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_DRIFTSURF_CONFIGURATION_NAME`（値`feddrift-driftsurf`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、サーバが役割`pred`・`stab`・`reac`の大域モデルを、全clientの正解率から公式のDriftSurfの状態遷移で切り替える構成 | FedDrift（`feddrift`）はclientごとに損失の増加でモデルを作成・統合する。Adaptive-FedAvgは単一のモデルで学習率を変える |
| `--driftsurf-accuracy-drop-thresholds` | 掃引のオプション | DriftSurfのδの値を並べる | `--feddrift-loss-increase-thresholds`はFedDriftのδ（損失の増加）。DriftSurfのδは正解率の低下 |
