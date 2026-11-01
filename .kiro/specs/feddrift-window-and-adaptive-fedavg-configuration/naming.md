# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_WINDOW_CONFIGURATION_NAME`（値`feddrift-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、単一のモデルを今の時刻の標本だけで、`feddrift`と同じoptimizer・学習率で学習するWindow | Adaptive-FedAvgは学習をSGDにし学習率を調整する。IFCA（`feddrift-ifca-window`）は $K$ 個のモデルから選ぶ |
| `FEDDRIFT_ADAPTIVE_FEDAVG_WINDOW_CONFIGURATION_NAME`（値`feddrift-adaptive-fedavg-window`） | `sweep/experiment_configurations`の定数 | Windowの上で、局所の学習をSGDにし、サーバが集約の後に公式の`AdaState`の式で次の時刻の学習率を決めるAdaptive-FedAvg | Windowは学習率を変えない。FedAvgの構成は窓を持たず学習率を変えない |
| `--adaptive-fedavg-initial-learning-rates` | 掃引のオプション | Adaptive-FedAvgの初期の学習率 $\eta_0$ の値を並べる | `--feddrift-loss-increase-thresholds`はFedDriftのδ |
