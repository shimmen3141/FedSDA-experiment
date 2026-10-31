# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_IFCA_WINDOW_CONFIGURATION_NAME`（値`feddrift-ifca-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、$K$ 個のクラスタのモデルから損失が最小のものを時刻ごとに選び、今の時刻の標本だけで学習するIFCA+Window | Oracle（`feddrift-oracle-concept-assignment`）は真のコンセプトで割り当てる。FedDrift（`feddrift`）はモデルを作成・統合する |
| 割り当ての方式の設定の新しい値（例: `minimum_loss_cluster_assignment`）と、学習の窓の設定の項目（実装担当が既存の名前に合わせて決め、ここへ追記する） | FedDriftの設定 | 損失が最小のクラスタのモデルへの割り当てと、学習の標本を今の時刻に限る窓 | — |
