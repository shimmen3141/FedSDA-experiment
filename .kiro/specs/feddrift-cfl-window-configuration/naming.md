# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_CFL_WINDOW_CONFIGURATION_NAME`（値`feddrift-cfl-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、1つのクラスタから始め、集約のたびにCFLの規則でクラスタを2分割し、各clientは割り当てたモデルを今の時刻の標本だけで学習するCFL+Window | IFCA+Window（`feddrift-ifca-window`）は $K$ 個のモデルから時刻ごとに損失が最小のものを選ぶ。FedDrift（`feddrift`）は損失の増加でモデルを作り、統合する |
| 割り当ての方式の新しい値、$\gamma$ の設定の項目、判定と割り当ての記録の表 | 実装担当が決めて追記する | | |
