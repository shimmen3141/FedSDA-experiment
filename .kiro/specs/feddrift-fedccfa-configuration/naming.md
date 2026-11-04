# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `feddrift-fedccfa-window` | 構成 | FedCCFA、学習の窓はその時刻の $I$ 標本、整合は通算20 round以降 | FedEMの`-window`は全成分を混ぜる |
| `feddrift-fedccfa-window-scaled-alignment` | 構成 | 上と同じで、整合は全roundの10%以降 | |
| `feddrift-fedccfa-window200` | 構成 | 学習の窓は直近 $\max(I,200)$ 標本、整合は通算20 round以降 | FedEMの`-history`は全履歴 |
| `feddrift-fedccfa-window200-scaled-alignment` | 構成 | 上と同じで、整合は全roundの10%以降 | |
| `fedccfa_classifier_clustering` | 割り当ての方式の値（案） | 表現部を共有し、個人の分類層をクラスごとのクラスタで平均する | `fedem_responsibility_assignment`は全成分の混合 |
| `--fedccfa-dbscan-eps` | 掃引の軸 | DBSCANの $\epsilon$（既定0.1） | DriftSurfの閾値・FedDriftのδとは別 |
| `fedccfa_label_cluster` | 記録表 | 時刻・round・クラスごとのclientの組 | IFCAの`cluster_selection`はclientごとの1モデルの選択 |
| `fedccfa_round` | 記録表 | 時刻・round・clientごとの整合の有無と重み、窓の件数と欠けたクラスの数 | |

実装担当が決めた設定の項目・表・関数は、実装の後にこの表へ追記する。
