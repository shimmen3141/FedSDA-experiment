# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_CFL_WINDOW_CONFIGURATION_NAME`（値`feddrift-cfl-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、1つのクラスタから始め、集約のたびにCFLの規則でクラスタを2分割し、各clientは割り当てたモデルを今の時刻の標本だけで学習するCFL+Window | IFCA+Window（`feddrift-ifca-window`）は $K$ 個のモデルから時刻ごとに損失が最小のものを選ぶ。FedDrift（`feddrift`）は損失の増加でモデルを作り、統合する |
| `cfl_bipartition_assignment` | `FeddriftRunClientSettings.model_assignment_policy` | サーバのCFL分割時だけ割り当てを変える | 最小損失を時刻ごとに選ぶIFCAとは異なる |
| `cfl_split_gamma`（既定0.1） | `FeddriftRunClientSettings`、`experiment_spec/options.py`・`parameters.py` | 分割の角度条件の閾値。CFLだけで有効、掃引軸なし | FedSDAのモデル距離閾値とは異なる |
| `cfl_cluster_splitting` | `methods/feddrift` | CFLの判定と2分割の計算 | runtimeの配布や登録を行わない |
| `CflClusterSplittingState` | `methods/feddrift/cfl_cluster_splitting.py` | run全体の観測最大ノルム・閾値・次IDを保持 | モデルのパラメータの所有者はサーバ |
| `check` | `CflClusterSplittingState` | クラスタID昇順に判定し、判定前後の状態・client群を返す | 配布や登録を行わない |
| `bipartition_complete_linkage` | `methods/feddrift/cfl_cluster_splitting.py` | 完全連結の近隣鎖とsklearnの木の切り方で2群のラベルを返す | 損失距離で統合する既存クラスタリングとは異なる |
| `cfl_training_round` | `runtime` | CFL判定と既存のモデル配布の接続 | 判定計算の部品とは異なる |
| `split_cfl_training_round` | `runtime/cfl_training_round.py` | 学習後パラメータから更新を作り判定へ渡し、分割時のθ0登録と全モデルの集約省略を接続 | 判定計算は状態部品へ委ねる |
| `cfl_assignment` | 記録表 | `timestep_index`・`client_id`・`first_sample_index`・`model_id`。時刻開始の割り当ての写し | `cluster_selection`の損失による選択ではない |
| `cfl_split_check` | 記録表 | `timestep_index`・`model_id`・`new_model_id`・`mean_norm`・`max_norm`・`alpha_cross`・`cfl_norm_before/after`・`eps1_before/after`・`eps2_before/after`・`checked_bipartition`・`did_split`・`client_ids`・`cl1_client_ids`・`cl2_client_ids` | 集約ごと・クラスタごとの分割判定。client ID列はカンマ区切り、未判定alphaはNaN、新IDなしは−1 |
