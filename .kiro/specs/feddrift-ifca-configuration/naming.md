# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_IFCA_WINDOW_CONFIGURATION_NAME`（値`feddrift-ifca-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、$K$ 個のクラスタのモデル（全部を乱数で初期化）から損失が最小のものを時刻ごとに選び、今の時刻の標本だけで学習するIFCA+Window（初期化は元の論文に準じる） | Oracle（`feddrift-oracle-concept-assignment`）は真のコンセプトで割り当てる。FedDrift（`feddrift`）はモデルを作成・統合する |
| 割り当ての方式の設定の新しい値`minimum_loss_cluster_assignment`、学習の窓`training_sample_window`（既定`all_timesteps`、新構成`current_timestep`）、初期化`cluster_model_initialization`（既定`all_pretrained`、新構成`all_random`／`pretrained_cluster0`）、選択表`cluster_selection` | `FeddriftRunClientSettings`・`runtime/feddrift_run_record_tables` | 最初はID 0、以後は最小平均有界損失へ割り当てる。選択表の列は`client_id`、`timestep_index`、`first_sample_index`、`last_sample_index`、`selected_model_id`、`model_ids`、`model_mean_bounded_losses`（後二者は行×K。モデルIDは昇順、最初の損失は未比較のNaN） | Oracleは真の概念で標本ごとに帰属。検出バッチの遷移は`detection_batch`で別に記録する |
| `ModelTrainingSampleStore.clear_model_training_samples` | `learning/training/model_training_sample_store` | 全モデルの学習標本を空にし、現在時刻の窓へ進める。取得済みsnapshotの構造を変更しない | IDの付替えや標本の連結を行う`remap_model_training_sample_collections`とは異なる |
| `FEDDRIFT_IFCA_WINDOW_PRETRAINED_CLUSTER0_CONFIGURATION_NAME`（値`feddrift-ifca-window-pretrained-cluster0`） | `sweep/experiment_configurations`の定数 | IFCA+Windowの派生版。ID 0だけを共通の事前学習の初期モデルにし、ほかを乱数で初期化する | `feddrift-ifca-window`は全部を乱数で初期化する（論文に準じる） |
