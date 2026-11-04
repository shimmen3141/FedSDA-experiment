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


## Task 1・2で使った名前

| 名前 | 場所 | 役割・区別 |
|---|---|---|
| `FEDCCFA_CONFIGURATION_NAMES` | `sweep/experiment_configurations.py` | 4構成を承認済みの順で束ねる。既存31構成の末尾へ追加 |
| `FEDDRIFT_FEDCCFA_WINDOW_CONFIGURATION_NAME` | 同上 | I標本、通算20 roundから整合 |
| `FEDDRIFT_FEDCCFA_WINDOW_SCALED_ALIGNMENT_CONFIGURATION_NAME` | 同上 | I標本、全roundの10%から整合 |
| `FEDDRIFT_FEDCCFA_WINDOW200_CONFIGURATION_NAME` | 同上 | 直近max(I,200)標本、通算20 roundから整合 |
| `FEDDRIFT_FEDCCFA_WINDOW200_SCALED_ALIGNMENT_CONFIGURATION_NAME` | 同上 | 直近max(I,200)標本、全roundの10%から整合 |
| `fedccfa_classifier_clustering` | `runtime`のmodule・割当方式 | モデルを分けず、診断用分類層のクラスの行からclientを分ける |
| `FedccfaClientState` | 同module | clientごとの窓・個人の分類層・診断用分類層・アンカーのowner。classifierは保有registryと同じ参照 |
| `FedccfaClientState.predict` | 同module | ラベル観測前の個人のモデルによる予測 |
| `FedccfaClientState.append_timestep_samples` | 同module | 予測後の観測済み窓の追加と古い標本の除去 |
| `FedccfaClientState.train_round` | 同module | 窓の分布、診断、個人の分類層、表現部、局所アンカーまで。集約は行わない |
| `fedccfa_training_round` | `runtime`のmodule | サーバの集約・配布・部品別通信の接続 |
| `aggregate_fedccfa_training_round` | 同module | 表現部の件数重み平均とクラスの組ごとの行・アンカーの等重み平均、配布と記録 |
| `fedccfa_dbscan_eps` | client設定・sweep plan・ParameterSpec | クラスの組のDBSCANのε。計画ではfloatのtuple、client設定ではfloat |
| `fedccfa_alignment_gamma` | client設定・ParameterSpec | H/γの分母（既定20、正値のみ）。CFLの分割γと別 |
| `fedccfa_alignment_temperature` | client設定・ParameterSpec | 対照の整合のcosロジットの温度（既定0.1） |
| `fedccfa_alignment_start` | client設定・OptionSpec | `fixed_round_20`／`scaled_to_run_rounds`。通算round番号の閾値 |
| `recent_observed_samples` | 学習窓の選択肢 | 観測済み直近max(I,200)標本。FedEMの全履歴と異なる |
| `FEDCCFA_DBSCAN_EPS_AXIS` | `experiment_spec/sweep.py` | 新API専用のSweepAxis宣言。旧SweepPlanで実行しない |
| `fedccfa_label_cluster` | 記録表 | `timestep_index, round_index, client_id, class_label, cluster_index, window_class_sample_count`。単独・欠けたクラスのcluster_indexは−1 |
| `fedccfa_round` | 記録表 | `timestep_index, round_index, client_id, alignment_active, alignment_weight, window_sample_count, window_missing_class_count` |
| `fedccfa_communication` | 記録表 | `timestep_index, round_index, component, direction, parameter_value_count, byte_count`。初回配布は時刻・roundとも−1。部品はrepresentation／initial_classifier／diagnostic_classifier／personal_classifier／personal_classifier_row／anchor |
| `binary_softmax_output` | 分類器の内部状態 | FedCCFAの変換後だけtrue。2出力のsoftmaxのクラス1確率を共通の二値予測APIへ返す |

要求に従い内部関数も記録する（通常の公開名の確認範囲に追加）:
`_convert_binary_classifier`（初期値の2出力への変換）、`_madd`（クラスの行のMADD距離）、`_label_groups`（DBSCANの2 client以上の組）、`_weighted_mean`（入力を変更しない平均）、`_entropy_weight`（存在するクラスのH/γ）、`_balanced_indices`（存在するクラスごとの均衡抽出）、`_alignment_start_round`（通算の閾値）、`_alignment_loss`（アンカーがある列と標本の対照CE）。

`train_round`の`official_steps`・`balanced_indices`は公式照合の専用入口。評価runの型付き設定・構成・CLIからは指定できず、通常runは必ず共通Adam予算を使う。段の回数は（診断、個人、表現）の順のtupleで、照合では(5,1,5)、SGDの固定条件を使う。
