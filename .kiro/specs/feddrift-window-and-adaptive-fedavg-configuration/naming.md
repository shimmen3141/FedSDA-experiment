# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `FEDDRIFT_WINDOW_CONFIGURATION_NAME`（値`feddrift-window`） | `sweep/experiment_configurations`の定数 | FedDriftの学習・集約の上で、単一のモデルを今の時刻の標本だけで、`feddrift`と同じoptimizer・学習率で学習するWindow | Adaptive-FedAvgは学習をSGDにし学習率を調整する。IFCA（`feddrift-ifca-window`）は $K$ 個のモデルから選ぶ |
| `FEDDRIFT_ADAPTIVE_FEDAVG_WINDOW_CONFIGURATION_NAME`（値`feddrift-adaptive-fedavg-window`） | `sweep/experiment_configurations`の定数 | Windowの上で、局所の学習をSGDにし、サーバが集約の後に公式の`AdaState`の式で次の時刻の学習率を決めるAdaptive-FedAvg | Windowは学習率を変えない。FedAvgの構成は窓を持たず学習率を変えない |
| `--adaptive-fedavg-initial-learning-rates` | 掃引のオプション | Adaptive-FedAvgの初期の学習率 $\eta_0$ の値を並べる | `--feddrift-loss-increase-thresholds`はFedDriftのδ |

| `adaptive_fedavg_learning_rate` | `methods/feddrift`のmodule | 公式AdaStateの学習率の推定と固定設定をruntimeから独立して持つ | runtimeはTensorの集約とoptimizerへの受渡しを担う |
| `AdaptiveFedavgLearningRateSettings` | 同module | 初期学習率と公式の固定係数を宣言する | optimizerの設定自体は既存の`SgdParameterOptimizerSettings` |
| `AdaptiveFedavgLearningRateRecord` | 同module | 集約1回の時刻・回数・分散比・補正済み比・次時刻の学習率を記録する | 学習の直前に使った学習率ではない |
| `AdaptiveFedavgLearningRateState` | 同module | サーバが所有する移動平均の状態 | clientのoptimizer stateではない |
| `AdaptiveFedavgLearningRateState.update` | 同module | 集約した全パラメータのベクトルで推定を1回進める | 局所のSGD更新を行わない |
| `AdaptiveFedavgLearningRateState.snapshot_records` | 同module | 検査した不変の記録列を返す | 推定を進めない |

実装で使った設定の名前は`model_assignment_policy=single_global_model_assignment`、
`training_sample_window=current_timestep`、`cluster_model_initialization=all_pretrained`。
前二者だけがWindowで変わる。Adaptive-FedAvgは追加で`adaptive_learning_rate_settings`を持つ。
その既定はNone（固定学習率）、有効にしたときの`initial_learning_rate`の既定は0.01、
`beta1`・`beta2`・`beta3`は0.5に固定。既定値は保存設定から省く。
旧schema上の名前は`feddrift_adaptive_learning_rate`（`fixed`／`adaptive_fedavg`）、
`adaptive_fedavg_initial_learning_rate`、`adaptive_fedavg_beta`。
新APIの数値宣言は`ParameterSpec.configuration_surface=typed_settings`で実際の型のfieldへ対応付ける。
掃引計画の軸は`adaptive_fedavg_initial_learning_rates`。
記録の表は`adaptive_learning_rate`、列は`timestep_index`、`aggregation_count`、`variance_ratio`、
`smoothed_ratio`、`learning_rate_for_next_timestep`（意味はdesignの式と同じ）。
