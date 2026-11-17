# 命名と役割

名前は略語`dwm`を避け、`dynamic_weighted_majority`と書く（他の概念との混同を避けるため）。構成名だけは、論文と比較の表で通じる短い`dwm-neural-network`にする。

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `dwm-neural-network` | 構成 | `fedavg`と同じ学習器を専門家にしたDWM（Kolter & Maloof 2007の図1）。clientごとに独立、集約しない | `fedavg`は専門家1つでドリフトへの対応なし。Riverの研究ツールの`dwm`は木を専門家にした研究用の実装で、製品の構成ではない |
| `DYNAMIC_WEIGHTED_MAJORITY_NEURAL_NETWORK_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 | 既存の構成名の定数と同じ扱い。一覧の末尾に足す |
| `methods/dynamic_weighted_majority` | package | DWMの手法の部品 | `methods/fedavg`（単一モデル）とは別。FedAvgのclientを継承しない |
| `dynamic_weighted_majority_rules` | module | 投票と周期ごとの重みの更新・削除・追加の判断の純粋な関数 | 学習・乱数を持たない |
| `vote_weighted_majority` | 同上 | 専門家の予測クラスへの重み付き投票。同点は最小のクラス | FedHASPのFixed-Shareの混合（確率の重み付き平均）ではない |
| `update_weights_at_period` | 同上 | 図1の8〜9行目と12〜19行目（減衰→投票し直し→正規化→θ未満の削除→追加の判断）を一度に行う | 評価の予測（学習の前の重みの投票）は求めない |
| `DynamicWeightedMajorityPeriodUpdate` | 同上 | 上の戻り値（減衰・正規化の後の重み、残す専門家の位置、投票し直した予測、追加するか） | 状態を持たない記録 |
| `dynamic_weighted_majority_client` | module | DWMのclient | — |
| `DynamicWeightedMajorityClient` | 同上 | 専門家・重み・保持標本・記録を所有し、標本ごとの予測と学習を行う | `FedavgClient`は専門家1つで、集約の配布を受ける |
| `DynamicWeightedMajorityExpert` | 同上 | 専門家1つの状態（ID、分類器、optimizerの状態、保持標本、重み、学習の計数）。clientが`experts`に作られた順で持つ | FedHASPの保有モデルとは別（共有部・adapterを持たない） |
| `DynamicWeightedMajoritySamplePrediction` | 同上 | 標本ごとの評価の予測の記録（予測の時点の専門家の数を含む） | `FedavgSamplePrediction`はモデルIDを持ち、専門家の数を持たない |
| `DynamicWeightedMajorityExpertLifetime` | 同上 | 専門家ごとの作られた標本の位置と消された標本の位置 | — |
| `dynamic_weighted_majority_run_settings` / `DynamicWeightedMajorityRunParticipantSettings` / `DynamicWeightedMajorityRunSettings` / `build_default_dynamic_weighted_majority_run_settings` | `runtime/` | DWMの参加者の設定、runの完全な設定、既定の組立て（`fedavg`の値を取り出し、β・θ・pを足す） | FedAvgの設定の型とは別。FedAvgの値を変えない |
| `dynamic_weighted_majority_run_server` / `DynamicWeightedMajorityRunServer` | `runtime/` | 集約・配布しないサーバ。通信量0と同期の回数を記録 | `FedavgRunServer`は集約する |
| `dynamic_weighted_majority_run_participant_factory` / `DynamicWeightedMajorityRunParticipantFactory` | `runtime/` | 事前学習1回とclient・サーバの準備 | — |
| `dynamic_weighted_majority_experiment_run` / `DynamicWeightedMajorityExperimentRun` / `DynamicWeightedMajorityRunMetrics` / `execute_dynamic_weighted_majority_experiment_run` / `derive_dynamic_weighted_majority_run_metrics` | `runtime/` | 計測つきの全体run、その結果、指標とその導出 | 指標の項目はFedAvgと同じ定義 |
| `dynamic_weighted_majority_run_record_tables` / `build_dynamic_weighted_majority_run_record_tables` | `runtime/` | 保存する記録の名前付きの配列 | — |
| `dynamic_weighted_majority_run_artifact_files` / `save_dynamic_weighted_majority_experiment_run` / `load_dynamic_weighted_majority_run_artifacts` | `runtime/` | JSON・NPZの保存と読込み | — |

実装担当が決めた公開名（あれば）は、実装の後にこの表へ追記する。

## 実装で追加した名前

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `DynamicWeightedMajorityExpertTrainingCounts` | `dynamic_weighted_majority_client` | 専門家ごとの学習と予測の計数（消された専門家は消される時点の値）。`model_training`の表の1行 | `DynamicWeightedMajorityExpertLifetime`は作られた・消された位置だけを持つ |
| `expert_lifetimes` / `expert_training_counts` | `DynamicWeightedMajorityClient`のproperty | 作られた順の全専門家（消されたものを含む）の寿命の記録と、学習の計数 | `experts`は今いる専門家だけ |
| `DynamicWeightedMajorityRoundComputationCounts` | `dynamic_weighted_majority_experiment_run` | ラウンドごとの、標本の処理と同期の間のモデルの計算 | FedAvgの`FedavgRoundComputationCounts`と同じ働き（非公開名をimportしないため、同じ形をこのmoduleに持つ） |
| `DynamicWeightedMajorityRunArtifactPaths` | `dynamic_weighted_majority_run_artifact_files` | 保存した要約と記録の表の場所 | FedAvgの`FedavgRunArtifactPaths`と同じ働き |
| `METHOD_NAME` | 同上 | 要約の`method_name`（`dynamic_weighted_majority`）。読込みはこの値でなければ拒否する | 構成名`dwm-neural-network`とは別（成果物の種類の名前） |
| `DEFAULT_WEIGHT_DECREASE_FACTOR` / `DEFAULT_REMOVAL_WEIGHT_THRESHOLD` / `DEFAULT_PERIOD_SAMPLE_COUNT` | `dynamic_weighted_majority_run_settings` | 既定のβ=0.5、θ=0.01、p=50 | 公式の実装の既定のpは1。50は論文4.2節の値 |
| `synchronized_client_expert_counts` | `DynamicWeightedMajorityRunServer`の属性 | 同期の時点のclientごとの専門家の数（`synchronization.client_held_model_count`の表） | FedAvgのサーバには無い |
| `build_experiment_run_settings` | `sweep/experiment_configurations.py` | 既存の関数。戻り値の型の和にDWMの設定を足し、構成`dwm-neural-network`の分岐をFedSDAの既定の経路より前に足しただけ | 役割は変えない |

