# FedSDAとFedDriftの構成比較

対象は[最終提案のFedSDA](proposed-method.md)（Residual Adapter＋ClassESR＋Switching SoftRouting）と、このリポジトリの`FedDrift`比較実装。過去のADWIN版・Meta-switching版FedSDAは含めない。
前半は手法の動作と構成、後半は既存実験の代表条件における性能を比較する。FedDrift原論文との実験条件の相違は[別資料](../experiments/differences-from-feddrift.md)を参照する。

| 比較項目 | FedSDA: 最終提案構成 | FedDrift: 本リポジトリの比較実装 |
|---|---|---|
| 共通の対象 | クライアントごとに変化・再来する概念に、複数の概念モデルを保持して対応 | 同左。モデルの選択・作成・連合集約・統合で対応 |
| モデル構造 | 共有特徴抽出部＋概念固有のrank 8 Residual Adapterと分類層 | 概念ごとに独立したMLP。共有特徴抽出部・Residual Adapterを用いない |
| ラベル観測前の予測 | 保有モデルの出力をSwitchingの重みで混合。現在標本のラベルで更新する前の重みを使用 | 現行の割当モデル一つで予測 |
| 予測とデータ帰属 | 予測混合の重みと、学習データの帰属先を別に管理 | 選択した現行モデルが予測と検出バッチの帰属先を担う |
| ドリフト監視 | 現行割当モデルの損失を、全体・正解クラス別にClassESRで逐次監視 | 検出バッチごとに全保有モデルを評価し、最小平均損失が前バッチの基準より閾値を超えて増加したかを判定 |
| 既存モデルの選択 | 警報区間の損失をモデル自身の履歴基準と比較し、適合モデル中で区間損失最小のものを再利用 | ドリフトなしの場合、検出バッチの平均損失が最小のモデルを選択 |
| 新規モデルの採用 | 警報後の新着標本で既存モデルの再適合を確認。全既存モデルが不適合なら、候補が最良参照モデルより前半・後半の両方で良い場合に採用 | ドリフト判定時に新規モデルを作成し、同時刻の登録対象にする。FedSDAの前向き採否検証は行わない |
| データの保留 | FIFOで最近の標本の帰属を保留し、検知区間や前向き検証に利用。最終設定はFIFO 30件・前向き10件 | 検出バッファを単位に判定・帰属を確定。FedSDAの前向き検証用FIFOとは役割が異なる |
| ローカル学習 | 概念別標本から共有部・adapter・分類層を共同更新。最終設定は平均勾配 | 保有する独立モデルを、それぞれの帰属標本で学習。新規作成時の即時学習epochは0で、その後のRラウンドで学習 |
| サーバ処理の順序 | 新規ID登録→同ラウンドのFedAvg→必要時のクラスタリング/merge→配布 | 時刻開始時に隔離解除済みモデルのクラスタリングと新規モデル登録。その後にローカル学習・FedAvg・配布をR回実施 |
| 統合候補の判定 | クラス別に「一方だけが正解する割合」の同時信頼下限の最大値からモデル対の相補性を評価し、average linkageで候補を作る。最終設定は新規モデル発生時に実行 | クロス評価から求めた損失増分距離と固定閾値で評価。linkageの既定はcomplete。新規モデルは隔離期間を経て統合対象になる |
| 統合の適用 | 最終構成は通常merge。現ラウンドのFedAvg後parameterを加重平均し、代表IDへ統合 | 既存モデルのキャッシュと重みを用いてmerge構成を適用し、代表IDへ統合 |
| 集約後の予測重み | 配布後のモデルでFIFO標本を再評価し、予測重みの損失情報を再較正 | 混合予測の重みを持たないため、この再較正は行わない |
| 処理・通信の単位 | サンプルごとに予測・監視・適応し、集約間隔Aで同期 | 検出バッチの完了を時刻処理・通信の単位とし、バッチあたりRラウンドを実行 |

## 比較実験で揃える条件

データセット、概念系列、クライアント数、標本数、seedと学習設定を明示する。FedSDAのAとFedDriftの検出バッチ長/Rは異なる処理を制御するため、値が同じだけで予算が等しいとは判断しない。
モデル構造も異なるので、精度に加えて実測の通信量・計算量・回復速度・モデル数を比較する。共有部の共同学習を含む最終FedSDAについて、標本数だけからoptimizer更新数の一致を推定しない。
FedDriftの復元byte値は既定MLP・float32を仮定した参考値であり、実行時前提を確認するまで正式baselineへ採用しない。

## 既存実験から得られる性能比較

### 条件と集計方法
2026-10-06に保存済みCSVから再集計した代表条件の比較。新しい実験は実行していない。

- 共通: random概念系列、6データセット、seed 0–4、各クライアント5000標本。
- FedSDA: 最終Switching構成、集約間隔A=100。
- FedDrift: 検出バッチ長B=50、損失増分閾値δ=0.1。検出バッチ長の掃引系列だけから選び、閾値掃引側に重複する同条件を二重に数えない。
- 全期間・定常精度は各runを等重みにした5 seed平均±標本標準偏差。差は丸め前のFedSDA平均−FedDrift平均（pp: パーセントポイント）。
- AとBは制御する処理が異なる。この組合せは既存の回復比較と同じ代表点で、同一通信予算・同一計算予算の比較や各手法の最適点を意味しない。

### 全期間精度と定常精度
単位は%。定常精度は真のドリフト直後200標本を除く既存の`stable_accuracy`。

| dataset | 全期間 FedSDA | 全期間 FedDrift | 全期間差 (pp) | 定常 FedSDA | 定常 FedDrift |
|---|---:|---:|---:|---:|---:|
| SEA2 | 85.87 ± 0.14 | 86.26 ± 0.09 | -0.39 | 86.02 ± 0.20 | 86.35 ± 0.17 |
| SEA4 | 84.74 ± 0.31 | 84.67 ± 0.34 | +0.07 | 85.06 ± 0.43 | 84.87 ± 0.37 |
| CIRCLE2 | 96.41 ± 0.11 | 97.55 ± 0.05 | -1.14 | 97.26 ± 0.16 | 98.85 ± 0.06 |
| SINE2 | 97.36 ± 0.61 | 94.23 ± 0.36 | +3.13 | 98.30 ± 0.44 | 99.33 ± 0.04 |
| MNIST2 | 92.90 ± 0.08 | 93.47 ± 0.22 | -0.57 | 93.29 ± 0.14 | 94.66 ± 0.17 |
| MNIST4 | 91.91 ± 0.31 | 90.68 ± 1.19 | +1.22 | 92.61 ± 0.24 | 92.20 ± 1.44 |

この代表点では全期間精度はSEA4・SINE2・MNIST4でFedSDAが高く、SEA2・CIRCLE2・MNIST2でFedDriftが高い。
SINE2ではFedSDAの全期間精度が高い一方、定常精度はFedDriftが高い。回復と定常の両方を示す必要がある。seed数は5であり、有意差検定は行っていない。

### ドリフト後の回復区間精度
単位は%。既存`recovery_analysis.py`の回復曲線について、5 seedのドリフトイベントをプールした各Δの精度を、0≤Δ<200で平均した値。seed平均±標準偏差の上表とは集計方法が異なる。

| dataset | FedSDA | FedDrift | 差 (pp) |
|---|---:|---:|---:|
| SEA2 | 85.24 | 85.89 | -0.64 |
| SEA4 | 83.33 | 83.79 | -0.46 |
| CIRCLE2 | 93.03 | 92.42 | +0.61 |
| SINE2 | 93.60 | 74.07 | +19.52 |
| MNIST2 | 91.33 | 88.74 | +2.59 |
| MNIST4 | 88.95 | 84.35 | +4.60 |

SINE2・MNIST2・MNIST4ではこの回復区間の平均精度がFedSDAで高い。SEA2・SEA4ではFedDriftが高い。これだけで全条件における優位性や同一通信量での優位性は主張しない。

### 計算量の指標と終了時モデル数
いずれも5 seed平均。論理モデル標本処理数は`compute_model_examples_total`を100万で割った値で、小さいほど記録上の処理数が少ない。終了時モデル数は`final_model_count`。

| dataset | モデル標本処理数 FedSDA (百万) | FedDrift (百万) | 終了時モデル数 FedSDA | FedDrift |
|---|---:|---:|---:|---:|
| SEA2 | 1.744 | 2.210 | 1.0 | 3.6 |
| SEA4 | 2.118 | 2.224 | 1.4 | 2.6 |
| CIRCLE2 | 4.217 | 3.701 | 3.4 | 4.2 |
| SINE2 | 3.325 | 7.703 | 2.2 | 41.6 |
| MNIST2 | 3.755 | 5.642 | 2.6 | 16.6 |
| MNIST4 | 4.685 | 8.448 | 4.4 | 32.6 |

終了時モデル数は全6データセットでFedSDAが少ない。モデル標本処理数はCIRCLE2ではFedSDAが多く、他の5データセットでは少ない。
モデル構造・共有特徴の再利用が異なるため、この処理数はFLOPs・実時間・消費電力の比ではない。終了時モデル数から途中の最大モデル数や総通信量も推定しない。
FedDriftの通信byteは復元前提が未確認のため、正式baselineへの採用を保留する。以下では前提を付した参考値として示す。

### 通信量: モデルparameterのpayload
上表と同じ代表条件、5 seed平均±標本標準偏差。up＋downの合計、単位はMiB（1 MiB = 2²⁰ bytes）。

- FedSDAはCSVの`comm_bytes_total`記録値。共有parameterの重複送信を除いたモデルtensorのpayloadで、ネットワークを実測した総byte量ではない。
- FedDriftは`comm_models_total × 1モデルのparameter値数 × 4 bytes`の**仮定に基づく復元値**。実行時のモデル構造・dtypeの確認が残っているため、正式baselineへの採用は保留する。
- 両列とも通信ヘッダ・シリアライズ・制御メッセージ・評価統計等の追加byteを含む実測総通信量として扱わない。

| dataset | FedSDA: CSV記録payload (MiB) | FedDrift: 仮定復元payload (MiB) |
|---|---:|---:|
| SEA2 | 6.77 ± 0.03 | 16.98 ± 1.17 |
| SEA4 | 7.34 ± 0.78 | 16.49 ± 1.51 |
| CIRCLE2 | 10.83 ± 1.61 | 27.05 ± 3.99 |
| SINE2 | 9.30 ± 0.65 | 111.47 ± 12.13 |
| MNIST2 | 5237.85 ± 69.86 | 65610.60 ± 12809.88 |
| MNIST4 | 5486.23 ± 90.06 | 112433.06 ± 14475.72 |

復元で仮定した1モデルの大きさはSEA2/SEA4が4,868 bytes、CIRCLE2/SINE2が4,740 bytes、MNIST2/MNIST4が4,986,280 bytes。既定MLPの値数とfloat32を前提とする。
この仮定の下では、代表条件の全6データセットでFedSDAのモデルpayloadが小さい。実行時前提の確認前に「実測総通信量を削減した」とは結論しない。また、A=100とB=50は同一通信予算の条件ではない。

### 通信量: 記録された転送回数
5 seed平均、いずれもup＋downの合計。モデル転送は`comm_models_total`、メッセージは`comm_messages_total`の実験内カウンタ。

| dataset | モデル転送 FedSDA (回) | FedDrift (回) | メッセージ FedSDA (件) | FedDrift (件) |
|---|---:|---:|---:|---:|
| SEA2 | 1003.4 | 3656.6 | 510.4 | 5341.6 |
| SEA4 | 1246.0 | 3552.2 | 542.8 | 5074.0 |
| CIRCLE2 | 2830.2 | 5983.4 | 678.8 | 11444.2 |
| SINE2 | 2115.6 | 24659.8 | 805.2 | 343891.8 |
| MNIST2 | 2322.6 | 13797.4 | 738.0 | 83914.0 |
| MNIST4 | 3336.8 | 23643.8 | 946.8 | 282706.8 |

メッセージ数には実装が計上するモデル・統計・制御等の送受信を含むが、ネットワークpacket数や各メッセージのbyte量ではない。
モデル構造と共有部の転送方式が異なるため、「モデル転送1回」が両手法で同じpayloadを持つとは限らない。この表の転送回数だけを通信byte比へ置き換えない。

### 出典・確認範囲
結果は元checkoutのgit管理外`results/`にある。worktreeへ複製していない。以下はリポジトリルートを基準にしたパス表記で、現在の環境では元checkoutのルートから参照する。git cloneだけでは取得できない。

| 集計 | 入力・選択条件 |
|---|---|
| FedSDA | `results/_audit_20261002/paper_selection.json`の`selected.reference.metrics_csv`。random・A=100・各dataset/seed一意、30 run |
| FedDrift | `results/baselines/feddrift/manifest.json`と`results/baselines/feddrift/<dataset>/metrics.csv`。random・B=50・δ=0.1・`sweep_parameter=feddrift_detection_batch_size`、30 run |
| 回復区間 | `results/_audit_20261002/recovery_summary.csv`の`Switching A=100`/`FedDrift B=50`。12行を`results/_audit_20261002/recovery_curves.csv`の各200点平均へ照合 |
| 既存集計との確認 | FedSDAの平均を`results/_audit_20261002/main_summary.csv`の`variant=reference`・A=100へ照合 |
| 通信payload復元 | `results/_audit_20261002/feddrift_payload_provenance.json`の式・モデル値数。baseline CSVから再計算し、`feddrift_payload_unique.csv`の選定30行と`feddrift_payload_summary.csv`の6群の平均・標本標準偏差へ照合 |

通信表についても各条件の5 seedを確認し、up＋down＝total、CSVハッシュ一致を検証した。FedDriftの各方向のpayloadは復元表と一致し、通信表の平均・標本標準偏差も既存復元集計と一致する。元CSV・baseline・復元表は変更していない。

FedSDA元CSVとFedDrift各dataset CSVのSHA256が、それぞれ採用索引・baseline manifestの記録値と一致することを確認した。各条件はseed 0–4の5件で、欠測・重複・非有限値なし。
過去監査では同dataset/seedのドリフト列一致を確認しているが、入力標本全体のバイト単位一致や全学習設定・実行環境の同一性まで保証しない。
FedDriftのCSV/rawにR・実行commit・dtypeの完全な実行記録がないため、既定値から補って同一環境の比較とは記載しない。
この表は指定した代表点の要約。A/Bの掃引全体やParetoの評価は元結果を用いて別途行う。

## 根拠と更新先

- FedSDAの構成・採用値の正本: [proposed-method.md](proposed-method.md)。本表に新たな採用設定を定義しない。
- 詳細: [監視](../components/drift-detection.md)、[既存モデル再利用](../components/model-reuse.md)、[候補採否](../components/new-model-creation.md)、[共有部・共同学習](../components/shared-backbone.md)、[モデル統合](../components/model-consolidation.md)。
- 旧実装の比較動作: [FedDrift client](../../federated_drift_experiment/clients/feddrift.py)、[FedDrift server](../../federated_drift_experiment/servers/feddrift.py)、[FedSDA server](../../federated_drift_experiment/servers/fedsda.py)、[mode定義](../../federated_drift_experiment/experiment.py)。
- 成果物・条件の所在: [baseline索引](../experiments/baselines.md)。個々のrun条件はmanifest、数値結果はCSV/NPZで確認する。

リファクタリング中の新パッケージは部品の移植段階。本表は実験で用いた手法の構成を説明し、新パッケージの全体実行が完成したことを示すものではない。
