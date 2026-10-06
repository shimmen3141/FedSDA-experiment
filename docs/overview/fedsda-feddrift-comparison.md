# FedSDAとFedDriftの構成比較

対象は[最終提案のFedSDA](proposed-method.md)（Residual Adapter＋ClassESR＋Switching SoftRouting）と、このリポジトリの`FedDrift`比較実装。過去のADWIN版・Meta-switching版FedSDAは含めない。
以下は手法の動作と構成の比較であり、性能順位の表ではない。FedDrift原論文との実験条件の相違は[別資料](../experiments/differences-from-feddrift.md)を参照する。

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

## 根拠と更新先

- FedSDAの構成・採用値の正本: [proposed-method.md](proposed-method.md)。本表に新たな採用設定を定義しない。
- 詳細: [監視](../components/drift-detection.md)、[既存モデル再利用](../components/model-reuse.md)、[候補採否](../components/new-model-creation.md)、[共有部・共同学習](../components/shared-backbone.md)、[モデル統合](../components/model-consolidation.md)。
- 旧実装の比較動作: [FedDrift client](../../federated_drift_experiment/clients/feddrift.py)、[FedDrift server](../../federated_drift_experiment/servers/feddrift.py)、[FedSDA server](../../federated_drift_experiment/servers/fedsda.py)、[mode定義](../../federated_drift_experiment/experiment.py)。
- 成果物・条件の所在: [baseline索引](../experiments/baselines.md)。個々のrun条件はmanifest、数値結果はCSV/NPZで確認する。

リファクタリング中の新パッケージは部品の移植段階。本表は実験で用いた手法の構成を説明し、新パッケージの全体実行が完成したことを示すものではない。
