# Requirements Document

## Project Description

FedSDAとFedDriftを比較する研究者のため、FedDriftのサーバ同期を新構成へ接続する。clientの観測/学習と保存済みモデルの統合計算は揃ったが、統合結果のclient適用、隔離登録、R回のFedAvg/配布が未接続である。旧実験の時刻手順を新構成だけで実行し、各操作後の値・状態・通信量・乱数消費を実旧へ一致させる。

## Requirements

### 1 clientへの統合と配布の適用

1.1 When 統合のID対応とclient自身の平均parameterまたはサーバ配布parameterが渡される, the FedDrift synchronization shall 統計を最多件数優先/同率は保有順で選び、0件/欠落にはサーバ統計を補う。学習/評価標本と計数を旧の順で再配置し、評価標本が上限を超えた場合だけ旧と同じ抽出を行う。
1.2 When 正式モデルを配布する, the FedDrift synchronization shall 配布順に分類器と空のoptimizerを作り直し、一時モデルの値・optimizer・参照を保持する。現在IDを対応付け、変更時だけ処理済み標本数の位置にserver適応記録を残す。
1.3 If 配布の対応・値・統計・optimizer設定またはclient状態が契約外である, the FedDrift synchronization shall owner更新と乱数消費より前に拒否する。

### 2 時刻の準備

2.1 When 時刻の準備を行う, the FedDrift synchronization shall 状態報告を数え、隔離期限に達したグローバルモデルだけを昇順でクロス評価し、損失増加距離と指定linkage（complete/average/connected）でクラスタを決める。評価依頼ではモデルparameterを送信しないが、評価用分類器は旧と同じ場合に生成する。
2.2 When クロス評価を行う, the FedDrift synchronization shall 候補ID→対象ID→clientの旧順で評価し、保持者が上限を超える場合に抽出する。評価storeが6件以上、または現行モデルの学習storeが11件以上の場合に評価し、標本上限で抽出した後5件未満なら評価しない。真の概念は診断だけに用いる。
2.3 When 複数モデルの群ができる, the FedDrift synchronization shall 完了済みの保存済み統合計算を利用してサーバとclient自身の値を別々に平均し、隔離中のモデルを単独群として保持する。統合の制御通知をclient数だけ数え、クラスタの診断を残す。
2.4 When 統合を終える, the FedDrift synchronization shall readyの新モデルをclient順に正式IDへ登録し、時刻＋隔離期間を期限とする。登録時にはparameterを転送せず、ID通知を1件数える。

### 3 学習ラウンド

3.1 When 学習済みclientを集約する, the FedDrift synchronization shall 正式ID昇順・client順に、モデルごとの学習store件数でFedAvgする。乗算→順次加算→合計で割る演算を維持し、学習storeが0件のclientは通信/統計集約にも含めない。
3.2 When 集約を終える, the FedDrift synchronization shall 正の集約件数だけで保存重みを更新し、0件なら過去重みとparameterを保持する。全グローバルモデルを全clientへ配布し、論理転送数・値数・バイト数を旧へ一致させる。
3.3 When 時刻同期の入口を呼ぶ, the FedDrift synchronization shall 2の準備の後に、設定R回だけclient順のローカル学習→集約→配布を実行する。観測済みclientを入力とし、観測を重ねて実行しない。

### 4 境界と検証

4.1 If 時刻・設定・client集合・初期server状態が契約外である, the FedDrift synchronization shall 状態記録/採番/通信/学習/乱数消費より前に拒否する。
4.2 The FedDrift synchronization shall 旧実装を実行時に読み込まず、固定旧/goldenを変更しない。正常経路の値・owner・optimizer・通信・離散記録・3RNGを実旧の連続操作に照合する。

## Scope

旧実験の通常FedDriftサーバ（collect_pair_diagnostics=False）のprepare_timestepとR回進行を移植する。全参加者/データ生成のfactory、終端run_round、サーバ精度評価、全体runの指標/成果物/CLI、任意の相補性診断は後続の全体run接続で扱う。全体golden一致を本specの完了として主張しない。
