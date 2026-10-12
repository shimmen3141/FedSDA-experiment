# Requirements Document

## Project Description
概念ドリフト下でFedSDAとFedDriftを比較する研究者が、よりナイーブな単一モデルの比較手法も新実装で実行できるようにする。旧Obliviousの一連の流れを、正式な構成名fedavgで移植する。ユーザーは単一specへの集約と命名判断を主担当へ委任した。

## Introduction
範囲は単一モデルの逐次学習から同期、全体run、共通評価、保存、既存掃引コマンドまで。blobs、旧名alias、汎用FedAvgの新しい変種、既知LEGACY/IMPROVE修正、新golden追加、旧実装の削除は含めない。

## Requirements
### 1. 単一モデルの逐次学習と同期
1.1 When fedavgを実行するとき, the system shall 全clientとserverでモデルID0だけを保持し、検出・モデル生成・切替・クラスタリングを行わない。
1.2 When 標本が到着するとき, the system shall 学習前に予測と損失を記録し、全到着標本を保持し、旧Obliviousと同じbatch抽出・更新予算・保留更新の区間末消化を行う。
1.3 When 同期するとき, the system shall 累積保持標本件数によるFedAvgをclient順に計算し、全clientへ配布してoptimizerを再生成し、clientの保持データと損失統計を維持する。
1.4 When 不正な標本・位置・設定が指定されるとき, the system shall 入力検査に伴う状態変更・乱数消費より前に拒否する。実行中の計算失敗は成功済み操作を巻き戻さず、失敗位置とcauseを残す。
### 2. 完全設定から全体run
2.1 When 完全設定でrunを実行するとき, the system shall 6dataset、seed、規模、集約間隔、概念変更条件、事前学習、モデル寸法、optimizer、batch、更新間隔・予算、評価条件を型付き設定から適用する。
2.2 When 全体runが進行するとき, the system shall 初期モデル生成→事前学習→client複製→概念列→観測列→標本位置/client順の処理→保留消化→同期の順を維持し、完全区間だけを処理して未処理末尾を記録し、終端で追加学習・同期を行わない。
2.3 When 真の概念IDが渡されるとき, the system shall 診断記録だけに用い、予測・学習・同期には用いない。
### 3. 共通評価と成果物
3.1 When runを評価するとき, the system shall 共通定義で予測精度、定常精度、切替検出指標（切替列は空）、モデル数、通信量、parameter footprint、計算量とその正規化を導出する。
3.2 When 計算・通信を計測するとき, the system shall 事前準備を本実行と分け、予測・損失統計・学習、集約、ラウンドごとの内訳を記録し、初期配布は通信として数えない。
3.3 When 成果物を保存するとき, the system shall 完全設定・指標・環境のJSONと、予測・生成概念列・同期・通信/計算・学習量/損失統計の名前付きNPZを保存し、既存の片方でも存在すれば上書きせず、pickleなしで読めるようにする。保存前に古い指標と現在記録の不一致を拒否する。
### 4. 掃引と回帰
4.1 When fedavgを構成として選ぶとき, the system shall 既存コマンドの計画表示、逐次/別process実行、保存、CSV一覧、再開・競合拒否まで接続する。FedDrift専用閾値軸ではfedavgを増殖させない。
4.2 When 移植を検証するとき, the system shall 実旧の初期準備・全体runと予測列、モデル値、optimizer、保持データ、統計、更新件数、通信、乱数順を照合し、既存手法・旧goldenを維持する。
