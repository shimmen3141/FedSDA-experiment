# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASPの仕組み（変化の検出、MDLによる保有モデルの再利用・新しいモデルの作成、固い割り当て、Fixed-Shareの混合予測）を、単一のデータストリーム（1クライアント）のオンライン分類として、既存の手法（ARF、SRP、PEARL、SELeCT、ECPFなど）と同じ条件で比べたい研究者。
- 今の状況: 製品のFedHASPは、事前学習した共有の特徴抽出部とモデルごとのadapter・出力層を持つ神経網を学習器にし、連合学習の実行基盤に組み込まれている。単一ストリームの既存手法の多くは、Hoeffding treeを学習器にし、事前学習なしで系列の最初の標本から学び、標本ごとに1回更新する条件で評価されている。学習器と条件が違うため、仕組みの比較にならない。製品の1件の標本の処理は神経網の部品に直接つながっていて、学習器を差し替えられない。
- 何を変えるか: FedHASPの1クライアントの純粋な形（同期・集約・統合・同期の版の混合のない形）を、Hoeffding treeを学習器にして、製品とは別のパッケージとして実装する。条件は単一ストリームの既存手法に合わせる（事前学習なし、系列の最初の標本から評価、標本ごとに1回学習、既定のハイパーパラメータ、データごとの調整なし）。製品のtorchを使わない判断の部品（e-processの監視、Fixed-Shareの重み、MDLの選択の計算、損失の統計）はimportで再利用し、依存は新しいパッケージから製品への一方向だけにする。後で連合学習の実装から切り離せるようにする。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
単一ストリームのオンライン分類の手法として、FedHASPの仕組みをHoeffding treeの学習器で動かすパッケージ`single_stream_hard_assignment_soft_prediction`を作る。書き出した系列（研究ツール`stream_export`の形式）を入力に、標本ごとに予測→ラベル観測→監視→割り当て→学習を行い、prequentialの精度と、警報・選択・モデルの記録を保存する。比較手法（Riverの手法）と同じ系列・同じ条件で評価できるようにする。

## Boundary Context
- **In scope**: Hoeffding treeの概念モデル（予測確率・損失・学習・写し）、1標本の処理（Fixed-Shareの混合予測、監視、帰属の保留、警報、MDLによる選択、割り当てと学習）、系列1本のrun、成果物の保存、書き出した系列を読む実行コマンド、製品との依存の境界。
- **Out of scope**: 製品（`federated_learning_experiments`）の変更。連合学習・集約・統合・同期の版の混合。神経網の学習器。新しいデータセットの生成（別のspecまたは研究ツール）。比較手法の追加（ECPF、PEARLほか）と結果の集計（別に行う）。
- **Adjacent expectations**: 製品のtorchを使わない判断の部品（`OverallAndTrueClassLossMonitor`、`FixedShareWeightController`、`select_assignment_by_minimum_description_length`、`select_new_model_learner`、`compute_switching_code_length`、`ModelAndClassLossStatisticsStore`、`select_loss_monitoring_baseline_mean_loss`）の挙動は製品のtestが保証する。系列のNPZは研究ツール`stream_export`が書き出す（`features`、`labels`、`concept_ids`）。RiverのHoeffding treeはRiverが保証する。

## Requirements

### Requirement 1: 既存手法に合わせた条件
**Objective:** As a 研究者, I want 単一ストリームの既存手法と同じ条件で動かす, so that 学習器と条件の違いを除いて仕組みを比べられる

#### Acceptance Criteria
1. The system shall 学習器にRiverの`HoeffdingTreeClassifier`を既定のハイパーパラメータのまま使い、事前学習も、系列とは別のデータも使わない。
2. The system shall 系列の最初の標本から、各標本について、学習・監視より前に予測し、その予測を評価に使う。
3. The system shall 各標本を、固く割り当てたモデルで1回だけ学習する（同じ標本を繰り返し学習しない）。
4. The system shall 真のコンセプトのIDを、記録と評価にだけ使い、予測・監視・選択・学習には使わない。
5. The system shall 手法のハイパーパラメータ（帰属の保留の容量30、最小の変化区間5、Fixed-ShareのT=120、監視の設定、ウォームアップ200件、確率の平滑化）を、データによらず固定し、runの記録に残す。

### Requirement 2: 概念モデル
**Objective:** As a 研究者, I want 各概念モデルから、監視とMDLに使える確率と損失を得る, so that 製品と同じ判断の部品へ渡せる

#### Acceptance Criteria
1. The system shall 各概念モデルの予測確率を、全クラス（系列のクラス数）について返し、木がまだ予測を持たないときは一様分布、持たないクラスは0とした上で、一様分布と重み0.01で混ぜて平滑化する（最大の確率のクラスは変わらない）。
2. The system shall 標本の損失を、監視用には1−正解クラスの確率（0以上1以下）、MDL用には−ln（正解クラスの確率）として、平滑化した確率から求める。
3. The system shall 概念モデルの写し（独立に学習できる複製）と、空の木から始める新しい概念モデルを作れる。

### Requirement 3: 予測（柔らかい予測）
**Objective:** As a 研究者, I want 保有する全概念モデルの予測をFixed-Shareの重みで混ぜる, so that FedHASPの柔らかい予測を単一ストリームで再現する

#### Acceptance Criteria
1. When 標本が到着したとき, the system shall 保有する全概念モデルの平滑化した確率を、製品の`FixedShareWeightController`のラベル観測前の重みで混ぜ、最大の確率のクラス（同点は小さいクラス番号）を予測とする。
2. When ラベルを観測したとき, the system shall 各概念モデルの監視用の損失で、製品と同じ設定のFixed-Shareの重みを更新する。
3. When 保有する概念モデルの集合が変わったとき, the system shall 重みの扱いを製品の`FixedShareWeightController`に任せる（集合の変化での同期の規則は製品と同じ）。

### Requirement 4: 監視と帰属の保留
**Objective:** As a 研究者, I want 現在のモデルの損失を製品と同じ検出器で監視し、標本を保留してから割り当てる, so that FedHASPの固い割り当てを再現する

#### Acceptance Criteria
1. The system shall 最初の200件（ウォームアップ）では警報を出さず、200件目の標本の処理の後に、現在のモデルの損失の統計の平均を基準値として、製品と同じ設定の`OverallAndTrueClassLossMonitor`を作り、201件目から監視する。
2. While ウォームアップの後, the system shall 各標本の現在のモデルの監視用の損失と正解クラスで監視を進める。基準値は、製品と同じく現在のモデルの損失の統計から`select_loss_monitoring_baseline_mean_loss`で選ぶ。
3. The system shall 各標本を、監視の後に帰属の保留（容量30）へ加え、警報が無ければ、容量を超えた最も古い標本を現在のモデルへ割り当てる。
4. When 標本をモデルへ割り当てるとき, the system shall そのモデルの学習の前の監視用の損失を、そのモデルの損失の統計（製品の`ModelAndClassLossStatisticsStore`）へ記録してから、その標本で1回学習する。

### Requirement 5: 警報とMDLによる選択
**Objective:** As a 研究者, I want 警報の後に変化点と帰属先をMDLで選ぶ, so that FedHASPの再利用・新規作成の判断を再現する

#### Acceptance Criteria
1. When 検出器が合図したとき, the system shall 帰属の保留の標本と警報の標本を古い順に並べた列W（長さn）で、変化区間の長さk（5以上n以下、1件刻み）と帰属先（現在のモデルのまま、損失の統計を持つ別の保有モデル、新しいモデル）を、製品の`select_assignment_by_minimum_description_length`で選ぶ。
2. The system shall 保有モデルの各標本の符号長を、そのモデルのMDL用の損失とし、新しいモデルの予測的符号長を、空の新しい概念モデル（reset）と現在のモデルの写し（keep）にWの標本を新しいものから順に1件ずつ予測させてから学習させた損失の和の前置き（長さkまで）として求め、`select_new_model_learner`で短い方＋ln 2を使う。切り替わりの符号長は`compute_switching_code_length`（T=120）とする。
3. The system shall 符号長を測るための写しと空の概念モデルを、保有モデルとは別に作り、測った後に捨てる（保有モデルの状態を変えない）。
4. If min(n, k)が5より小さいとき, then the system shall 警報とせず、警報の無い標本として処理を続ける（検出器は始め直さない）。
5. When 警報としたとき, the system shall Wの前のn−k件を現在のモデルへ、後のk件を選んだ帰属先へ、古い順に割り当てて学習し（要求4.4）、現在のモデルを帰属先に替え、帰属の保留を空にし、新しい現在のモデルの損失の統計の平均を基準値として監視を始め直す。
6. When 帰属先が新しいモデルのとき, the system shall 選ばれた学習者の種類で新しい概念モデルを作り（resetなら空の木、keepなら現在のモデルの写し）、保有モデルへ加えてから、後のk件を割り当てる。
7. The system shall 警報・選択（仮説、k、帰属先、新しいモデルの学習者、符号長）・適応の結果を、標本の番号とともに記録する。

### Requirement 6: runと成果物
**Objective:** As a 研究者, I want 系列1本を処理して、比較に必要な記録と指標を保存する, so that 既存手法と同じ系列で集計できる

#### Acceptance Criteria
1. When 書き出した系列のNPZ（`features`、`labels`、`concept_ids`）を与えたとき, the system shall 全標本を順に処理し、標本ごとの予測クラス・正誤・観測クラス・真のコンセプト・現在のモデル・保有モデル数・監視の対数e値を記録する。
2. The system shall runの要約として、prequential精度、Kappa、Kappa-M、Kappa-Temporal、警報の回数、作ったモデルの数、最後の保有モデル数、木の節点の数の合計（最後）、経過時間を求める。
3. When 成果物を保存するとき, the system shall 完全な設定・要約・環境・入力の系列のhashのJSONと、記録の名前付きの配列のNPZを保存し、既存のファイルを上書きせず、pickleを使わずに読み戻せるようにする。
4. The system shall 実行コマンドで、系列のファイル（複数）を並列に処理し、データ・seedごとの成果物と一覧のCSVを出力先へ書く。

### Requirement 7: 製品との境界
**Objective:** As a 研究者, I want 新しいパッケージを製品から切り離せる形にする, so that 後で連合学習の実装と別に扱える

#### Acceptance Criteria
1. The system shall 製品（`federated_learning_experiments`）から、要求の「Adjacent expectations」に挙げた判断の部品と、その設定の型だけをimportし、製品は新しいパッケージをimportしない。
2. The system shall 新しいパッケージをimportしても、torchを読み込まない。
3. The system shall 監視・Fixed-Share・MDLの設定の値が、製品のFedHASPの既定（1クライアントの純粋な形で働くもの）と同じであることを、testで確かめられるようにする。
