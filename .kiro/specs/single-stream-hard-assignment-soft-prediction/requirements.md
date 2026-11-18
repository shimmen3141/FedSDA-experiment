# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASPの仕組み（変化の検出、MDLによる保有モデルの再利用・新しいモデルの作成、固い割り当て、Fixed-Shareの混合予測）を、単一のデータストリーム（1クライアント）のオンライン分類として、既存の手法（ARF、SRP、PEARL、SELeCT、ECPFなど）と同じ条件で比べたい研究者。
- 今の状況: 製品のFedHASPは、事前学習した共有の特徴抽出部とモデルごとのadapter・出力層を持つ神経網を学習器にし、連合学習の実行基盤に組み込まれている。単一ストリームの既存手法の多くは、Hoeffding treeを学習器にし、事前学習なしで系列の最初の標本から学び、標本ごとに1回更新する条件で評価されている。学習器と条件が違うため、仕組みの比較にならない。製品の1件の標本の処理は神経網の部品に直接つながっていて、学習器を差し替えられない。
- 何を変えるか: FedHASPの1クライアントの純粋な形（同期・集約・統合・同期の版の混合のない形）を、Hoeffding treeを学習器にして、製品とは別のパッケージとして実装する。条件は単一ストリームの既存手法に合わせる（事前学習なし、系列の最初の標本から評価、標本ごとに1回学習、既定のハイパーパラメータ、データごとの調整なし）。製品のtorchを使わない判断の部品（e-processの監視、Fixed-Shareの重み、MDLの選択の計算、損失の統計）はimportで再利用し、依存は新しいパッケージから製品への一方向だけにする。後で連合学習の実装から切り離せるようにする。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
単一ストリームのオンライン分類の手法として、FedHASPの仕組みをHoeffding treeの学習器で動かすパッケージ`single_stream_hard_assignment_soft_prediction`を作る。書き出した系列（研究ツール`stream_export`の形式）を入力に、標本ごとに予測→ラベル観測→監視→割り当て→学習を行い、prequentialの精度と、警報・選択・モデルの記録を保存する。比較手法と同じ系列・同じ条件で評価できるようにする。

## Boundary Context
- **In scope**: Hoeffding treeの概念モデル、手法の設定、1標本の処理（Fixed-Shareの混合予測、監視、帰属の保留、警報、MDLによる選択、割り当てと学習）、系列1本のrunと指標、成果物の保存、書き出した系列を読む実行コマンド、製品との依存の境界。
- **Out of scope**: 製品（`federated_learning_experiments`）の変更。連合学習・集約・統合・同期の版の混合。神経網の学習器。新しいデータセットの生成。比較手法の追加（ECPF、PEARLほか）と、既存の比較手法の再実行。結果の集計。
- **Adjacent expectations**:
  - 製品のtorchを使わない判断の部品（`OverallAndTrueClassLossMonitor`、`LossChangeDetectionSettings`、`FixedShareWeightController`、`PredictionCombinationSettings`、`select_assignment_by_minimum_description_length`、`select_new_model_learner`、`compute_switching_code_length`、`ModelAndClassLossStatisticsStore`、`select_loss_monitoring_baseline_mean_loss`、`DATASET_DEFINITIONS`系のデータセットの定義）の挙動は、製品のtestが保証する。
  - 系列のNPZとそのJSONは研究ツール`stream_export`が書き出す（`features`、`labels`、`concept_ids`、`dataset_name`、`random_seed`、`array_sha256`）。
  - 既存の比較手法の研究ツール`river_baselines.py`は、事前学習の標本を先に学習してから系列を評価している。このパッケージの条件（事前学習なし）とそろえるには、比較手法を事前学習なしで流し直す必要がある。その変更と再実行は、このspecの範囲外とし、別に行う。

## Requirements

### Requirement 1: 既存手法に合わせる条件
**Objective:** As a 研究者, I want 単一ストリームの既存手法と同じ条件で動かす, so that 学習器と条件の違いを除いて仕組みを比べられる

#### Acceptance Criteria
1. The system shall 学習器にRiverの`HoeffdingTreeClassifier`を既定のハイパーパラメータのまま使い、事前学習も、系列とは別のデータ（NPZの事前学習の標本を含む）も使わない。
2. The system shall 系列の最初の標本から、各標本について、学習・監視より前に予測し、その予測を評価に使う。
3. The system shall 各標本を、固く割り当てたモデルで1回だけ学習する（同じ標本を同じモデルで繰り返し学習しない）。
4. The system shall 真のコンセプトのIDを、記録と評価にだけ使い、予測・監視・選択・学習には使わない。
5. The system shall 手法のハイパーパラメータ（design.mdの設定の一覧）を、データによらず固定し、runの記録に残す。
6. The system shall 特徴を、比較手法の研究ツール`river_baselines.py`と同じ変換（各行を、列の番号をキーとしfloatを値とする辞書）でHoeffding treeへ渡す。
7. The system shall クラス数Kを、製品のデータセットの定義（系列のJSONの`dataset_name`で`get_dataset_definition`を引いた`class_count`）から取り、ラベルから推定しない。定義の無いデータセット名と、ラベルが0以上K未満でない系列は、処理の前に拒否する。

### Requirement 2: 概念モデル
**Objective:** As a 研究者, I want 各概念モデルから、監視とMDLに使える確率と損失を得る, so that 製品と同じ判断の部品へ渡せる

#### Acceptance Criteria
1. The system shall 各概念モデルの予測確率を、K個のクラスについて返す。木が予測を持たないときは一様分布、木が返さないクラスは0とし、その上で一様分布と重み0.01で混ぜて平滑化する（最大の確率のクラスは変わらない）。
2. The system shall 標本の損失を、監視用には1−（正解クラスの平滑化した確率）（0以上1以下）、MDL用には−ln（正解クラスの平滑化した確率）として求める。
3. The system shall 概念モデルの写し（元と独立に学習できる複製）と、空の木から始める新しい概念モデルを作れる。

### Requirement 3: 予測（柔らかい予測）
**Objective:** As a 研究者, I want 保有する全概念モデルの予測をFixed-Shareの重みで混ぜる, so that FedHASPの柔らかい予測を単一ストリームで再現する

#### Acceptance Criteria
1. When 標本が到着したとき, the system shall 保有する全概念モデルの平滑化した確率を、製品の`FixedShareWeightController.get_weights_before_label_observation`の重みで混ぜ、最大の確率のクラス（同点は小さいクラス番号）を予測とする。
2. When ラベルを観測したとき, the system shall 予測に使った重みと、各概念モデルの監視用の損失を渡して、`FixedShareWeightController.update_weights_after_loss_observation`で重みを更新する。
3. The system shall 保有モデルの集合が変わったときの重みの同期を`FixedShareWeightController`に任せる（新しいモデルを作った後は、この部品が重みを一様に戻す）。保有モデルの集合が変わらない警報（`stay`・`reuse`）では、重みを作り直さない。

### Requirement 4: 監視と帰属の保留
**Objective:** As a 研究者, I want 現在のモデルの損失を製品と同じ検出器で監視し、標本を保留してから割り当てる, so that FedHASPの固い割り当てを再現する

#### Acceptance Criteria
1. The system shall 最初の200件（ウォームアップ）では監視の部品を呼ばず警報もせず（記録の対数e値の欄はNaN）、200件目の標本の処理の最後に、現在のモデルの損失の統計から`select_loss_monitoring_baseline_mean_loss`で選んだ値を基準値として、design.mdの設定の`OverallAndTrueClassLossMonitor`を作り、201件目から監視する。ウォームアップ中の監視の対数e値はNaNとして記録する。
2. While ウォームアップの後, the system shall 各標本で、現在のモデルの監視用の損失・正解クラス・標本の番号と、現在のモデルの損失の統計から毎回`select_loss_monitoring_baseline_mean_loss`で選んだ基準値を渡して、`observe_loss_after_label_observation`で監視を進める（損失の低下による基準の引き下げは、この部品の既定の挙動に従う）。
3. The system shall 各標本を、監視の後に帰属の保留（容量30）へ加え、警報としなければ、容量を超えた最も古い標本を現在のモデルへ割り当てる。
4. When 標本をモデルへ割り当てるとき, the system shall そのモデルの学習の前の監視用の損失を、そのモデルの損失の統計（`ModelAndClassLossStatisticsStore.record_assigned_loss`、正解クラスつき）へ記録してから、その標本で1回学習する。

### Requirement 5: 警報とMDLによる選択
**Objective:** As a 研究者, I want 警報の後に変化点と帰属先をMDLで選ぶ, so that FedHASPの再利用・新規作成の判断を再現する

#### Acceptance Criteria
1. When 検出器が合図したとき, the system shall 帰属の保留の標本（警報の標本を最後に含む、古い順）の列W（長さn）で、変化区間の長さk（5以上n以下、1件刻み）と帰属先（現在のモデルのまま、損失の統計を持つ現在のモデル以外の保有モデル、新しいモデル）を、製品の`select_assignment_by_minimum_description_length`で選ぶ。現在のモデルのままを選んだときは、製品と同じくk=nとする。
2. The system shall 保有モデルの各標本の符号長をMDL用の損失とし、新しいモデルの予測的符号長を、警報の時点の空の概念モデル（reset）と警報の時点の現在のモデルの写し（keep）に、Wの標本を新しいものから順に1件ずつ「MDL用の損失を足す→学習する」で求めた前置きの和（長さkまで）とする。`select_new_model_learner`で短い方＋ln 2と学習者の名前を得て、切り替わりの符号長は`compute_switching_code_length(switching_probability=1/T, destination_count=帰属先の候補の数+1)`とする。
3. The system shall 符号長を測るための学習者を保有モデルとは別に作り、保有モデルの状態を変えない。
4. If 検出器の合図はあったが、min(n, k)が5より小さいとき（変化区間の探索はk≥5なので、実質はn<5のとき）, then the system shall 警報とせず、警報の無い標本として処理し（容量を超えた最古の標本だけを割り当てる）、検出器は始め直さない。次の標本で検出器の合図が続けば、その標本を最後に含む列Wで選択をやり直す（製品の待つ方針と同じ）。
5. When 警報としたとき, the system shall 次の順で処理する: (a) 帰属先が新しいモデルなら、選ばれた学習者と同じ作り方（resetは空の木、keepは警報の時点の現在のモデルの写し）で新しい概念モデルを作り、Wの後のk件を新しいものから順に「監視用の損失を統計へ記録→学習」して保有モデルへ加える。(b) Wの前のn−k件を古い順に現在のモデルへ割り当てる（要求4.4）。(c) 帰属先が保有モデルなら、後のk件を古い順にそのモデルへ割り当てる（要求4.4）。帰属先が現在のモデルのままなら、全n件を古い順に現在のモデルへ割り当てる。(d) 現在のモデルを帰属先に替え、帰属の保留を空にし、帰属先の損失の統計から`select_loss_monitoring_baseline_mean_loss`で選んだ基準値で`OverallAndTrueClassLossMonitor.reset`を呼ぶ。
6. The system shall 検出器の合図ごとに、標本の番号、選んだ仮説、k、帰属先、新しいモデルの学習者、符号長、警報としたか、適応の結果（`stay`＝現在のモデルのまま、`reuse`＝保有モデルへ、`new`＝新しいモデルへ、`too_short`＝min(n,k)<5で警報としない）を記録する。

### Requirement 6: runと成果物
**Objective:** As a 研究者, I want 系列1本を処理して、比較に必要な記録と指標を保存する, so that 既存手法と同じ系列で集計できる

#### Acceptance Criteria
1. When 書き出した系列のNPZとJSONを与えたとき, the system shall `array_sha256`で内容を確かめてから全標本を順に処理し、標本ごとの予測クラス・正誤・観測クラス・真のコンセプト・現在のモデル・保有モデル数・監視の対数e値を記録する。
2. The system shall runの要約として、prequential精度、Kappa、Kappa-M、Kappa-Temporal（定義はdesign.md）、真の変化点の後の最初の30・100・500件の精度、警報の回数、作ったモデルの数、最後の時点の保有モデル数、最後の時点の木の節点の数の合計、経過時間を求める。
3. The system shall 概念の同定の指標として、最後の時点で各モデルに割り当てた標本の最多の真のコンセプトをそのモデルの概念とし、警報ごとの選択を、正しい再利用・誤った再利用・再来なのに新規作成・初めての概念での新規作成・そのまま、に分類した数を求める（定義はdesign.md）。
4. When 成果物を保存するとき, the system shall 完全な設定・要約・環境・入力の系列のhashのJSONと、記録の名前付きの配列のNPZを保存し、既存のファイルを上書きせず、pickleを使わずに読み戻せるようにする。
5. The system shall 実行コマンドで、系列のファイル（複数）を並列に処理し、データ・seedごとの成果物と一覧のCSVを出力先へ書く。移動窓の精度などの時系列の指標は、保存した標本ごとの正誤から集計の側で求める。

### Requirement 7: 製品との境界
**Objective:** As a 研究者, I want 新しいパッケージを製品から切り離せる形にする, so that 後で連合学習の実装と別に扱える

#### Acceptance Criteria
1. The system shall 製品（`federated_learning_experiments`）から、Boundary Contextの「Adjacent expectations」に挙げた部品だけをimportし、製品は新しいパッケージをimportしない。
2. The system shall 新しいパッケージをimportしても、torchを読み込まない。
3. The system shall 監視・Fixed-Share・MDLの設定のうち、このパッケージで働くfield（design.mdの設定の一覧で「製品と同じ」としたもの）の値が、製品のFedHASPの既定と同じであることを、testで確かめられるようにする。
