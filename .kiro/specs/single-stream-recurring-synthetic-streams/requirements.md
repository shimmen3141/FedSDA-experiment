# Requirements Document

## Project Description (Input)
- 誰が困っているか: 単一ストリームの手法（`single_stream_hard_assignment_soft_prediction`、FedHASPの1クライアントの純粋な形のHoeffding treeの版）を、単一ストリームの再来ドリフトの文献でよく使われる合成データで、比較手法と比べたい研究者。
- 今の状況: 比べられる系列は、製品（連合学習）の10データ（SINE2、SEA2・4、CIRCLE2、MNIST系6つ）を研究ツール`stream_export`で書き出したものだけ。文献の定番のSTAGGER・Agrawal・Random RBFの再来の系列が無い。単一ストリームの手法の系列の読込み（`load_single_stream_input`）は、クラス数を製品のデータセットの定義から取るので、製品に無いデータの系列を読めない。製品のデータセットの定義へ足すと、連合学習の側（製品）を変えることになり、単一ストリームの枠組みを後で切り離しにくくなる。
- 何を変えるか: RiverのSTAGGER・Agrawal・Random RBFの生成器から、既存の系列と同じ形式・同じ概念の切り替わりの作り方（製品のランダムな概念列と同じ規則）で再来の系列を書き出す道具を、単一ストリームのパッケージに足す。系列のJSONにクラス数を書き、読込みはそれを使えるようにする。比較手法の研究ツール`river_baselines.py`は、事前学習の標本の無い系列を事前学習ありで流そうとしたら、処理の前に止まるようにする。製品（`federated_learning_experiments`）は変えない。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
単一ストリームの手法と比較手法を、文献の定番の合成データ（STAGGER 3概念、Agrawal 4概念、Random RBF 4概念）の再来の系列で比べられるようにする。系列は既存の書き出しと同じNPZ・JSONの形式で、木の版の実行コマンド・比較手法の研究ツール・比較の集計が、変更なしか小さな変更で読める。

## Boundary Context
- **In scope**: 概念列の生成（製品の規則の写し）、3つのデータの系列の生成と書き出し（実行コマンド）、読込みでのクラス数の扱い、比較手法の研究ツールの事前学習の標本の有無の検査。
- **Out of scope**: 製品の変更（データセットの定義・概念列の生成・連合学習の実行）。実データ（INSECTS、Electricity）。比較手法の追加（PEARLほか）。系列での実験の実行と集計（このspecの後に行う）。神経網・連合学習の側でこのデータを使うこと。
- **Adjacent expectations**:
  - 製品の概念列の生成（`generate_random_client_concept_traces`）と、その規則の値（変化を試すまでの最小の間隔300、試す標本ごとの変化の確率0.0015）は製品が持つ。このspecはそれを写し、製品と同じ列になることをtestで確かめる。
  - 比較の集計（`single_stream_comparison.py`）は、系列のJSONの`dataset_name`・`random_seed`・`array_sha256`を読む。新しい系列も同じ項目を持つ。

## Requirements

### Requirement 1: 概念列
**Objective:** As a 研究者, I want 新しいデータの概念の切り替わりを、既存の系列と同じ規則で作る, so that データの違いだけを比べられる

#### Acceptance Criteria
1. The system shall 概念列を、製品のランダムな概念列と同じ規則（概念0・直前の変化の番号0から始め、直前の変化から300標本を超えた各標本で確率0.0015で変化を試し、変化するときは今と違う概念から一様に選ぶ）で作る。
2. When 同じ乱数の状態・標本数・概念の数を与えたとき, the system shall 製品の概念列の生成（1クライアント）と同じ列を返す。
3. The system shall 概念列の乱数を、系列のseedから作るPythonの乱数だけから取り、標本の生成の乱数と分ける。

### Requirement 2: 3つのデータの系列
**Objective:** As a 研究者, I want STAGGER・Agrawal・Random RBFの再来の系列を作る, so that 文献の定番のデータで比べられる

#### Acceptance Criteria
1. The system shall データ`stagger3`（RiverのSTAGGER、概念iは分類の関数i、i=0〜2）、`agrawal4`（RiverのAgrawal、概念iは分類の関数i、i=0〜3、数値の特徴の摂動0.05）、`randomrbf4`（RiverのRandom RBF、概念iは中心の配置のseed i、i=0〜3、2クラス・10特徴・中心50）を作れる。どれも2クラス。
2. The system shall 各概念の標本を、その概念専用の生成器から順に取り、概念が再来したときは、その概念の生成器の続きから取る。
3. The system shall 概念ごとの生成器のseedを、系列のseedと概念の番号から決め、データの定義と一緒に記録する。
4. When 同じデータ・seed・標本数を与えたとき, the system shall 同じ系列（同じ内容のhash）を作る。
5. The system shall 製品のデータセットの名前と同じ名前のデータを作らない。

### Requirement 3: 書き出し
**Objective:** As a 研究者, I want 既存の系列と同じ形式で書き出す, so that 木の版・比較手法・集計がそのまま読める

#### Acceptance Criteria
1. The system shall 系列ごとに、NPZ（`features`、`labels`、`concept_ids`）とJSON（`dataset_name`、`random_seed`、`sample_count`、`class_count`、`concept_count`、`feature_count`、特徴の名前、`array_sha256`、概念の変化の回数、生成器の設定、Riverの版）を、既存の系列と同じファイル名の規則で書く。事前学習の標本は書かない。
2. The system shall 内容のhashを、既存の書き出しと同じ計算（`compute_stream_array_digest`）で求める。
3. When NPZとJSONが両方ある系列を書き出そうとしたとき, the system shall その系列を飛ばし、片方だけあるときは止める。
4. The system shall データ・seed・標本数・出力先・並列の数をコマンドの引数で受け取り、知らないデータの名前を処理の前に拒否する。

### Requirement 4: 読込み
**Objective:** As a 研究者, I want 製品に無いデータの系列を木の版で読む, so that 製品を変えずに新しいデータで実行できる

#### Acceptance Criteria
1. When 系列のJSONに`class_count`があるとき, the system shall そのクラス数を使う。
2. If 系列のJSONの`class_count`と、同じ名前の製品のデータセットの定義のクラス数が違うとき, then the system shall 読込みを止める。
3. When 系列のJSONに`class_count`が無いとき, the system shall これまでどおり製品のデータセットの定義からクラス数を取り、定義も無ければ止める。
4. If `class_count`が2以上の整数でないとき, then the system shall 読込みを止める。

### Requirement 5: 比較手法の研究ツール
**Objective:** As a 研究者, I want 事前学習の標本の無い系列を事前学習ありで流す誤りを、処理の前に止める, so that 途中の失敗や条件の違う結果を作らない

#### Acceptance Criteria
1. If `--without-pretraining`を付けずに、事前学習の標本の無い系列を渡したとき, then the system shall 処理を始める前に、その系列の名前を示して止まる。
2. When `--without-pretraining`を付けたとき, the system shall 事前学習の標本の無い系列を処理する。

### Requirement 6: 製品との境界
**Objective:** As a 研究者, I want 製品を変えない, so that 連合学習の結果と既定の挙動が変わらない

#### Acceptance Criteria
1. The system shall 製品（`src/federated_learning_experiments/`）のファイルを変えない。
2. The system shall 新しいmoduleから製品へのimportを、torchを使わない部品に限り、製品から新しいmoduleをimportしない。
3. The system shall 概念列の生成のmoduleで、riverとtorchをimportしない。
