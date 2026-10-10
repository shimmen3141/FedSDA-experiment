# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装の全体runは、datasetがSINE-2（2特徴、2クラス、2概念）のときだけ動く。観測標本の型は「2特徴・二値」、概念列の型は「0/1」に固定で、実行の枠・参加者の準備・事前学習は、SINEの生成器を名指しで受け取る。最終構成のgoldenは3ケース（sine2・sea2・mnist2）あるが、新実装で照合できているのは、sine2だけである。

ユーザーの決定（2026-10-10）: 論文の実験で使うdatasetは、blobs以外の全部（sea4、circle2、sine2、sea2、mnist2、mnist4）。datasetの移植は、Linux用のgoldenより前に行う。

調査で分けたこと: datasetの移植を、2つのspecに分ける。(1)本spec——合成データ（sea2・sea4・circle2）と、そのための一般化（特徴数・クラス数・概念数を、datasetの定義から取る）。(2)次のspec——MNIST（mnist2・mnist4。画像ファイルの読込み、784次元、10クラス、datasetごとの学習率と隠れ層の幅）。

調査で確かめたこと（旧実装。固定）:

- 生成（federated_drift_experiment/data/synthetic.py、streams.py）: 標本1件ごとに、NumPyの全体の乱数で生成する。SEAは、3特徴を`uniform(0, 10)`で引き、`特徴0＋特徴1 <= 閾値`ならラベル1、その後、`rand() < 雑音率`ならラベルを反転する（閾値は概念ごと: 9.0、8.0、7.0、9.5。雑音率0.10）。sea2は、SEAの概念0・1だけを使う。CIRCLE-2は、2特徴を`uniform(0, 1)`で引き、円（概念ごとの中心と半径: (0.2, 0.5, 0.15)、(0.6, 0.5, 0.25)）の外ならラベル1。ラベルは、float64の特徴で判定し、特徴は、その後でfloat32へ直す（SINEと同じ）。
- datasetの定義（data/specs.py）: sea2は3特徴・2概念、sea4は3特徴・4概念、circle2は2特徴・2概念。どれも2クラスで、隠れ層の幅と学習率は、既定（(32, 32)、共通の学習率）。
- 概念列（data/schedules.py の`make_random_schedules`）: 変更のとき、現在の概念以外の概念から、Pythonの乱数の`choice`で選ぶ（概念数は、datasetの定義から）。
- SEAの閾値・雑音率と、CIRCLEの円は、旧では設定（`config.SEA_THRESHOLDS`ほか）だが、掃引の対象ではなく、datasetを決める定数として使われている（goldenの条件は、既定値を明示して固定している）。

## Introduction

研究者が、新実装の全体runを、合成データのdataset（sine2・sea2・sea4・circle2）で実行できるようにする。sea2で、goldenの33指標と31の離散列を照合する。

## Boundary Context

- **In scope**: datasetの定義（特徴数・概念数・クラス数）。SEAとCIRCLE-2の観測標本の生成。観測標本・概念列の型の一般化。概念列の生成の、概念数の一般化。実行の枠・参加者の準備・事前学習・clientが、datasetの定義に従うこと。sea2のgoldenの照合と、sea4・circle2の、実旧との照合。
- **Out of scope**: MNIST（次のspec）。blobs（移植しない）。FedDriftの固定の概念列（`feddrift_fixed`）。datasetごとの学習率と隠れ層の幅（合成データは、既定）。SEA・CIRCLEの定数を、設定で変えること。Linux用のgolden。
- **Adjacent expectations**: sine2の全体runの結果（状態、指標、乱数）は、変えない。固定旧実装とgoldenは、変えない。

## Requirements

### Requirement 1: datasetの定義

**Objective:** As a 研究者, I want datasetごとの特徴数・概念数・クラス数を、1箇所から得たい, so that 部品が、datasetごとの値を、別々に持たない

#### Acceptance Criteria

1. When dataset名を与えたとき, the Dataset Definitions shall そのdatasetの、入力の特徴数・概念数・クラス数を返す（sine2: 2・2・2、sea2: 3・2・2、sea4: 3・4・2、circle2: 2・2・2）。
2. If 定義のないdataset名（mnist2・mnist4・blobs・未知の名前）を与えたとき, the Dataset Definitions shall 拒否する。
3. The 定義 shall 実旧のdatasetの定義と一致する。

### Requirement 2: 観測標本の生成

**Objective:** As a 研究者, I want SEAとCIRCLE-2の標本を、旧と同じ規則・同じ乱数の順で得たい, so that 旧の結果と照合できる

#### Acceptance Criteria

1. When SEAの生成器へ概念IDを与えたとき, the Sea Sample Generator shall 借りたNumPyの乱数で、3特徴を0以上10未満の一様分布から引き、特徴0＋特徴1が概念の閾値以下ならラベル1、そうでなければ0とし、その後、雑音率の確率でラベルを反転した、観測標本を返す。
2. When CIRCLE-2の生成器へ概念IDを与えたとき, the Circle Sample Generator shall 借りたNumPyの乱数で、2特徴を0以上1未満の一様分布から引き、概念の円の外ならラベル1、中または円周上なら0の、観測標本を返す。
3. The 生成器 shall ラベルを、float32へ直す前の特徴で判定し、特徴を、float32へ直した値で返す。
4. If 概念IDが、そのdatasetの概念でないとき（型の不正、範囲外）, the 生成器 shall 乱数を進めずに拒否する。
5. When dataset名と、借りたNumPyの乱数を与えたとき, the Observed Sample Generation shall そのdatasetの生成器を返す。
6. The 生成 shall 同じ乱数の状態から、実旧の生成（`generate_data`）と、特徴・ラベル・生成の後の乱数の状態が一致する。

### Requirement 3: 型と概念列の一般化

#### Acceptance Criteria

1. The 観測標本 shall 1つ以上のfloatの特徴と、0以上の整数のクラスラベルを持てる（真の概念は持たない）。
2. The 概念列 shall 0以上の整数の概念IDを持てる。
3. When 概念列を生成するとき, the Random Concept Schedule Generation shall 変更のとき、datasetの概念のうち、現在の概念以外から、借りたPythonの乱数で選ぶ（候補が1つでも、乱数の消費を省かない）。4概念のdatasetで、実旧の概念列と一致する。

### Requirement 4: 全体runが、datasetの定義に従う

#### Acceptance Criteria

1. When 全体runを実行するとき, the Single Run System shall 実行条件のdataset名から、生成器を作り、参加者の準備と、観測列の生成に使う。
2. When 参加者を準備するとき, the FedSDA Run Participant Factory shall 初期モデルの、入力の特徴数とクラス数を、datasetの定義から取る。
3. If 実行条件のdataset名が、対応していないdataset（mnist2ほか）のとき, the Single Run System shall 実行を始める前に拒否する。
4. The 変更 shall sine2の全体runの結果（状態、指標、離散列、計算量、乱数）を変えない。

### Requirement 5: 実旧・goldenとの一致

#### Acceptance Criteria

1. When goldenの条件（sea2）で、新の全体runと、実旧の`run_random_drift_experiment`を実行したとき, the 新の結果 shall 33指標が完全に一致し、31の離散列が、形・型・値で一致し、Windowsの基準環境では、Windows用のgolden（sea2）と一致する。
2. When sea2・sea4・circle2の小さい条件で実行したとき, the 新の全体run shall 実旧の全体runと、サーバ・全client・診断の記録・乱数の最終状態が一致する。
3. The 対照 shall sea4で、3つ以上の概念が現れる条件を含む。
4. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、sine2以外のdatasetの全体runを実行できる。
5. The 追加 shall 依存の許可の範囲に収める。
