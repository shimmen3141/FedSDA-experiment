# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装の全体runは、合成データ（sine2・sea2・sea4・circle2）で実行できる。最終構成のgoldenの3ケース（sine2・sea2・mnist2）のうち、新実装で照合できていないのは、mnist2だけである。

ユーザーの決定（2026-10-10）: 論文の実験で使うdatasetは、blobs以外の全部（sea4、circle2、sine2、sea2、mnist2、mnist4）。datasetの移植は、Linux用のgoldenより前に行う。

調査で確かめたこと（旧実装。固定）:

- 読込み（federated_drift_experiment/data/mnist.py）: 学習用の画像とラベルの2つのgzipのIDXファイル（`train-images-idx3-ubyte.gz`・`train-labels-idx1-ubyte.gz`）を、`FDE_MNIST_DATA_DIR`（なければ、リポジトリ直下の`data/mnist`）から読む。ファイルがなければ、ネットワークから取得する。画像は、uint8の画素を、float32へ直して255.0で割る（60000件×784次元）。読んだ結果は、processの中で使い回す。
- 生成（`sample_mnist`、data/streams.pyの`generate_data`）: 標本1件ごとに、NumPyの全体の乱数の`randint(0, 件数, size=1)`で、画像の位置を1つ選ぶ（乱数の消費は、これだけ）。ラベルは、概念ごとに、数字の対を交換する（概念0: そのまま、概念1: 1と2、概念2: 3と4、概念3: 5と6）。mnist2は概念0・1、mnist4は概念0〜3を使う。
- datasetの定義（data/specs.py）: mnist2は784特徴・2概念・10クラス、mnist4は784特徴・4概念・10クラス。どちらも、隠れ層の幅は(1568,)、学習率は1e-3で、この学習率は、通常の学習率（`BASE_LR`）と、新規モデルの学習率（`NEW_MODEL_LR`）の両方に優先する（models.py）。
- goldenの条件（tests/test_proposed_regression.pyの`CASES`の`mnist2`）: client 2、標本100件、事前学習2 epoch、`MIN_STABLE_PERIOD=30`。

## Introduction

研究者が、新実装の全体runを、MNIST（mnist2・mnist4）で実行できるようにする。mnist2で、goldenの33指標と31の離散列を照合する。

## Boundary Context

- **In scope**: MNISTの学習用データの読込み（置いてあるファイルから）。MNISTの観測標本の生成。datasetの定義へのmnist2・mnist4の追加。mnist2のgoldenの照合と、mnist2・mnist4の、実旧との照合。
- **Out of scope**: ファイルの取得（ネットワーク）。blobs。FedDriftの固定の概念列。datasetごとの学習率と隠れ層の幅を、新実装の中に既定値として持つこと（設定の束が受け取る。完全なrun設定を決めるspecで扱う）。観測標本の特徴の持ち方（floatのtuple）の変更。Linux用のgolden。
- **Adjacent expectations**: 合成データ（sine2・sea2・sea4・circle2）の全体runの結果は、変えない。固定旧実装とgoldenは、変えない。

## Requirements

### Requirement 1: datasetの定義

**Objective:** As a 研究者, I want mnist2・mnist4の特徴数・概念数・クラス数を、ほかのdatasetと同じ所から得たい, so that 実行の枠と参加者の準備が、そのまま使える

#### Acceptance Criteria

1. When dataset名を与えたとき, the Dataset Definitions shall mnist2は784・2・10、mnist4は784・4・10（入力の特徴数・概念数・クラス数）を返す。
2. The 定義 shall 実旧のdatasetの定義と一致する。
3. The 実行条件 shall dataset名として、mnist4を受け取る（mnist2は、すでに受け取る）。

### Requirement 2: 学習用データの読込み

**Objective:** As a 研究者, I want MNISTの学習用データを、置いてあるファイルから読みたい, so that 実験のコードが、実行中にネットワークへ出ない

#### Acceptance Criteria

1. When ディレクトリを与えたとき, the Mnist Training Data shall 画像とラベルの2つのファイルを読み、画素（件数×784）と、数字のラベル（件数）を返す。
2. The 読んだ画素から作る特徴 shall 実旧の読込みの結果（float32の画像）と、全件で一致する。ラベルも、全件で一致する。
3. If ファイルがないとき, the Mnist Training Data shall 取得を試みずに、足りないファイルと、置き場所の指定の方法が分かる例外で拒否する。
4. If ファイルの形式が不正なとき（識別の数値の不一致、中身の長さの不一致、画像とラベルの件数の不一致、1件が784画素でない）, the Mnist Training Data shall 拒否する。
5. When 同じディレクトリを、同じprocessの中で再び読むとき, the Mnist Training Data shall ファイルを読み直さずに、同じ結果を返す。返した配列は、書き換えられない。
6. When 置き場所を決めるとき, the Mnist Training Data shall 環境変数`FDE_MNIST_DATA_DIR`があればその値、なければ、リポジトリ直下の`data/mnist`を使う（旧と同じ規則）。

### Requirement 3: 観測標本の生成

**Objective:** As a 研究者, I want MNISTの標本を、旧と同じ規則・同じ乱数の順で得たい, so that 旧の結果と照合できる

#### Acceptance Criteria

1. When 生成器へ概念IDを与えたとき, the Mnist Sample Generator shall 借りたNumPyの乱数で、0以上・件数未満の位置を1つ選び、その画像の784個の特徴（float32の値を、Pythonのfloatにしたもの）と、概念で交換したラベルの、観測標本を返す。
2. The ラベルの交換 shall 概念0はそのまま、概念1は1と2、概念2は3と4、概念3は5と6を入れ替える。
3. If 概念IDが、そのdatasetの概念でないとき（型の不正、範囲外）, the 生成器 shall 乱数を進めずに拒否する。
4. When dataset名（mnist2・mnist4）と、借りたNumPyの乱数を与えたとき, the Observed Sample Generation shall 置き場所の規則（2.6）で読んだ学習用データを使う、そのdatasetの生成器を返す。乱数は進めない。
5. The 生成 shall 同じ乱数の状態から、実旧の生成（`generate_data`）と、特徴・ラベル・生成の後の乱数の状態が一致する。
6. The 生成器 shall 標本ごとの特徴の保持に、画素の値ごとのfloatを使い回す（1標本あたりのメモリを、特徴ごとにfloatを作る場合より小さくする）。

### Requirement 4: 実旧・goldenとの一致

#### Acceptance Criteria

1. When goldenの条件（mnist2）で、新の全体runと、実旧の`run_random_drift_experiment`を実行したとき, the 新の結果 shall 33指標が完全に一致し、31の離散列が、形・型・値で一致し、Windowsの基準環境では、Windows用のgolden（mnist2）と一致する。
2. When mnist2・mnist4の小さい条件で実行したとき, the 新の全体run shall 実旧の全体runと、サーバ・全client・診断の記録・乱数の最終状態が一致する。
3. The 対照 shall mnist4で、3つ以上の概念が現れる条件を含む。
4. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、mnist2・mnist4の全体runを実行できる。
5. The 変更 shall 合成データの全体runの結果（状態、指標、離散列、計算量、乱数）を変えない。追加は、依存の許可の範囲に収める。
