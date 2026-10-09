# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: clientは、初期モデル（分類器と、概念固有部・共有部の2つのoptimizerの状態）と初期の損失統計から組み立てられる（fedsda-run-client-assembly）が、その初期モデルを作る処理がない。対照testは、実旧の事前学習の結果を写して使っている。旧は`federated_drift_experiment/experiment.py`の`_pretrain_initial_model`（294〜334行）が、runの最初に、概念0の標本でモデル0を学習し、その標本での損失統計を求める。

変えたいこと: 初期モデルの事前学習を、旧と同じ順序・数値・乱数の消費で行い、結果を、clientの組立てがそのまま受け取れる形で返す。

調査で確かめたこと（固定旧748c3aa）:

- 旧の順序: モデルを作る（torchの乱数で初期化）→概念0の標本をN件作る（NumPyの乱数）→epochごとに、標本の並びをshuffleし（Pythonの乱数）、batchごとにモデルを1回更新する（共有部のoptimizer→概念固有部のoptimizerの順にstep）→最後の並びのまま、標本を1件ずつ評価して、損失統計（全体と、観測クラス別。Welfordの逐次更新）を求める。
- 旧のrunは、3つの乱数（Python、NumPy、torch）を同じseedで初期化してから、事前学習を行う（`experiment.py` 2236〜2241行）。
- 新には、対応する部品がある: 分類器の生成（旧と同じ初期化の順）、SINEの標本生成器、1モデルの共同更新（候補の学習が同じ形で使い、実旧のモデルの更新と照合済み）、1標本の有界損失の評価、損失統計の逐次更新。
- 新の実行の枠は、準備の処理（`prepare_run`）へ、runの乱数源（PythonとNumPy）とSINEの標本生成器を渡し、torchの乱数はseedつきの範囲の中で実行する。

## Introduction

研究者が、初期モデルの事前学習を、旧と同じ結果になる1回の呼出しで行い、その結果からclientを組み立てられるようにする。

## Boundary Context

- **In scope**: 事前学習の条件（標本数、epoch数、batchの件数）の設定型。初期の分類器と2つのoptimizerの状態の生成、概念0の標本の生成、epochごとのshuffleとbatchごとの更新、損失統計の計算。結果を、clientの組立てが受け取る形の記録で返すこと。
- **Out of scope**: 全clientとサーバを準備する処理（実行の枠の`prepare_run`を持つfactory）と、サーバへの初期モデル・統計の登録。SINE以外のdataset。datasetごとのモデルの寸法（隠れ層の幅、クラス数）の設定の置き場所（引数で受け取る）。計算量の記録。
- **Adjacent expectations**: 分類器の生成、標本生成器、共同更新、損失の評価、損失統計の逐次更新は、移植済みの部品をそのまま使い、中の処理を変えない。torchの乱数は、呼出し側が初期化した全体の乱数を使う（実行の枠が、seedつきの範囲の中で呼ぶ）。

## Requirements

### Requirement 1: 事前学習

**Objective:** As a 研究者, I want 初期モデルの事前学習を1回の呼出しで行いたい, so that clientの組立てへ渡す初期モデルと統計が得られる

#### Acceptance Criteria

1. When 事前学習を求められたとき, the Initial Model Pretraining shall 分類器を新しく1つ作り、その概念固有部と共有部のそれぞれにoptimizerの状態を作る。
2. When 分類器を作った後, the Initial Model Pretraining shall 概念0の観測標本を、設定の標本数だけ、標本生成器から順に得る。
3. When 標本を得た後, the Initial Model Pretraining shall 設定のepoch数だけ、標本の並びを借りた乱数生成器でshuffleし、先頭から設定のbatchの件数ずつ（最後は残り全部）取り出して、batchごとに、共有部と概念固有部を1回更新する。
4. When 全epochを終えた後, the Initial Model Pretraining shall 最後の並びのまま、標本を1件ずつ評価し、その有界損失で、全体と観測クラス別の損失統計を逐次に更新する。クラス別の統計は、クラスが最初に現れた順に持つ。
5. The Initial Model Pretraining shall 分類器、2つのoptimizerの状態、損失統計を、clientの組立てがそのまま受け取れる1つの記録で返す。

### Requirement 2: 実旧との一致

**Objective:** As a 研究者, I want 事前学習の結果と乱数の消費が、旧と同じであってほしい, so that その後のclientの生成と標本処理が、旧と同じ状態から始まる

#### Acceptance Criteria

1. When 3つの乱数を同じseedで初期化して事前学習を行ったとき, the Initial Model Pretraining shall 実旧の事前学習と、分類器の全パラメータ、2つのoptimizerの状態、損失統計（全体とクラス別、クラスの順）が一致し、実行後の3つの乱数の状態も一致する。
2. When 事前学習の結果からclientを組み立てたとき, the Run Client shall 実旧の事前学習の結果から実`__init__`で作った実旧のclientと、生成直後と、続く標本列の処理の後に、全状態が一致する。

### Requirement 3: 検査

**Objective:** As a 研究者, I want 不正な設定や引数が、何かを作る前に拒否されてほしい, so that 乱数だけが進んだ状態が残らない

#### Acceptance Criteria

1. If 事前学習の条件（標本数とbatchの件数は1以上の整数、epoch数は0以上の整数）が型または範囲に合わないとき, the 事前学習の設定 shall 生成時に拒否する。
2. If 事前学習の設定、モデル構造の設定、optimizerの設定、標本生成器、乱数生成器が決まった型でないとき、または隠れ層の幅とクラス数が分類器の生成の条件に合わないとき, the Initial Model Pretraining shall 分類器を作らず、どの乱数も消費せずに拒否する。
3. The Initial Model Pretraining shall 並行した呼出しでの結果を保証しない。

### Requirement 4: 新実装だけで動くことと依存の向き

**Objective:** As a 研究者, I want 事前学習からclientの実行までが、旧実装なしでつながってほしい, so that 残りはサーバ側だけになる

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、事前学習を行い、その結果からclientを組み立てて、実行の枠で標本列を最後まで進められる。
2. The Initial Model Pretraining shall 新しいmoduleの依存を、既存と同じ形で登録した許可の範囲に収める。
