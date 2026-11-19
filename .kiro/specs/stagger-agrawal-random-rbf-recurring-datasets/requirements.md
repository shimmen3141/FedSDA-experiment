# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASP（FedSDA）を評価する研究者（ユーザー）。1クライアントの単一ストリームの版では、STAGGER・Agrawal・Random RBFの再来の系列で、木のアンサンブル（ARF・SRP・LB）に負けた（[記録](../../../docs/research/fedhasp-single-client-pilot-v16.md)）。一方、連合学習の比較は、SINE・SEA・CIRCLE・MNIST系だけで行ってきた。FedHASPが連合学習でも、これらのデータで有望かが分からない。
- 今の状況: 3データの系列は、単一ストリームのパッケージ（`src/single_stream_hard_assignment_soft_prediction/river_synthetic_recurring_streams.py`）がRiverの生成器で作る。製品（`federated_learning_experiments`）の全手法（FedSDA・FedDrift・FedAvg）は、これらのデータを使えない。製品の環境（手元のvenv、uraniumの`.venv`）にRiverは無い。
- 何を変えるか: `stagger3`・`agrawal4`・`randomrbf4`を、全手法の新しいデータとして足す。概念の中身（分類の関数、Random RBFの中心の配置）は単一ストリームの版（Riverの定義）と同じにし、標本は製品の既存の合成データ（SEAなど）と同じく、runが貸すNumPyの乱数で作る。

## Introduction
根拠の定義はRiver 0.27.0の`river.datasets.synth`の`STAGGER`・`Agrawal`・`RandomRBF`（単一ストリームの版が使う生成器。WSLの`~/river-env`）。製品はRiverに依存しない。Riverと同じ定義であることは、Riverのある環境のtestで確かめる。
標本の乱数の列はRiverと同じにはしない（製品の乱数の規則に従う）。分類の関数・特徴の生成の分布・中心の配置を同じにする。
全手法で共通の条件（概念の時系列、clientへの分け方、事前学習、評価）は、今のデータと同じにする。既存の10データの振る舞いと値は変えない。

## Boundary Context
- In scope: 3データの定義・生成器、全手法の設定（ネットワーク・学習率・FedDriftのδの既定）、掃引での指定、単一ストリームの版の名前の検査の調整、文書。
- Out of scope: 3データでの本規模の実験とFedDriftのδの調整（このspecの後に、研究として行う）。INSECTS・Electricityなどの実データ。STAGGER・Agrawalの`balance_classes`。

## Requirements

### Requirement 1: STAGGER（`stagger3`）
1.1 The data shall `stagger3`を足す。入力3、概念3、クラス2。
1.2 The generator shall 1標本ごとに、size・color・shapeを、それぞれ0・1・2から等確率に選ぶ（Riverの`STAGGER`の`randint(0, 2)`と同じ分布）。
1.3 The generator shall 概念cのラベルを、Riverの`STAGGER`の分類の関数c（0: sizeが0かつcolorが0なら1、1: colorが2またはshapeが0なら1、2: sizeが1または2なら1）で決める。ラベルの雑音は入れない。
1.4 The generator shall 特徴を、各値を2で割った値（0・0.5・1）にして返す。

### Requirement 2: Agrawal（`agrawal4`）
2.1 The data shall `agrawal4`を足す。入力9、概念4、クラス2。
2.2 The generator shall 9つの特徴（salary、commission、age、elevel、car、zipcode、hvalue、hyears、loan）を、Riverの`Agrawal`の`__iter__`と同じ式・同じ分布・同じ順序で作る（commissionはsalaryが75000未満のときだけ乱数を使う）。
2.3 The generator shall 概念cのラベルを、Riverの`Agrawal`の分類の関数c（0〜3）で、摂動の前の特徴から決める。
2.4 The generator shall ラベルを決めた後に、Riverの`Agrawal`の`perturbation=0.05`と同じ摂動（`_perturb_value`の式・切り詰め・丸め・適用の順序と、hvalueの呼び出しの引数の並びを含む）を特徴へ加える。
2.5 The generator shall 各特徴を、設計書に書く固定の範囲で0〜1へ線形に写して返す（標本やrunから範囲を推定しない）。

### Requirement 3: Random RBF（`randomrbf4`）
3.1 The data shall `randomrbf4`を足す。入力10、概念4、クラス2、中心50。
3.2 The generator shall 概念cの50個の中心（位置、ラベル、標準偏差、重み）を、Riverの`RandomRBF(seed_model=c, n_classes=2, n_features=10, n_centroids=50)`の`_generate_centroids`と同じ値にする（runのseedによらない）。
3.3 The generator shall 1標本ごとに、Riverの`_generate_sample`と同じ分布（重みに比例した中心の選択、各特徴の[-1, 1)の一様、標準正規×中心の標準偏差の大きさ）で特徴を作り、中心のラベルを返す。特徴の尺度は変えない。

### Requirement 4: 3データに共通の生成
4.1 The generators shall 標本の乱数に、runが貸すNumPyの乱数だけを使い、同じseed・同じ概念列なら同じ標本を返す。
4.2 The generators shall 範囲の外の概念・型の違う引数を拒否する（既存の生成器と同じ形）。
4.3 The generators shall 特徴を、SEAと同じくfloat32へ丸めてから返す。
4.4 The data shall 概念の時系列・clientへの分け方・事前学習（概念0の標本）・評価を、今のデータと同じ経路で使う。

### Requirement 5: 全手法の設定と掃引
5.1 The settings shall 3データで、FedSDA・FedDrift・FedAvg・DWM（`dwm-neural-network`。FedAvgの設定を使う）のネットワークと学習率を、今の合成データ（`sea4`など）と同じにする。
5.2 The FedDrift default settings shall 3データのδの既定を0.10にする（公式の`default_deltas`に3データの値は無い。設計書と文書に開示する）。
5.3 The sweep shall 3データを`--datasets`で指定でき、実行の重さの見積りで今の合成データと同じに扱う。

### Requirement 6: 単一ストリームの版との整合
6.1 The single-stream generator shall 製品と同じ名前のデータを拒否せず、製品の定義の概念数・クラス数・特徴数が、単一ストリームの定義と一致することを確かめる（一致しなければ拒否する）。単一ストリームの系列の値は変えない。製品の外の名前としてこれらの名前を使っていた既存の単一ストリームのtestは、製品に無い名前へ替える（読込みが確かめる内容は変えない）。
6.2 When Riverを使える環境でtestを実行するとき, the tests shall 次を確かめる: (i) STAGGER・Agrawalの分類の関数が、Riverの関数と、同じ入力で全て同じラベルを返す（STAGGERは27通りの全ての入力、Agrawalは摂動なしのRiverの生成器の特徴）。(ii) Random RBFの中心が、Riverの中心と一致する。(iii) 3データの各概念で、製品の生成器とRiverの生成器（Agrawalは摂動0.05。Riverの特徴は製品と同じ尺度へ写す）の、クラス1の割合と各特徴の平均の差が、2万標本ずつで、差の標準誤差の4倍以内である。Riverの無い環境では、理由を出してskipする。

### Requirement 7: 既存の値の維持と検証
7.1 The goldens shall 全版のgoldenの値を変えない。`BEHAVIOR_BASELINE_VERSION`は上げない。
7.2 The tests shall 次を確かめる: (i) 要求1〜3の定義（特徴の値の範囲、ラベルの規則、尺度）。(ii) 要求4.1の再現性と4.2の拒否。(iii) 4手法の設定が要求5.1・5.2のとおりになる。(iv) 全体runが3データで4手法とも完了する（各データ1 run、10 client、各client 100標本、集約間隔50）。(v) 掃引の`--datasets`で3データを指定でき（計画の作成）、重さの見積りが今の合成データと同じになる。

### Requirement 8: 文書
8.1 The docs shall `docs/experiments/dataset-characteristics.md`に、3データの定義・根拠（Riverの版とクラス）・特徴の尺度・開示する差（標本の乱数の列はRiverと違う、Agrawalのhvalueの摂動はRiverの引数の並びのまま、δの既定に公式の値は無い）を足す（今の状態の正本）。
8.2 The docs shall `docs/reference/options.md`を直接編集せず、必要なら`python -m tools.generate_option_docs`で再生成する。
