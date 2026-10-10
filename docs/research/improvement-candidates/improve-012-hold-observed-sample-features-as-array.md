# IMPROVE-012. 観測標本の特徴を、配列で持つ

- 記録日: 2026-10-10。種別: メモリ・計算効率。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/data/observed_streams.py`（`ObservedSample.feature_values`）と、それを読む箇所（`runtime/fedsda_run_client.py`・`runtime/initial_model_pretraining.py`の、標本をtensorへ直す箇所、学習データ・評価標本・保留の各owner）。発見spec: mnist-sample-generation。

## 確認した事実

観測標本の特徴は、Pythonのfloatのtupleで持つ。合成データ（2〜3特徴）では小さいが、MNISTは784特徴である。Windowsの基準環境での実測（2026-10-10）:

- 特徴ごとにfloatを作ると、1標本あたり約25KB。旧の既定の規模（client 10×標本5000＝5万件）で、約1.26GB。
- mnist-sample-generationでは、画素の値（0〜255）ごとのfloatを256個だけ作り、標本のtupleはその参照を並べる形にした。1標本あたり約6.3KB、5万件で約316MB。並列14 workersなら、約4.4GB。
- 旧実装は、標本ごとにfloat32のtensor（784×4バイト＝約3.1KB＋tensorの管理分）を持つ。
- clientは、標本を処理するたびに、tupleを1行のtensorへ直す（784特徴で約57マイクロ秒。学習・評価でも、標本の集まりをtensorへ直す）。

## 案

`ObservedSample.feature_values`を、書込み不可のfloat32の配列（またはtensor）にする。MNISTでは、読み込んだ画素の行から、標本ごとに784×4バイトで持てる。

## 挙動への影響

値は変わらない（float32の値を、そのまま持つ）。ただし、観測標本・実行の結果・各ownerのsnapshotの等値の比較（多くのtestと、結果の型の検査が使う）が、tupleの`==`に依っている。配列にすると、等値の定義を、型ごとに書く必要がある。影響は、観測標本を持つ全部のownerと、そのtestに及ぶ。

## 必要な検証

全体runの対照（全dataset）と、goldenの照合（3ケース）が、変わらず成功すること。MNISTの既定の規模での、メモリと時間の実測。採否・変更commitは未定。
