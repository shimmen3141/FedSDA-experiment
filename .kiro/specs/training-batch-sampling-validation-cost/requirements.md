# Requirements Document

## Project Description (Input)

誰の問題か: 新実装で、本番の規模（client 10、標本5000、1構成120 run）の実験を実行する研究者。

現在の状況（2026-10-11の実測）: 新実装の1 runは、旧実装より遅い（client 10・標本500・sine2で、約60秒対17秒）。本番の規模の1 runは、手元で、68分たっても終わらなかった（旧実装は、サーバで、120 runが2時間5分）。プロファイル（標本300）では、実行時間の約45%が、学習のたびに、保持している学習標本の全部を検査し直す処理（`learning/training/held_model_training_batch_sampling.py`の`_validate_sampling_request`。標本ごとに、型・形・有限性を確かめる）だった。保持する標本は、runの間、増えていくので、1回の学習の費用が、保持する標本の数に比例し、全体は、標本数の2乗に近づく。

この検査は、完了済みのspec（held-model-training-batch-sampling）が、意図して決めた契約である（抽出に入らない標本も、乱数を進める前に、全部検査する）。旧実装は、この検査をしない（`random.sample`で抽出するだけ）。

## Introduction

研究者が、新実装で、本番の規模の実験を、実用の時間で実行できるようにする。学習標本の抽出の、1回あたりの費用を、保持する標本の数に依らないようにする。正しい入力での結果は、変えない。

## Boundary Context

- **In scope**: 学習標本の抽出（`sample_training_batches_for_held_models`）の、標本の中身の検査の範囲と時点の変更。速度の実測。
- **Out of scope**: ほかの検査の重複（分類器の入力、共同更新の入力、予測のtensor。プロファイルで、それぞれ3〜4%。このspecの後に、もう一度測って判断する）。学習標本を入れる側（保持のowner）の検査の追加。
- **Adjacent expectations**: 正しい入力での、抽出の結果と、乱数の進み方は、変わらない。全体runの結果（goldenの3ケース、実旧との照合）は、変わらない。

## Requirements

### Requirement 1: 検査の範囲と時点

#### Acceptance Criteria

1. When 学習標本を抽出するとき, the Held Model Training Batch Sampling shall 入れ物（保有モデルのID、モデルごとの標本の列の型、IDの重複、batchの件数、乱数の生成器）を、乱数を進める前に検査する（これまでと同じ）。
2. When 抽出したとき, the 抽出 shall 抽出した標本の中身（型、tensorの種類・形・有限性、同じbatchの中の特徴数の一致）を、これまでと同じ基準で検査する。抽出に入らなかった標本の中身は、検査しない。
3. If 抽出した標本が不正なとき, the 抽出 shall 乱数の生成器の状態を、呼出しの前へ戻して、拒否する。入力は、変えない。
4. The 1回の抽出の、標本の中身の検査の回数 shall 保持する標本の数に依らず、抽出する件数で決まる。

### Requirement 2: 結果を変えない

#### Acceptance Criteria

1. The 正しい入力での、抽出の結果と、呼出しの後の乱数の状態 shall これまでと同じである（実旧の抽出との照合が、そのまま成功する）。
2. The 全体runの結果 shall 変わらない（goldenの3ケースと、小さい条件の、実旧との照合が、そのまま成功する）。

### Requirement 3: 速度の記録

#### Acceptance Criteria

1. The 記録 shall 変更の前後の、同じ条件の実行時間と、プロファイルの上位を持つ。本番の規模の1 runの時間を、測って書く。
