# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装は、1 runの結果を、新実装の名前で保存でき、概念の変更への応答の指標と記録まで、旧実装と照合してある（fedsda-change-response-metrics-and-records）。旧実装の出力のうち、「記録から導けるが、未照合」の残りは、指標71・配列64（docs/research/legacy-run-output-migration.md）。

ユーザーの決定（2026-10-11）: 未照合の区分は、全部、新実装で導出して、旧の値と照合する。保存する名前は、短さより、分かりやすさと、実態との一致を優先する。

このspecが扱うまとまり: モデル別の診断・登録・クラスタリング（指標26、配列17）。

調査で確かめたこと（旧実装。固定）:

- ラウンドごとのグローバルモデル数の、平均・最大・合計（`mean_model_count`・`max_model_count`・`model_count_auc`。experiment.pyの`_add_telemetry_results`）。
- モデル別の学習量（`_add_model_diagnostic_results`）: clientが最後に保有するモデルごとに、保持している学習標本の数、学習に使った標本の延べ数、パラメータ更新の回数を、全clientで足し、モデルの間の、合計・平均・最小・変動係数（母標準偏差÷平均。平均が0なら0）を取る（`model_assigned_samples_*`・`model_training_examples_*`・`model_optimizer_steps_*`）。モデルがなければ、すべて0。
- クラスタリングの判定と、真の概念の一致の比較（servers/clustering.pyの`clustering_oracle_diagnostic_summary`）: 真の概念の一致・不一致が分かる対（両方のモデルに、割当の最多の真の概念が一意にある対）について、同じクラスタにしたか、の4通りの数と、適合率・再現率・F1、損失の距離と、概念固有部のパラメータの距離の、ROC-AUC（小さい値を、同じ概念の証拠とみなす。有限の値だけ。一致の対か、不一致の対が、なければNaN）。
- 配列（`_save_raw_run`）: 登録したclient、モデルごとの、予測で最大重みになった標本の数。クラスタリングのモデルごとの、最も近い相手とその距離、クラスタの大きさ、クラスタの中の距離の最大、距離を持つ対の数と、可能な対の数。対ごとの、損失の距離、判定の値、真の概念の一致（分からなければ−1）、概念固有部のパラメータの距離。clientと保有モデルごとの、上の3つの学習量。
- 新の記録には、これらの入力が、全部ある（`GlobalModelRegistrationRecord.registering_client_id`、`ModelClusteringObservation`と`ModelPairClusteringObservation`の全field、`ModelTrainingSampleStore`、`ModelTrainingAndAssignmentCountsStore`、サーバの同期の配布の記録）。

## Introduction

研究者が、モデル別の診断（モデル数の推移のまとめ、モデル別の学習量、クラスタリングの判定と真の概念の一致の比較）の指標と、登録・クラスタリング・保有モデルの記録を、新実装で導出・保存し、旧実装の値と一致することを確かめられるようにする。

## Boundary Context

- **In scope**: 上の指標の計算と、指標への追加。登録・クラスタリングの表の列の追加、clientの保有モデルごとの表。実旧との照合。移行の一覧の更新。
- **Out of scope**: 予測の集計（`routing_*`）、クロス評価の配列と、モデル対の予測の相補性（`cross_evaluation_*`・`model_pair_*`）（後のspec）。旧の結果の変換（(A2-変換)）。未移植の診断と所要時間。
- **Adjacent expectations**: 全体runの結果と、既存の指標・表の値は、変えない。固定旧実装とgoldenは、変えない。保存の形式を使う既存のfileは、ない（形式の版は、1のまま）。

## Requirements

### Requirement 1: 指標の計算

#### Acceptance Criteria

1. When 値の列を与えたとき, the Model Diagnostic Metric Calculations shall 合計・平均・最小・変動係数（母標準偏差÷平均）を返す。列が空なら、すべて0。平均が0以下なら、変動係数は0。
2. When 同じ概念の対の値と、違う概念の対の値を与えたとき, the 計算 shall 小さい値を同じ概念の証拠とみなしたROC-AUC（同じ概念の値が小さい組の割合。等しい組は半分）を返す。有限でない値は、数えない。どちらかの列に、有限の値がなければ、NaN。
3. When 対ごとの、同じクラスタにしたか、と、真の概念が一致するか、を与えたとき, the 計算 shall 4通りの数と、適合率・再現率・F1（分母が0なら0）を返す。
4. If 引数の型や値が不正なとき, the 計算 shall 拒否する。

### Requirement 2: 指標の導出

#### Acceptance Criteria

1. When 全体runの結果と参加者を与えたとき, the FedSDA Run Metric Derivation shall これまでの指標に加えて、次を返す: サーバの同期のラウンドごとの、同期の後のグローバルモデル数の、平均・最大・合計。保有モデル別の、保持している学習標本の数・学習に使った標本の延べ数・パラメータ更新の回数（全clientの合計）の、モデルの間のまとめ。クラスタリングの判定と、真の概念の一致の比較（4通りの数、適合率・再現率・F1、2つの距離のAUC）。
2. The 追加した指標 shall goldenの3ケースで、実旧の実行結果の値と、全部の項目で一致する。モデル別の学習量と、クラスタリングの比較は、小さい条件でも、実旧の`_add_model_diagnostic_results`の値と一致する。NaNは、新旧ともNaNであることで、一致とする。
3. The 既存の指標 shall 値が変わらない。

### Requirement 3: 記録の表

#### Acceptance Criteria

1. The 記録の表 shall 次を持つ: モデルの登録の表の、登録したclient（初期モデルは−1）と、そのモデルが予測で最大重みになった標本の数。クラスタリングの、モデルごとの表の、最も近い相手（なければ−1）とその距離（なければNaN）、クラスタのモデル数、クラスタの中の距離の最大（なければNaN）、距離を持つ対の数、可能な対の数。対ごとの表の、損失の距離、判定の値、真の概念の一致（一致1、不一致0、分からなければ−1）、概念固有部のパラメータの距離。clientの保有モデルごとの表（client、モデル、保持している学習標本の数、学習に使った標本の延べ数、パラメータ更新の回数。実行の終わりの値）。
2. The 追加した列 shall goldenの3ケースと、小さい条件で、実旧が保存した配列と、値で一致する。

### Requirement 4: 記録

#### Acceptance Criteria

1. The 移行の一覧 shall このspecで照合した項目の、旧→新の対応と、新の記録だけでは導けなかった項目（あれば、理由）を持つ。
