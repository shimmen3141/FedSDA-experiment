# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 旧実装の出力のうち、「記録から導けるが、未照合」の残りは、指標45・配列47（docs/research/legacy-run-output-migration.md）。応答の指標と、モデル別の診断までは、新実装で導出・保存して、旧と照合してある。

ユーザーの決定（2026-10-11）: 未照合の区分は、全部、新実装で導出して、旧の値と照合する。保存する名前は、短さより、分かりやすさと、実態との一致を優先する。

このspecが扱うまとまり: クロス評価の記録と、モデル対の予測の相補性（指標5、配列28）。

調査で確かめたこと（旧実装。固定）:

- クロス評価（servers/clustering.pyの`evaluate_models_on_clients`）は、候補のモデルと、対象のモデル（標本を持つ側）の組ごと、対象のモデルを保有するclientごとに、clientの評価を1回行う。最終構成のサーバは、評価のたびに、診断の記録を1行足す（`cross_evaluation_diagnostics`: ラウンド、client、候補、対象、標本数、損失の和、損失の二乗和、正誤の4つの数——候補だけ正解、対象だけ正解、両方正解、両方不正解。正誤を比べない評価——候補と対象が同じモデル——では、4つとも−1）。
- 最終構成は、クラスごとの正誤も集める（`cross_evaluation_class_diagnostics`: 上の識別に、クラスと、クラスの標本数と、4つの数）。
- 正誤を比べた評価は、別の一覧（`pair_prediction_diagnostics`）にも入り、配列`model_pair_*`（候補、対象、標本数、4つの数）と、指標`model_pair_*`（`pair_diagnostic_summary`: 評価の数、標本数の合計、正誤が食い違う標本の割合、良いほうのモデルに対して標本ごとに選べたら得られる上限の改善の割合——評価ごとの、min(候補だけ正解, 対象だけ正解)の和÷標本数——、両方正解の割合。標本数の合計が0なら、評価の数を含めて、すべて0）になる。
- 新の記録（`ClientCrossEvaluationRecord`）は、旧の1行と、同じ内容を持つ（正誤を比べなかった評価では、正誤の数がNone。クラスごとの数は、クラスの昇順の列）。

## Introduction

研究者が、クロス評価の、clientの評価ごとの記録と、モデル対の予測の相補性の指標を、新実装で導出・保存し、旧実装の値と一致することを確かめられるようにする。

## Boundary Context

- **In scope**: モデル対の予測の相補性の指標の計算と、指標への追加。clientの評価ごとの表と、クラスごとの表。実旧との照合。移行の一覧の更新。
- **Out of scope**: 予測の集計（`routing_*`。後のspec）。旧の結果の変換（(A2-変換)）。未移植の診断と所要時間。
- **Adjacent expectations**: 全体runの結果と、既存の指標・表の値は、変えない。固定旧実装とgoldenは、変えない。保存の形式を使う既存のfileは、ない（形式の版は、1のまま）。

## Requirements

### Requirement 1: 指標の計算

#### Acceptance Criteria

1. When clientの評価の記録の列を与えたとき, the Cross Evaluation Metric Calculations shall 正誤を比べた評価について、評価の数、標本数の合計、正誤が食い違う標本の割合、標本ごとの選択で得られる上限の改善の割合、両方正解の割合を返す。標本数の合計が0なら、評価の数を含めて、すべて0（旧と同じ）。
2. If 引数の型が不正なとき, the 計算 shall 拒否する。

### Requirement 2: 指標の導出

#### Acceptance Criteria

1. When 全体runの結果と参加者を与えたとき, the FedSDA Run Metric Derivation shall これまでの指標に加えて、モデル対の予測の相補性の指標を返す。
2. The 追加した指標 shall goldenの3ケースと、小さい条件で、実旧の値と、全部の項目で一致する。
3. The 既存の指標 shall 値が変わらない。

### Requirement 3: 記録の表

#### Acceptance Criteria

1. The 記録の表 shall 次を持つ: clientの評価ごとの表（サーバの同期のラウンド、client、候補のモデル、対象のモデル、評価した標本数、損失の和、損失の二乗和、正誤の4つの数——比べなかった評価では−1）。クラスごとの表（上の識別に、クラス、クラスの標本数——`class_evaluated_sample_count`——、正誤の4つの数）。
2. The 表 shall goldenの3ケースと、小さい条件で、実旧が保存した配列（`cross_evaluation_*`・`cross_evaluation_class_*`）と、値で一致する。旧の`model_pair_*`の配列は、clientの評価ごとの表の、正誤を比べた行と一致する。

### Requirement 4: 記録

#### Acceptance Criteria

1. The 移行の一覧 shall このspecで照合した項目の、旧→新の対応と、新の記録だけでは導けなかった項目（あれば、理由）を持つ。
