# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装は、1 runの結果を、新実装の名前で保存できる（fedsda-experiment-run-artifacts）。保存する項目は、照合済みの33指標と31の配列に当たるものと、ラウンドごとのモデル数ほか。旧実装の出力のうち、「記録から導けるが、未照合」の項目（指標124、配列83。docs/research/legacy-run-output-migration.md）は、まだ保存していない。

ユーザーの決定（2026-10-11）: 未照合の区分は、全部、新実装で導出して、旧の値と照合する（旧の結果を、新の名前へ変換するためにも）。保存する名前は、短さより、分かりやすさと、実態との一致を優先する。主担当が、保存の形式の名前を見直して、改善の余地があれば適用する。

調査で確かめたこと（旧実装。固定）:

- 旧の指標の計算（federated_drift_experiment/metrics.pyの`compute_metrics`）は、精度と検出の指標（照合済み）のほかに、検出の遅れ、対応づかなかった切替の分類（重複・早期・遅延・単独）、警報の適合率・再現率、変化点の推定の誤差、適応の区分ごとの件数と適合率、候補の判定の統計（採用率、変更に対応した数、件数や損失の差の平均、棄却の理由ごとの数）、サーバの統合による付け替えの数を返す。
- 対応づけは、clientごとに、変更位置を順に見て、変更位置以上・変更位置＋許容遅延以下の、まだ使っていない最初のイベントを対応づける（`match_events`）。切替と警報は、位置の昇順。適応と判定は、記録の順。
- 配列（`_save_raw_run`）: 警報ごとの位置・推定した変化点・検出器の候補の開始位置、適応ごとの検出器の名前・推定した変化点、判定ごとの件数・参照モデル・5つの平均損失・変更に対応したか、標本ごとの検出器の値（log e）。
- 新の記録には、これらの入力が、全部ある（警報の記録、適応の記録、判定の記録、損失の監視の値）。

## Introduction

研究者が、概念の変更への応答（検出、警報、適応、候補の判定）の指標と記録を、新実装で導出・保存し、旧実装の値と一致することを確かめられるようにする。あわせて、保存する表と列の名前を、時点と意味が分かる名前へ直す。

## Boundary Context

- **In scope**: 保存する表・列の名前の見直し（前のspecの形式の変更）。検出・警報・適応・候補の判定の指標の計算と、指標への追加。警報・適応・判定の表の列の追加、損失の監視の値の表。実旧との照合。移行の一覧の更新。
- **Out of scope**: モデル別の診断・登録・クラスタリングの指標、予測の集計、クロス評価の配列（後のspec）。旧の結果の変換（(A2-変換)）。未移植の診断と所要時間。
- **Adjacent expectations**: 全体runの結果と、既存の指標の値は、変えない。固定旧実装とgoldenは、変えない。保存の形式を使う既存のfileは、ない（形式の版は、1のまま）。

## Requirements

### Requirement 1: 保存する表と列の名前

#### Acceptance Criteria

1. The ラウンドごとの表 shall 1つの表`server_round`にまとまり、同期の後のモデル数（`global_model_count_after_synchronization`、`client_held_model_count_after_synchronization`）と、ローカルの処理・同期の間のモデルの計算（`local_processing_*`、`synchronization_*`）の列を持つ。
2. The 標本ごとの保有モデル数の列 shall `sample_prediction.held_model_count_at_prediction`という名前になる。概念の変更の位置の列 shall `concept_change.first_new_concept_sample_index`という名前になる。学習帰属の切替の位置の列 shall `training_model_switch.switch_sample_index`、サーバの同期のラウンドを指す列 shall `server_round_index`（クラスタリングの2つの表）と`model_registration.registered_server_round_index`という名前になる（独立レビューの指摘で足した）。
3. The 名前の変更 shall 値を変えない（前のspecの照合が、新しい名前で、そのまま成功する）。

### Requirement 2: 指標の計算

#### Acceptance Criteria

1. When 変更位置とイベントの位置を与えたとき, the Change Response Metric Calculations shall 対応づいたイベントと遅れ、対応づかなかったイベントの分類（重複・早期・遅延・単独。この順で、最初に当てはまる区分）を返す。結果は、実旧の関数（`match_events`・`_classify_unmatched_events`）と一致する。
2. When 警報の位置と、警報ごとの推定した変化点を与えたとき, the 計算 shall 変更に対応した警報の、推定の誤差（推定−変更位置）を返す。結果は、実旧の`_change_point_errors`と一致する。
3. If 引数の型や値が不正なとき（負の位置、長さの違う列）, the 計算 shall 拒否する。

### Requirement 3: 指標の導出

#### Acceptance Criteria

1. When 全体runの結果と参加者を与えたとき, the FedSDA Run Metric Derivation shall これまでの指標に加えて、次を返す: 学習帰属の切替の、平均の遅れと、対応づかなかった切替の分類ごとの数。警報の検出の指標。変化点の推定の誤差（平均絶対誤差、平均の偏り、件数）。適応の区分ごとの件数と、再利用・採用の適合率、現行モデルの維持・別モデルの再利用の選択の数、サーバの統合による付け替えの数。候補の判定の統計（採用率、変更に対応した数——全体・採用・棄却——、採用の適合率、学習区間・検証の件数と確定までの遅れの平均、損失の差の平均——採用・棄却・対応・非対応の別、全区間と後半——、理由ごとの数、参照の損失の超過の平均）。監視の値の最大。
2. The 追加した指標 shall goldenの3ケースと、小さい条件で、実旧の`compute_metrics`ほかの値と、全部の項目で一致する。
3. The 既存の指標 shall 値が変わらない。

### Requirement 4: 記録の表

#### Acceptance Criteria

1. The 記録の表 shall 次を持つ: 警報の表（client、警報の位置、推定した変化点、検出器の候補の開始位置）。標本ごとの監視の値の表。適応の表の、検出器の名前と、推定した変化点の列。候補検証の判定の表の、検出器の名前、学習区間の件数、検証の件数、比較した参照モデル、5つの平均損失、変更に対応したか、の列。
2. The 追加した列 shall goldenの3ケースで、実旧が保存した配列と、値で一致する（欠損の表し方の違いは、対応を決めて照合する）。

### Requirement 5: 記録

#### Acceptance Criteria

1. The 移行の一覧 shall このspecで照合した項目の、旧→新の対応と、新の記録だけでは導けなかった項目（あれば、理由）を持つ。
