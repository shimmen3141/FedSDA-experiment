# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装の全体runは、3つの設定（実行の枠の設定`StreamProtocolExecutionSettings`、参加者の設定の束`FedsdaRunParticipantSettings`、指標の設定`RunMetricSettings`）を、呼出し側が別々に組み立てて渡す。最終構成の値（約40個）を組み立てているのは、testのhelperだけで、source側には、最終構成の既定値がない。datasetごとの学習率と隠れ層の幅（mnistは1e-3、(1568,)。ほかは1e-2、(32, 32)）も、testが渡している。1 runの条件を、保存できる形（辞書）にする手段もない。

ユーザーの決定（2026-10-11）: 次のspecの順序は、(A)新実装で実験を実行するための入口→(B)FedDriftの移植→(C)残る診断→(D)旧実装を外す準備。(A)は、(A1)完全なrun設定、(A2)1 runの実行と結果の保存、(A3)掃引・並列・コマンド、に分ける（主担当の案）。このspecは(A1)。

調査で確かめたこと:

- 旧の最終構成の既定値は、`federated_drift_experiment/config.py`の初期値に、最終構成の固定設定（docs/overview/proposed-method.mdの表。tests/test_proposed_regression.pyの`ALGORITHM`）を重ねたもの。旧の回帰testの条件（`COMMON`・`CASES`）は、これを縮小したもので、client数・標本数・事前学習の件数・概念の変更の確率と最小の間隔・回復の窓が違う。
- 旧のdatasetの定義（`data/specs.py`）: 隠れ層の幅は、mnistが(1568,)、ほかが(32, 32)。学習率は、mnistが1e-3（通常の学習率と、新規モデルの学習率の両方に優先する）、ほかは、設定の値（どちらも1e-2）。
- 検出器の賭け率の既定は、旧の`drift_detectors/e_detector.py`の`DEFAULT_LAMBDAS`（0.05、0.1、0.2、0.4、0.8）。
- 新実装には、機能の組合せを検証する部分型（`configuration/run_settings.py`の`ValidatedExperimentRunSettingsSubset`）がある。統合の設定（`ModelConsolidationSettings`）は、この部分型にはあるが、参加者の設定の束には入っていない（サーバが、最終構成の方式だけを実装している）。

## Introduction

研究者が、最終構成のFedSDAの1 runの条件を、1つの検証済みの値として作り、渡し、保存できるようにする。

## Boundary Context

- **In scope**: 1 runの完全な設定の型。最終構成の既定値を返す関数（datasetごとのモデルの既定を含む）。設定を、保存用の辞書にする変換。
- **Out of scope**: 1 runの実行と結果の保存（A2）。掃引・並列・コマンド・重複の確認（A3）。辞書から設定へ戻す変換。設定の束の中の、仮の置き場所（`FedsdaRunClientScalarSettings`ほか）の作り直し（挙動を変えない移植の後で判断する）。FedDriftの設定（B）。
- **Adjacent expectations**: 全体runの結果は、変えない。既存の設定型は、変えない。ただし、clientの設定の束（`FedsdaRunClientSettings`）へ、検出の方式と検出器の表示名の一致の検査を足す（独立レビューの指摘。これまで、食い違う表示名を受け入れていた）。

## Requirements

### Requirement 1: 完全なrun設定

#### Acceptance Criteria

1. The FedSDA Run Settings shall 1 runに要る設定（実行の枠の設定、参加者の設定の束、統合の設定、指標の設定）を、1つの不変の値として持つ。
2. When 生成されたとき, the FedSDA Run Settings shall 各部分の型と値、機能の組合せ（既存の部分型の検証）、部分の間で一致すべき値（下の3）を確かめる。
3. If 最終構成として一致すべき値が、食い違うとき（検出の方式と、記録に残す検出器の表示名。モデルの再利用の許容量と、クラスタリングの同じクラスタの判定の上限。既存の束が確かめている組——Fixed-Shareの時間尺度と保留の容量、候補の最小改善量と早期終了の最小改善量、ローカル学習と候補の学習のbatchの件数——）, the FedSDA Run Settings shall 拒否する（frozenを回避して組み立てた値を含む）。
4. The 完全なrun設定 shall 全体runの実行（実行の枠の設定と、参加者のfactory）と、指標の導出へ、そのまま渡せる。

### Requirement 2: 最終構成の既定値

#### Acceptance Criteria

1. When 実行条件（dataset、seed、client数、標本数、集約間隔）を与えたとき, the 最終構成の設定の関数 shall 最終構成の既定値を持つ、完全なrun設定を返す。
2. The 既定値 shall 旧の設定の初期値に、最終構成の固定設定を重ねた値と、全部の項目で一致する（datasetごとの隠れ層の幅と学習率を含む）。
3. The 関数 shall 定義のある全部のdatasetについて、モデルの既定（隠れ層の幅、学習率）を持つ。
4. When 既定値の一部を変えたいとき, the 利用者 shall 返された値の一部を置き換えた、新しい完全なrun設定を作れる（置き換えた後も、1.2の検証が行われる）。
5. The goldenの3ケースの条件 shall 最終構成の既定値から、旧の回帰testが縮小した項目だけを置き換えて作った設定と、一致する。

### Requirement 3: 保存用の表現

#### Acceptance Criteria

1. When 完全なrun設定を与えたとき, the 変換 shall 全部の項目の名前と値を持つ、入れ子の辞書（JSONにできる値だけ）を返す。設定の型の名前を含む（同じ場所に、種類の違う設定——AdamとSGD——が入りうるため）。
2. The 変換 shall 同じ設定からは、同じ辞書を返し、どれか1つの値が違う設定からは、違う辞書を返す。
3. If JSONにできない値（設定の型でも、数値・文字列・真偽値・Noneでも、その並びでもない値。非有限のfloat）を含むとき, the 変換 shall 拒否する。
