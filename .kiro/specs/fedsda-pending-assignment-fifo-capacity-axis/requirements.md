# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: circle2で、FedSDAの保有モデルの学習データに他の概念の標本が混ざり、FedDriftに届かない原因になっている（研究バックログの8）。混ざりの大部分は、帰属の保留（FIFO、容量30件）から現在のモデルへ確定された標本で、警報までの遅れが容量より長いために起きる。手元の試行で、容量を60にすると（Fixed-Shareの時間尺度も60）、circle2が間隔50・200で5 seedとも上がり、sine2・sea2はほぼ変わらなかった。容量は、今は掃引のコマンドで選べない。
- 何を変えるか: 掃引のコマンドで、FedSDAの構成の帰属の保留の容量を、ハイパーパラメータの軸として並べられるようにする。既定の動作と値は変えない。

## Introduction
2026-10-15の会話で、ユーザーは、容量を構成名でなくハイパーパラメータの指定（掃引の軸）で選ぶこと、引数の名前`--fedsda-pending-assignment-fifo-capacities`を決めた。Fixed-Shareの時間尺度は、今の検査（容量と同じ値）を保ち、軸の値で両方を替える。将来は別々に指定できる方がよい（IMPROVE-015）。

## Boundary Context
- 対象: 掃引の計画（軸）、コマンドの引数、一覧の表の列、掃引の記録の形式の版、文書。
- 対象外: 帰属の保留とFixed-Shareの処理、同値の検査（IMPROVE-015で扱う）、既定の値（30）、FedDrift・FedAvg。
- 仮定: 2つのFedSDAの軸（学習batchの抜き出し方と、この容量）を両方指定したときは、全部の組を作る。

## Requirements
### Requirement 1: 容量を軸として並べられる
**Objective:** As a 研究者, I want 掃引のコマンドで、帰属の保留の容量を複数並べて実行できること, so that 研究室サーバで、1回の掃引で比べられる。
#### Acceptance Criteria
1. The sweep command shall `--fedsda-pending-assignment-fifo-capacities`で、正の整数の一覧を受け、FedSDAの構成の各runを、容量ごとに作る。各runの`pending_assignment_buffer_capacity_samples`と`fixed_share_weight_redistribution_time_scale_samples`は、その値になる（ほかの設定は変えない）。
2. If 値が重複する・正のbuiltin intでない、またはFedSDAの構成を含まない計画, the sweep command shall 引数の誤り（終了コード2）として拒否する。作ったrun設定の検査が拒否する値（例: 警報を待つ扱いで、容量が変化区間の最小件数より小さい）も、終了コード2になる。
3. When 学習batchの抜き出し方の軸と、この軸の両方を指定したとき, the sweep command shall 全部の組（抜き出し方の順の中で、容量の順）を作る。
4. When この軸を指定しないとき, the sweep command shall 今と同じ計画を作る（FedSDAのrunの識別子も変わらない）。既定の値（30）を指定したrunは、指定しないrunと同じ設定・同じ識別子になる。

### Requirement 2: 一覧の表と記録
**Objective:** As a 研究者, I want 結果の表で、各runの容量が分かること, so that 分析で比べられる。
#### Acceptance Criteria
1. The run summary table shall FedSDAの構成の行に、列`pending_assignment_fifo_capacity`（そのrunの帰属の保留の容量）を持つ。
2. The sweep record shall 計画に新しい項目を持ち、掃引の記録の形式の版を5へ上げる（動作の基準版とは別の番号）。

### Requirement 3: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存のgoldenと照合が値を変えずに通ること, so that 既定の動作が変わっていないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenを、値を変えずに通す（run設定の保存の形は変わらない）。既存の掃引のtestは、掃引の記録の形式の版（4→5）と、記録の計画の新しい項目（既定`None`）と、コマンドの引数の一覧（引数の集合を完全に照合するtest）の期待だけを変え、ほかの期待は変えずに通す。

### Requirement 4: 文書
**Objective:** As a 研究者, I want 文書が新しい軸を説明すること, so that サーバのコマンドを正しく書ける。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書に、軸、時間尺度も同じ値で替わること、容量の目安と制約、2つの軸の組合せを書く。
