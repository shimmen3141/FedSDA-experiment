# Design Document

## Overview
掃引の計画（`ExperimentSweepPlan`）へ`fedsda_pending_assignment_fifo_capacities: tuple[int, ...] | None = None`を足し、`build_planned_experiment_runs`で、FedSDAのrun設定の変種を作るときに、学習batchの抜き出し方の軸の後に、この軸を掛け合わせる。コマンドの引数、一覧の表の列、掃引の記録の形式の版を足す。帰属の保留とFixed-Shareの処理は変えない。

## Boundary Commitments
### This Spec Owns
- `sweep/experiment_sweep_plan.py`: 計画の項目と検査、変種の作り方（2つのFedSDAの軸の組）。
- `sweep/experiment_sweep_command.py`: 引数`--fedsda-pending-assignment-fifo-capacities`（`nargs="+"`、`type=int`）。
- `sweep/experiment_sweep_records.py`: 列`pending_assignment_fifo_capacity`、`SWEEP_RECORD_FORMAT_VERSION`を5へ。
- testの期待（記録の形式の版4→5）と、新しいtest `tests/refactoring/test_fedsda_pending_assignment_fifo_capacity_axis.py`。
- 文書: `docs/experiments/experiment-sweep.md`、研究バックログの8、再開案内。
### Out of Boundary
帰属の保留・Fixed-Shareの処理と同値の検査、既定の値、FedDrift・FedAvg、run設定の保存の形。
### Allowed Dependencies
新しい依存は無い（既存の`dataclasses.replace`と、計画のmoduleが既に読む設定の型だけ）。依存境界のtestは変える必要がない見込み。
### Revalidation Triggers
掃引の計画のキー（掃引の記録）、一覧の表の列。

## Architecture
### 計画
- 項目`fedsda_pending_assignment_fifo_capacities: tuple[int, ...] | None = None`（抜き出し方の軸の次）。`__post_init__`で、既存の`_validate_distinct_values`（builtin intの重複なし）に加え、全部が1以上であることを確かめる（`ValueError`）。実効の下限は2（Fixed-Shareの時間尺度の下限が2のため、容量1は、作ったrun設定の検査で`RunSettingsValidationError`になる）。boolは`_validate_distinct_values`の型の検査（`type(value) is int`）で拒否される。
- `build_planned_experiment_runs`の、`FedsdaRunSettings`の分岐: 変種を`(run_settings,)`から始め、抜き出し方の軸があれば、各変種に抜き出し方を当てた組（今の処理）、容量の軸があれば、さらに各変種に容量を当てた組にする（抜き出し方の順の中で、容量の順）。容量を当てるのは、`run_client_settings`の`training_data_assignment_settings.pending_assignment_buffer_capacity_samples`と`prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples`を同じ値へ替える`replace`。2つの部分の設定を、それぞれ先に`replace`で作り、その2つを`run_client_settings`の1回の`replace`へ同時に渡す（`FedsdaRunClientSettings.__post_init__`と、`FedsdaRunSettings`が作る全体の設定の検査は、構築のたびに同値を確かめるので、片方ずつ`run_client_settings`を作り直すと、途中の状態で必ず失敗する）。作った設定の検査が値を拒否すれば、`RunSettingsValidationError`（コマンドは終了コード2）。
- 今の分岐の条件（`type(run_settings) is FedsdaRunSettings and sweep_plan.fedsda_training_sample_selections is not None`）を、「FedSDAのrun設定で、どちらかの軸がある」へ変え、抜き出し方の軸が無いときは、抜き出し方の解析を呼ばない。
- どちらかの軸があり、FedSDAのrun設定が1つも無ければ、指定した軸の名前で`ValueError`（今の抜き出し方の軸と同じ。計画の最後の検査を、両方の軸に対応させる）。
- 軸が無ければ、今と同じ1つ。既定の値（30）を当てた変種は、元の設定と等しいので、識別子も同じ。

### コマンド
`--fedsda-pending-assignment-fifo-capacities`（`nargs="+"`、`type=int`、`metavar="CAPACITY"`）。説明: 「FedSDAの構成の、帰属の保留（FIFO）の容量（標本数）。Fixed-Shareの時間尺度も同じ値にする。FedSDAの構成を選んだときだけ」。整数でない値は、argparseが終了コード2で拒否する。計画の`ValueError`も、今の仕組みで終了コード2。

### 一覧の表
FedSDAの行（`run_settings.settings_type == "FedsdaRunSettings"`）に、列`pending_assignment_fifo_capacity`を足す。値は、要約の`run_settings`→`run_participant_settings`→`run_client_settings`→`training_data_assignment_settings`→`pending_assignment_buffer_capacity_samples`。この項目は、要約の形式の版1から在る（要約のrun設定は、どの版も`convert_settings_to_plain_mapping(run_settings)`で書き、版2で足したのは来歴だけ。項目の名前は、FedSDAの移植のときから変わっていない）。前の形式の要約から表を作る既存のtestの補助（`test_experiment_sweep.py`の`save_as_previous_format`）で、この列が埋まることを確かめる。

### 掃引の記録
計画（`asdict(sweep_plan)`）に新しい項目が増えるので、`SWEEP_RECORD_FORMAT_VERSION`を5へ上げる。文書の、記録の形式の版の説明へ「記録の形式の版5から、帰属の保留の容量の軸を持つ」を足す（動作の基準版5と取り違えないよう、文書とtestでは、常に「記録の形式の版」と書く）。

## Validation Design
`tests/refactoring/test_fedsda_pending_assignment_fifo_capacity_axis.py`:
1. 計画: 容量の軸で、FedSDAのrunが容量ごとに作られ、各runの2つの値がその値で、それ以外の設定は元と同じ（保存の形で比べる）。FedDrift・FedAvgのrunは増えない。
2. 既定と同じ: 軸を指定しない計画と、`(30,)`を指定した計画で、識別子が同じ。
3. 組: 抜き出し方の軸（2値）と容量の軸（2値）で、FedSDAの1つの条件から4つのrunが、抜き出し方の順の中で容量の順に作られる。
4. 拒否: 重複、0、負、bool、FedSDAの構成を含まない計画は`ValueError`または`TypeError`。容量1は、時間尺度の下限（2）で、作ったrun設定の検査が`RunSettingsValidationError`。既定の`fedsda`（警報を待つ扱い）で、容量が変化区間の最小件数（5）より小さい値（4）も`RunSettingsValidationError`。警報を即時にする構成（`fedsda-classwise-adwin-detection`）では、容量4は受理される。コマンドでは、拒否はどれも終了コード2。
5. コマンドと表: 小さい掃引で、`--fedsda-pending-assignment-fifo-capacities 30 60`の2 runが保存され、`run_summary.csv`の列`pending_assignment_fifo_capacity`が30と60。掃引の記録の形式の版が5で、計画の項目を持つ。

6. 前の形式の要約: 要約の形式の版1の要約（既存の補助で作る）から作った表でも、列`pending_assignment_fifo_capacity`が埋まる。

文書（`docs/experiments/experiment-sweep.md`）に書く内容: 軸の書き方、時間尺度も同じ値で替わること（IMPROVE-015で将来は分ける）、既定は30のまま、制約（実効の下限は2、警報を待つ扱いの構成では変化区間の最小件数の5以上）、目安（手元の試行で、60がcircle2で上がり、sine2・sea2はほぼ同じ。100は60より少し悪い。研究バックログの8）、抜き出し方の軸との組合せの順、掃引の記録の形式の版5。

既存のtest: `test_experiment_sweep.py`（記録の形式の版の期待を5へ）、`test_held_model_training_sample_recency.py`（抜き出し方の軸）、goldenの版1〜5、依存境界、全pytest。

## 旧と違う点
既存の構成と軸の動作・値・識別子は変わらない。記録の形式の版だけが5になる。
