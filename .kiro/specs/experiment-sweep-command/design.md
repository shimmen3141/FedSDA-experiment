# Design Document: experiment-sweep-command

## Overview

**Purpose**: 構成・dataset・seed・集約間隔の組合せを、1つのコマンドで、並列に実行し、runごとの結果と、一覧の表を保存する。

**Impact**: 新しいpackage `sweep/`と、リポジトリ直下のscriptを足す。1 runの実行と保存は、変えない。

## Boundary Commitments

### This Spec Owns

- `src/federated_learning_experiments/sweep/`（新規のpackage）。
- `run_experiment_sweep.py`（リポジトリ直下。`src`を読込みの経路へ足して、コマンドを呼ぶだけ）。
- `docs/experiments/experiment-sweep.md`（使い方）。

### Out of Boundary

- 1 runの実行と保存（`runtime/`。読取りと呼出しだけ）。旧実装の入口（`run_pareto_sweep.py`、`tools/`）。

## 出力先の形

```
<出力先>/
  sweep_record_<開始時刻>.json        実行の記録（実行ごとに1つ）
  <構成の名前>/
    run_summary.csv                   一覧の表（その構成のruns/の全部から、毎回作り直す）
    runs/
      <識別子>.json                   1 runの要約
      <識別子>.npz                    1 runの記録の表
```

- 識別子: `<dataset>_interval<集約間隔>_seed<seed>_<hash>`。hashは、構成の名前と、完全なrun設定（保存用の辞書）を、キーの順をそろえたJSONにして、SHA-256の先頭12桁。client数や標本数が違えば、hashが違う。
- 出力先を省略したとき: `results/<開始時刻>_<ラベル>`（リポジトリ直下から）。ラベルの既定は`sweep`。

## 構成の名前

コマンドで指定する、実験の構成の名前。このspecで作るのは、`fedsda`（提案構成）だけ。後のspecで足す構成の、名前の案（旧の構成名との対応。ユーザーの確認を受けてから使う）:

| 新（案） | 旧 | 提案構成から変える点 |
|---|---|---|
| `fedsda` | `reference` | なし（提案構成） |
| `fedsda-single-model-prediction` | `hard-routing` | 保有モデルの出力を結合せず、学習帰属のモデル1つで予測する |
| `fedsda-adahedge-prediction-weights` | `global-routing` | 予測重みを、Fixed-Shareから、AdaHedgeへ |
| `fedsda-without-prediction-weight-recalibration` | `no-recalibration` | 集約後の、予測重みの再較正をしない |
| `fedsda-without-candidate-validation` | `immediate-creation` | 警報の後の候補の検証をせず、警報の時点で、新しいモデルを作る |
| `fedsda-loss-distance-clustering` | `distance-average` | モデルの統合の判定を、正誤の比較から、損失の距離へ |
| `fedsda-overall-loss-esr-detection` | `overall-esr` | 損失の変化の検出（e-SR）を、クラス別の系列を使わず、全体の損失だけで |
| `fedsda-classwise-adwin-detection` | `class-adwin` | 検出器を、e-SRから、ADWIN（クラス別の系列あり）へ |
| `fedsda-overall-loss-adwin-detection` | `overall-adwin` | 検出器を、ADWIN（全体の損失だけ）へ |
| `fedsda-independent-models` | `independent` | 共有部を持たず、モデルごとに独立したパラメータにする |
| `fedsda-without-concept-specific-adapter` | `shared-backbone` | 概念固有部のうち、Residual Adapterを外す（出力層以外を、全モデルで共有する） |
| `feddrift` | `FedDrift` | 比較手法 |

名前の作り方: 提案構成から、要素を外す構成は`fedsda-without-<外す要素>`、要素を置き換える構成は`fedsda-<置き換えた後のもの>`。語は、新実装の語（予測重み、候補の検証、概念固有部、損失の変化の検出）を使う。独立レビューの指摘（検出器の名前を、3つの構成でそろえる。「fully shared」は、出力層を共有しない実態と合わない）を反映した。

## Components

### sweep: experiment_configurations

| 名前 | 内容 |
|---|---|
| `PROPOSED_FEDSDA_CONFIGURATION_NAME` | `"fedsda"` |
| `EXPERIMENT_CONFIGURATION_NAMES` | 使える構成の名前のtuple（いまは、上の1つ） |
| `build_experiment_run_settings(*, configuration_name, experiment_run_conditions)` | 構成の名前と実行条件から、完全なrun設定（`FedsdaRunSettings`）を作る。知らない名前は、`ValueError` |

### sweep: experiment_sweep_plan

| 名前 | 内容 |
|---|---|
| `ExperimentSweepPlan` | `configuration_names`、`dataset_names`、`random_seeds`、`client_count`、`per_client_sample_count`、`server_aggregation_intervals_per_client_samples`（tuple。生成のときに、空・重複・型を検査する） |
| `PlannedExperimentRun` | `configuration_name`、`run_identifier`、`run_settings` |
| `build_planned_experiment_runs(*, sweep_plan)` | 全部の組合せを、構成・dataset・集約間隔・seedの順で返す。実行条件とrun設定の検査は、ここで全部済む |
| `make_run_identifier(*, configuration_name, run_settings)` | 上の識別子 |

### sweep: experiment_sweep_execution

| 名前 | 内容 |
|---|---|
| `ExperimentRunOutcome` | `configuration_name`、`run_identifier`、`run_status`（`completed`・`skipped_existing`・`failed`）、`failure_description`（失敗のときだけ） |
| `ExistingRunState` / `find_existing_run_state(*, planned_run, output_directory)` | 出力先の、そのrunのファイルの状態（`run_state`: `missing`・`completed`・`conflicting`と、`description`: 理由） |
| `execute_planned_experiment_run(*, planned_run, output_directory)` | 1 runを、実行して保存する（workerのprocessで呼ばれる）。計算のthreadを1つにする |
| `execute_experiment_sweep(*, planned_runs, output_directory, worker_count, report_progress=None)` | 済んだrunを飛ばし、残りを、逐次（worker 1）か、並列（`spawn`のprocess）で実行し、結果を、計画の順で返す。並列のときは、時間のかかる見込みのrun（MNIST、短い集約間隔）から投入する |

- 済んだrunの判定: 要約と表の両方があり、要約の`run_settings`が、計画のrun設定の辞書と等しい。
- 失敗の扱い: runの中の例外は、そのrunの失敗として記録し、ほかを続ける。保存は、既存の、上書きしない保存を使う。
- 中断の扱い: 並列の実行の途中で、例外（Ctrl-C、進み具合の報告の失敗）が出たら、まだ始まっていないrunを取り消し、実行中のrunが終わるのを待ってから、例外を伝える。その実行の記録は、書かれない（済んだrunは残り、同じコマンドで続けられる）。
- 保存の途中でprocessが止まると、表のファイル（`.npz`）だけが残ることがある。次の実行は、そのrunを、上書きせずに、失敗として報告する（残ったファイルを消してから、もう一度実行する）。

### sweep: experiment_sweep_records

| 名前 | 内容 |
|---|---|
| `RunSummaryTableResult` / `write_run_summary_table(*, configuration_directory, configuration_name)` | `runs/`の全部の要約から、`run_summary.csv`を作り直す（一時の名前へ書いてから、置き換える）。行の順は、dataset、集約間隔、seed、識別子。読めない要約は、表から外して、ファイルと理由を返す |
| `describe_code_revision(*, repository_directory)` | gitのcommitと、未コミットの変更の有無。取れなければ、理由 |
| `write_sweep_record(...)` | 実行の記録を、`sweep_record_<開始時刻>.json`へ書く（同じ名前があれば、`_2`・`_3`…をつけた名前にする）。環境には、torchのthreadの数と、数値計算のライブラリのthreadの数の環境変数を含む |

### sweep: experiment_sweep_command

`main(argv=None) -> int`。引数: `--configurations`（既定`fedsda`）、`--datasets`、`--seeds`、`--client-count`（既定10）、`--per-client-sample-count`（既定5000）、`--aggregation-intervals`、`--workers`（既定1）、`--output-directory`、`--label`、`--print-plan`。

### run_experiment_sweep.py

リポジトリ直下。数値計算のライブラリのthreadの数の環境変数（`OMP_NUM_THREADS`・`MKL_NUM_THREADS`・`OPENBLAS_NUM_THREADS`）を、未設定なら1にし（NumPyとtorchを読み込む前。workerのprocessも、このscriptを読み直すので、同じ設定になる。旧の`experiment_runtime.py`と同じ扱い）、自分の場所の`src`を、読込みの経路の先頭へ足して、`main()`を呼ぶ。

## 検証の方針

- 計画: 順、識別子の一意性と再現性、拒否。
- 実行: 小さい規模で、逐次の結果が、1 runを直接実行して保存した内容と同じ。並列の結果が、逐次と同じ（要約の指標と、表の全部）。再開（済んだrunを飛ばす）、食い違うファイルを上書きしない、runの失敗の後も続ける。
- 記録: 表の列と行、実行の記録の内容。
- コマンド: 別のprocessで、直下のscriptを実行する（installと環境変数なし）。`--print-plan`が、何も作らない。MNISTのファイルがないときに、実行の前に止まる。終了コード。
- 旧実装との照合は、このspecでは行わない（1 runの結果は、前のspecまでで、実旧と照合してある。このspecは、その結果を変えない）。提案構成を、ablationの規模で、新旧で比べるのは、この後の、サーバでの1回の確認。
