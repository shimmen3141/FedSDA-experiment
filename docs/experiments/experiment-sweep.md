# 実験の掃引のコマンド（新実装）

新実装（`src/federated_learning_experiments/`）で、構成・dataset・seed・集約間隔の全部の組合せを実行し、
runごとの結果と、一覧の表を保存する。旧実装の`run_pareto_sweep.py`・`tools/run_server_sweep.sh`に当たる。

## 準備

- Pythonの環境（リポジトリの`.venv`ほか。torchとNumPyが入っていること）を有効にする。
- packageのinstallや、`PYTHONPATH`の設定は、要らない。リポジトリの直下の`run_experiment_sweep.py`を実行する。
- MNIST（`mnist2`・`mnist4`）を使うときは、`train-images-idx3-ubyte.gz`と`train-labels-idx1-ubyte.gz`を、
  リポジトリ直下の`data/mnist/`へ置く（別の場所なら、環境変数`FDE_MNIST_DATA_DIR`で指定する）。
  コマンドは、ファイルを取得しない。置かれていなければ、runを始める前に、置き場所を示して止まる。

## 実行

```bash
python run_experiment_sweep.py \
    --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 \
    --aggregation-intervals 50 100 200 500 \
    --workers 14 \
    --label main
```

| 引数 | 内容 | 既定 |
|---|---|---|
| `--configurations` | 実験の構成の名前（複数可。下の「構成」） | `fedsda`（提案構成） |
| `--datasets` | datasetの名前（複数可）。`sine2`・`sea2`・`sea4`・`circle2`・`mnist2`・`mnist4` | 必須 |
| `--seeds` | 乱数のseed（複数可） | 必須 |
| `--aggregation-intervals` | サーバの集約の間隔（clientあたりの標本数。複数可） | 必須 |
| `--client-count` | clientの数 | 10 |
| `--per-client-sample-count` | clientあたりの標本数 | 5000 |
| `--workers` | 並列に実行するprocessの数。各processの計算のthread（torchと、OpenMP・MKL・OpenBLAS）は、1つ | 1 |
| `--output-directory` | 出力先 | `results/<開始時刻>_<ラベル>` |
| `--label` | 出力先を省略したときの、ディレクトリ名のラベル | `sweep` |
| `--print-plan` | 計画のrunと、出力先にある済んだrunを表示するだけ。何も実行せず、何も作らない | — |

- 研究室サーバ（16コア）では、同時に動かす`--workers`の合計を、14以下にする。長い実験は、tmuxの中で実行する。
- 終了コード: 0＝全部のrunが、済んだか、飛ばされた。1＝失敗したrunがある、または、出力先に、読めない要約がある。2＝引数や計画が不正、または、MNISTのファイルがない。
- scriptは、`OMP_NUM_THREADS`・`MKL_NUM_THREADS`・`OPENBLAS_NUM_THREADS`を、未設定なら1にする（設定されている値は、変えない。結果を再現するには、1のままにする）。

## 途中で止まったとき

同じ`--output-directory`を指定して、同じコマンドを、もう一度実行する。済んだrun（要約と表の両方があり、
run設定が同じもの）は、飛ばして、残りだけを実行する。出力先を省略して実行していた場合は、作られたディレクトリ
（`results/<開始時刻>_<ラベル>`）を、`--output-directory`で指定する。

Ctrl-Cで止めたときは、まだworkerへ渡していないrunを取り消して、止まる（その実行の記録`sweep_record_*.json`は、書かれない）。
実行中だったrunは、途中で止まることがある。そのとき、`runs/`に、一時のファイルや、`.npz`だけが残ることがある（下の「片方しかない」場合に当たる）。

同じ識別子のファイルが、片方しかない、または、中のrun設定が計画と違うときは、上書きせずに、そのrunを失敗として報告する。
そのファイルを確かめて、消すか、別の出力先を使う。保存の途中でprocessが強制終了されると、`.npz`だけが残ることがある（そのファイルを消して、もう一度実行する）。

## 出力

```
<出力先>/
  sweep_record_<開始時刻>.json        実行の記録（コマンドの実行ごとに1つ）
  <構成の名前>/
    run_summary.csv                   一覧の表（1 runが1行）
    runs/
      <識別子>.json                   1 runの要約（run設定の全部、指標、実行環境）
      <識別子>.npz                    1 runの記録の表（`表の名前.列の名前`の配列）
```

- 識別子は、`<dataset>_interval<集約間隔>_seed<seed>_<hash>`。hashは、構成の名前と、run設定の全部から計算する
  （client数や標本数が違えば、hashが違う）。
- `run_summary.csv`は、その構成の`runs/`にある全部の要約から、毎回、作り直す。列は、識別子、構成の名前、実行条件、
  指標の全部（入れ子の指標は、`.`でつないだ名前。値のない指標は、空の欄）。
- `sweep_record_*.json`は、計画、コマンドの引数、コードの版（gitのcommitと、未コミットの変更の有無）、実行環境、
  開始と終了の時刻、runごとの結果（`completed`・`skipped_existing`・`failed`と、失敗の理由）を持つ。
- 指標と、表の列の意味、旧実装の名前との対応は、[旧実装の出力と、新実装への移行の一覧](../research/legacy-run-output-migration.md)。

## 構成

| 名前 | 内容 | 旧実装（`tools/run_main_ablation_suite.sh`）での名前 |
|---|---|---|
| `fedsda` | 提案構成 | `reference` |
| `fedsda-without-prediction-weight-recalibration` | 提案構成から、サーバの集約の後の、予測重みの再較正を外す（集約の後も、予測重みと診断重みを、作り直さずに使う） | `no-recalibration` |
| `fedsda-without-candidate-validation` | 提案構成から、候補モデルの検証を外す（警報の区間に合う保有モデルが無いとき、区間で学習した候補を、警報の後の標本で検証せずに、その場で採用する） | `immediate-creation` |
| `fedsda-loss-distance-clustering` | 提案構成の、サーバのクラスタリングの、モデル対の比べ方を替える（クラス別の正誤の信頼下限の代わりに、互いの標本での平均損失の増加量で判定する。閾値とlinkageは同じ） | `distance-average` |
| `fedsda-overall-loss-esr-detection` | 提案構成の、損失の監視の対象を、全体の損失の系列だけにする（正解クラス別の系列を混合しない） | `overall-esr` |
| `fedsda-adahedge-prediction-weights` | 提案構成の、予測に使う重みを、Fixed-Shareの重みから、AdaHedgeの重みへ替える（Fixed-Shareの重みも更新し続け、比較用に記録する） | `global-routing` |
| `fedsda-single-model-prediction` | 提案構成から、モデルの予測の結合を外す（現在の学習帰属のモデルだけで予測する。重みを持たないので、集約後の重みの再較正も行わない） | `hard-routing` |

提案構成以外の構成は、提案構成の、1つの要素だけを変える。小さい条件（sine2）で、旧実装の、同じ指定の全体runと、指標と離散列が一致することを確かめている。

`fedsda-single-model-prediction`では、標本ごとに、現在のモデルだけを評価する。記録の、重みに関する項目（最大重みのモデル、実効モデル数、Fixed-Share・AdaHedgeの重みでの結合の正否ほか）は、「現在のモデルに重み1」の値になり、「どれかのモデルが正解」「確信度が最大のモデルが正解」も、現在のモデルの正否と同じになる。予測の内訳の指標を、ほかの構成と比べるときは、この違いに注意する。

## いまの範囲

- 構成は、上の表のもの。残りのablationの構成と、FedDriftは、順に足す。
- 図（Paretoの図ほか）は、作らない。`run_summary.csv`から作る。
- 別の出力先との、runの重複は、調べない。

## 旧実装の結果との比較（移行の間だけ）

提案構成の、新旧の結果が一致することを確かめるには、同じ条件を、旧実装（`bash tools/run_main_ablation_suite.sh reference`）でも
実行し、次のscriptで比べる。同じdataset・seed・集約間隔のrunどうしで、精度・定常精度・検出の指標・モデル数・通信量の
11項目が、完全に同じ値かを調べる（終了コード0＝全部一致）。

```bash
python compare_sweep_results_with_legacy.py \
    <新の出力先>/fedsda/run_summary.csv \
    <旧の結果>/reference/pareto/pareto_*.csv
```
