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
期待する動作の基準版と、run設定が同じもの）は、飛ばして、残りだけを実行する。commitや、未commitの変更の有無が
違っても、基準版が同じなら、飛ばす。出力先を省略して実行していた場合は、作られたディレクトリ
（`results/<開始時刻>_<ラベル>`）を、`--output-directory`で指定する。

Ctrl-Cで止めたときは、まだworkerへ渡していないrunを取り消して、止まる（その実行の記録`sweep_record_*.json`は、書かれない）。
実行中だったrunは、途中で止まることがある。そのとき、`runs/`に、一時のファイルや、`.npz`だけが残ることがある（下の「片方しかない」場合に当たる）。

済んだrunの要約の基準版が、今のコードの基準版と違うとき（このしくみより前に保存した、基準版の記録が無い要約を含む）は、
どのrunも実行せずに、当たるrunのファイルと、記録された基準版・今の基準版を、標準エラーへ警告として出し、終了コード2で止まる
（一覧の表と、実行の記録は、書かない）。動作の違うコードの結果が、同じ出力先で混ざらないようにするため。今のコードの結果は、
別の出力先で実行する。`--print-plan`では、そのrunを`different_behavior_baseline`と、理由つきで表示する。
比較script（`run_behavior_fix_comparison.py`）も、同じ扱い。

同じ識別子のファイルが、片方しかない、読めない、または、中のrun設定が計画と違うときは、上書きせずに、そのrunを失敗として報告する。
そのファイルを確かめて、消すか、別の出力先を使う。保存の途中でprocessが強制終了されると、`.npz`だけが残ることがある（そのファイルを消して、もう一度実行する）。

## 出力

```
<出力先>/
  sweep_record_<開始時刻>.json        実行の記録（コマンドの実行ごとに1つ）
  <構成の名前>/
    run_summary.csv                   一覧の表（1 runが1行）
    runs/
      <識別子>.json                   1 runの要約（run設定の全部、指標、実行環境、runの来歴）
      <識別子>.npz                    1 runの記録の表（`表の名前.列の名前`の配列）
```

- 識別子は、`<dataset>_interval<集約間隔>_seed<seed>_<hash>`。hashは、構成の名前と、run設定の全部から計算する
  （client数や標本数が違えば、hashが違う）。
- 要約の`run_provenance`（runの来歴）は、gitのcommit（`code_commit`）、未commitの変更の有無（`has_uncommitted_changes`）、
  gitの情報が取れなかった理由（`code_revision_unavailable_reason`）、期待する動作の基準版（`behavior_baseline_version`。
  重要動作goldenの現在の版と同じ番号。runの設定が選んだ扱いとは関係しない）を持つ。要約の形式の版は2（版1の要約は、来歴を持たない）。
  未commitの変更は、gitが追跡するファイルだけで判定する（追跡しないファイルは数えないので、gitに入っていない新しいsourceを
  使っても「変更なし」になる。コードの版を完全に表すものではない）。
- `run_summary.csv`は、その構成の`runs/`にある全部の要約から、毎回、作り直す。列は、識別子、構成の名前、実行条件、
  runの来歴（`code_commit`・`has_uncommitted_changes`・`behavior_baseline_version`。版1の要約は、空の欄）、
  指標の全部（入れ子の指標は、`.`でつないだ名前。値のない指標は、空の欄）。
- `sweep_record_*.json`は、計画、コマンドの引数、コードの版（gitのcommitと、未コミットの変更の有無。runの要約と同じ値）、
  期待する動作の基準版、実行環境、
  開始と終了の時刻、runごとの結果（`completed`・`skipped_existing`・`failed`と、失敗の理由）を持つ。
  記録の形式の版2から。版1の記録の未コミットの変更の有無は、追跡しないファイルも数えていたので、版2の値とは比べられない。
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
| `fedsda-classwise-adwin-detection` | 提案構成の、損失の監視の検出器を、e-SRから、ADWINへ替える（全体と、正解クラス別の損失の窓を監視する。ADWINが検知しないときも、直近の損失の悪化で警報を出す補助の判定を使う） | `class-adwin` |
| `fedsda-overall-loss-adwin-detection` | 同じく、ADWINで、全体の損失の窓だけを監視する | `overall-adwin` |
| `fedsda-without-concept-specific-adapter` | 提案構成のモデルから、概念固有の残差adapterを外す（概念固有部は、出力層だけ） | `shared-backbone` |
| `fedsda-independent-models` | 提案構成のモデルを、共有部を持たないモデルへ替える（隠れ層と分類層の全部を、モデルごとに独立に持つ。ローカル学習は、モデルごとに順に行い、集約後の重みの再較正は行わない） | `independent` |

提案構成以外の構成は、提案構成の、1つの要素だけを変える。小さい条件（sine2）で、旧実装の、同じ指定の全体runと、指標と離散列が一致することを確かめている。

`fedsda-single-model-prediction`では、標本ごとに、現在のモデルだけを評価する。記録の、重みに関する項目（最大重みのモデル、実効モデル数、Fixed-Share・AdaHedgeの重みでの結合の正否ほか）は、「現在のモデルに重み1」の値になり、「どれかのモデルが正解」「確信度が最大のモデルが正解」も、現在のモデルの正否と同じになる。予測の内訳の指標を、ほかの構成と比べるときは、この違いに注意する。保有モデル×標本の延べ数（`held_model_sample_count`）は、どの構成でも、標本ごとの保有モデル数の合計で、この構成では、評価したモデルの数（標本ごとに1）より大きい。「保有モデル×標本あたり」の計算量は、その分、小さく出る。

## いまの範囲

- 構成は、上の表のもの。残りのablationの構成と、FedDriftは、順に足す。
- 図（Paretoの図ほか）は、作らない。`run_summary.csv`から作る。
- 別の出力先との、runの重複は、調べない。

## 旧実装の結果との比較（移行の間だけ）

**2026-10-13から、FedSDAの構成は、旧実装と一致しないrunがある（意図した差）。** 新実装は、clientが採用したモデルを、現行モデルとして送信までの待ちを終えたときに、サーバへ登録する（2026-10-14から。待ちの途中で現行から外れたモデルは、送信保留のまま保持し、現行へ戻って待ちを終えたら登録する。旧実装は、送信までの待ちの間に現行モデルが変わると、正式IDだけを採番して、登録しない。待ちの済んだ送信保留を全部登録する扱いも、設定で選べる）。また、推定変化区間が最小件数に達するまで、警報を待つ（2026-10-14から。IMPROVE-006。旧実装は、すぐに警報とする。ADWINの検出器の2構成は、待つ扱いと組み合わせられないので、すぐに警報とする。[spec](../../.kiro/specs/fedsda-default-waited-registration-and-alarm-wait/design.md)、判断の根拠は[LEGACY-016](../research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)、[spec](../../.kiro/specs/pending-model-registration-after-waiting-as-current-model/design.md)）。また、短い警報（推定変化区間が最小件数より少ない警報）のあと、現行モデルへ取り込んだ前区間を、保留から外す（旧実装は、保留に残すので、同じ標本が、後で、もう一度取り込まれる。[LEGACY-002](../research/implementation-findings/legacy-002-retained-assigned-samples-after-short-alarm.md)）。これらの経路を通ったrunでは、モデル数・通信量・精度が、旧実装と違う。旧と同じ扱いは、実験のコマンドからは選べない（旧実装との照合のtestだけが使う）。これらの修正のたびに、FedSDAのrunの識別子（run設定から計算する）は、全部変わった。修正より前に新実装で実行した結果は、同じ出力先を指定しても、済んだrunとしては扱われない（別のrunとして、実行される）。

提案構成の、新旧の結果が一致することを確かめるには、同じ条件を、旧実装（`bash tools/run_main_ablation_suite.sh reference`）でも
実行し、次のscriptで比べる。同じdataset・seed・集約間隔のrunどうしで、精度・定常精度・検出の指標・モデル数・通信量の
11項目が、完全に同じ値かを調べる（終了コード0＝全部一致）。

```bash
python compare_sweep_results_with_legacy.py \
    <新の出力先>/fedsda/run_summary.csv \
    <旧の結果>/reference/pareto/pareto_*.csv
```

## FedDriftを比較する

新しい構成名は`feddrift`。`--aggregation-intervals`はFedDriftでは検出バッチ件数も指定する。`--feddrift-loss-increase-thresholds`は検出と統合に共通の損失増分閾値で、省略時は0.1。FedDriftを含まない計画では指定できない。

```bash
python run_experiment_sweep.py --configurations fedsda feddrift --datasets sine2 --seeds 0 1 --client-count 10 --per-client-sample-count 5000 --aggregation-intervals 50 --feddrift-loss-increase-thresholds 0.05 0.1 --workers 14 --output-directory results/comparison
```

この例はFedSDA 2 run、FedDrift 4 run。同じdataset・seed・規模・間隔で比較し、閾値軸はFedSDAを複製しない。FedDriftのRは既定1で、client/serverの閾値を同時に設定する。対応datasetはsine2・sea2・sea4・circle2・mnist2・mnist4。MNISTのファイルを配置する手順は既存と同じ。

保存先は`<出力先>/feddrift/runs/`。要約JSONは完全条件と共通指標、NPZは記録表。`feddrift/run_summary.csv`は検出バッチ件数と損失増分閾値も含む。再実行は同じ条件の結果を飛ばし、破損・片方だけ・手法/条件不一致は上書きせず失敗とする。


## 単一モデル比較のFedAvg

`--configurations fedavg`で、旧Obliviousに対応する単一モデルの比較手法を実行できる。全到着標本を保持してオンライン学習し、累積保持件数でFedAvgする。同期後のoptimizerは初期化する。検出・モデル生成・切替・クラスタリングは行わない。正式名は`fedavg`で、旧名aliasは用意しない。

FedSDA・FedDriftと同じdataset/seed/client数/標本数/集約間隔で比較でき、精度・定常精度・通信量・計算量は共通定義。切替検出の列は空、モデル数は常に1。候補検証・routing・クロス評価などの固有指標は生成しない。`--feddrift-loss-increase-thresholds`はFedDriftだけを増やす。

```powershell
python run_experiment_sweep.py --configurations fedsda feddrift fedavg --datasets sine2 --seeds 0 --client-count 10 --per-client-sample-count 5000 --aggregation-intervals 100 --workers 1 --output-directory results/three-method-comparison
```

FedAvgの成果物は`fedavg/runs/`配下の完全設定・共通指標・環境のJSONと名前付き記録のNPZ。再開と競合拒否、`run_summary.csv`の生成は他構成と同じ。準備の計算は本実行へ含めず、処理されない末尾は生成概念列と進行件数へ残す。

## 挙動の修正の、適用の有無の比較（検証用）

LEGACY-016（採用したモデルの登録）とLEGACY-002（短い警報のあとの保留）の修正が、精度・通信量へ与える影響を、
提案構成で、適用の有無の組合せを同じ条件・同じseedで実行して比べる。登録の水準は3つ: `legacy`（固定旧と同じ）、
`fixed`（待ちの済んだ送信保留の全部を、現行かどうかに依らず登録する。基準版2・3の扱い）、`waited-as-current`
（現行モデルとして待った後に登録する。既定）。`--registration-levels`を省くと、`legacy`と`fixed`を実行する
（2026-10-13の480 runと同じ計画）。旧と同じ扱いは、この検証と、旧実装との照合の
testだけが使う（通常の実験のコマンドからは選べない）。

```bash
python run_behavior_fix_comparison.py --datasets sine2 circle2 sea4 mnist2 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 \
    --workers 14 --output-directory results/behavior-fix-comparison
python analyze_behavior_fix_comparison.py results/behavior-fix-comparison --chains > behavior-fix-comparison.md
```

出力先に済んだrunがあれば、飛ばして続ける（dataset・seed・集約間隔を、後から足せる）。集計は、datasetと集約間隔ごとに、
組合せの絶対値（平均±標準偏差）と、同じseedのrunどうしの差（平均±標準誤差。登録の扱いの効果、LEGACY-002の効果、
両方の効果、警報を待つ扱いの効果）と、短い警報の連鎖の数と長さを出す。種類の違うデータを越えて平均した行は出さない。

現行モデルとして待った後に登録する扱い（2026-10-14に一度既定にした扱い）を、短い警報の扱いの2水準と、警報を待つ扱いと組み合わせて
足すときは、次の2つを、上と同じ出力先へ実行する（組合せの全部を作るので、2回に分ける）。

```bash
python run_behavior_fix_comparison.py --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 200 \
    --registration-levels waited-as-current --short-alarm-levels legacy fixed \
    --workers 14 --output-directory results/behavior-fix-comparison
python run_behavior_fix_comparison.py --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 200 \
    --registration-levels waited-as-current --short-alarm-levels fixed --alarm-wait-levels on \
    --workers 14 --output-directory results/behavior-fix-comparison
```

推定変化区間が最小件数に達するまで警報を待つ扱い（IMPROVE-006。2026-10-14から既定。比較scriptの水準`off`は、すぐに警報とする扱い）の効果は、
`--alarm-wait-levels`で比べる。指定しなければ、既定と違う`off`（待たない。即時）だけを実行する（2026-10-13の480 runの計画を変えない）。上の結果へ後から足すときは、足す組合せだけを
指定する（設定の項目が増えたので、前のrunは識別子が違い、済んだrunとして飛ばされない。集計は、dataset・集約間隔・seedで
対にするので、前の結果もそのまま比べられる）。

```bash
python run_behavior_fix_comparison.py --datasets sine2 circle2 sea4 mnist2     --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50     --registration-levels fixed --short-alarm-levels fixed --alarm-wait-levels on     --workers 14 --output-directory results/behavior-fix-comparison
```

待つ扱いの組合せは、名前の末尾に`__alarm-wait-on`が付く。集計は、結果がある組合せについて、待つ扱いの効果（登録と短い警報の
扱いを固定して、待たない→待つ）と、切替の検出の再現率・適合率、変化から切替までの遅れを出す。待つ扱いは、検出器がe-SRの
ときだけ選べる（ADWINとの組合せは、設定の検査で拒否する）。
