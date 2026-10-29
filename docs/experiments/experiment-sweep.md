# 実験の掃引のコマンド（新実装）

2026-10-27の新しい既定: FedSDAは損失の低下で監視の基準を下げ、クロス評価の上限を実験のclient数にする。
ADWINの2構成は基準を警報まで保ち上限はclient数。
コマンドの引数は変わらないが、完全設定が変わるのでrunの識別子は変わる。過去の結果は保存された設定と来歴で区別する。
現在の基準版は16。版16のgoldenは生成元`810e56f9`（WindowsとWSL）（[対象spec](../../.kiro/specs/fedsda-every-sample-change-interval-search-default/tasks.md)）。版15は生成元`c13149ea`（[削除spec](../../.kiro/specs/remove-post-alarm-candidate-validation/tasks.md)）。
研究CLIの非product系は版10を保ち、IMPROVE-031と上限撤廃だけの比較は版12を基点にする。
研究入口はbuilderの結果を置き換え、product-defaultだけは製品の新しい既定を使う。
[検証と未実施範囲](../../.kiro/specs/fedsda-loss-monitor-rebaseline-and-all-evaluating-clients/tasks.md)。

新実装（`src/federated_learning_experiments/`）で、構成・dataset・seed・集約間隔の全部の組合せを実行し、
runごとの結果と、一覧の表を保存する。旧実装の`run_pareto_sweep.py`・`tools/run_server_sweep.sh`に当たる。

FedSDAの全構成は、クロス評価で対象を保有し、
対象の標本を返せるclientだけを評価させる既定（`clients_with_target_model_samples`）を引き継ぐ。
コマンドの引数は変わらず、選び方は保存するrun設定に入る。
研究ツールの`product-default`系は製品の新しい既定を使い、`product-default-sample-holding-cross-evaluation`は
`product-default`と同じ選び方になる。それ以外の変種は基準版10の既定と従来の選び方を基点にする。
検証の入口は[対象specのImplementation Notes](../../.kiro/specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)。

基準版15では候補検証の記録の表・指標・設定を保存しない。設定が減るため、FedSDAのrun識別子の末尾hashは変わる。
基準版14までの結果を読む集計の道具（`tools/experiments/accuracy_periods.py`、
`tools/research_feddrift_delta_comparison.py`など）はそのまま使える。runはデータ・集約間隔・seed（構成も揃える）で対応させ、
hashで対応させない。過去の掃引への再開・seed追加は行わず、新しい保存先へ実行する。
FedDrift・FedAvgの保存の形とrun識別子は変わらない。

## 準備

- Pythonの環境（リポジトリの`.venv`ほか。torchとNumPyが入っていること）を有効にする。
- packageのinstallや、`PYTHONPATH`の設定は、要らない。リポジトリの直下の`run_experiment_sweep.py`を実行する。
- MNIST（`mnist2`・`mnist4`・`rotmnist2`・`rotmnist4`）を使うときは、`train-images-idx3-ubyte.gz`と`train-labels-idx1-ubyte.gz`を、
  リポジトリ直下の`data/mnist/`へ置く（別の場所なら、環境変数`FDE_MNIST_DATA_DIR`で指定する）。
  コマンドは、ファイルを取得しない。置かれていなければ、runを始める前に、置き場所を示して止まる。
- Fashion-MNIST（`fmnist2`・`fmnist4`）は、同名の学習用IDXファイルを`data/fashion-mnist/`へ置く
  （別の場所なら`FDE_FASHION_MNIST_DATA_DIR`）。[定義・根拠と取得後のMD5照合の手順](dataset-characteristics.md#fashion-mnistの配置)に従う。
  4データの概念・先行実装との差も[データ特性](dataset-characteristics.md#fashion-mnistのラベルの入れ替えとmnistの回転)を参照する。

## 実行

```bash
python run_experiment_sweep.py \
    --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 \
    --aggregation-intervals 50 100 200 500 \
    --workers 14 \
    --study main-results --label all-datasets \
    --purpose "提案構成の主結果を、6データで取る"
```

| 引数 | 内容 | 既定 |
|---|---|---|
| `--configurations` | 実験の構成の名前（複数可。下の「構成」） | `fedsda`（提案構成） |
| `--datasets` | datasetの名前（複数可）。`sine2`・`sea2`・`sea4`・`circle2`・`mnist2`・`mnist4`・`fmnist2`・`fmnist4`・`rotmnist2`・`rotmnist4` | 必須 |
| `--seeds` | 乱数のseed。列挙・両端込みの範囲・混在が可能（下記） | 必須 |
| `--aggregation-intervals` | サーバの集約の間隔（clientあたりの標本数。複数可） | 必須 |
| `--client-count` | clientの数 | 10 |
| `--per-client-sample-count` | clientあたりの標本数 | 5000 |
| `--workers` | 並列に実行するprocessの数。各processの計算のthread（torchと、OpenMP・MKL・OpenBLAS）は、1つ | 1 |
| `--output-directory` | 出力先（再開のときに指定する。`--study`とは同時に使えない） | `results/<研究の問い>/<開始時刻>_<ラベル>`（`--study`が無ければ`results/<開始時刻>_<ラベル>`） |
| `--study` | 研究の問いの名前（出力先の1段目） | なし |
| `--label` | 出力先を省略したときの、ディレクトリ名のラベル（`/`・`\`・`:`と、Windowsでファイル名に使えない文字を含まない名前） | `sweep` |
| `--fedsda-pending-assignment-fifo-capacities` | FedSDAの構成の、帰属の保留（FIFO）の容量（標本数。複数可。下の「帰属の保留の容量とFixed-Shareの時間尺度」）。FedSDAの構成を選んだときだけ | `30`だけ |
| `--fedsda-fixed-share-time-scales` | FedSDAの構成の、Fixed-Shareの時間尺度T（標本数。複数可。同じ節）。FedSDAの構成を選んだときだけ | `120`だけ |
| `--purpose` | この実験の目的（1行）。実行するときは必須。`README.md`と実行の記録に書く | 必須（`--print-plan`では不要） |
| `--print-plan` | 計画のrunと、出力先にある済んだrunを表示するだけ。何も実行せず、何も作らない | — |
| `--new-output` | 同条件の過去掃引を再開せず、新しい保存先で実行する（`--output-directory`とは併用不可） | — |

- 長い実験はtmux内で実行する。uraniumは256コア・標準`--workers 64`。手元16コア向けの14という目安と区別する。環境別の現行条件は[steeringのtech.md](../../.kiro/steering/tech.md)を参照する。下の14 workerの例をサーバの上限と解釈しない。
- 終了コード: 0＝全部のrunが、済んだか、飛ばされた。1＝失敗したrunがある、または、出力先に、読めない要約がある。2＝引数や計画が不正、または、MNIST・Fashion-MNISTのファイルがない。
- scriptは、`OMP_NUM_THREADS`・`MKL_NUM_THREADS`・`OPENBLAS_NUM_THREADS`を、未設定なら1にする（設定されている値は、変えない。結果を再現するには、1のままにする）。

## 結果の置き場所、README、台帳

- 結果は、研究の問いごとに、`results/<研究の問い>/<開始時刻>_<ラベル>/`へ置く（`--study`と`--label`）。名前に長い条件を入れない
  （条件は、要約・実行の記録・READMEに残る）。1つの出力先は、1つの実験（1つの目的）に使い、再開は同じ目的のときだけにする。
- コマンドは、実行を始めるとき、出力先に`README.md`が無ければ作る（目的、研究の問い、作成の日時、コマンドの引数、commit、
  未commitの変更の有無、基準版、結論の欄）。既にあれば変えない（人が書いた結論を消さないため。各回の目的は、実行の記録に残る）。
  基準版の違う済んだrunで止まるときは、READMEも作らない。
- 実験を始めたら、[実験の台帳](experiment-log.md)に1行足し、結論が出たら、台帳とREADMEの結論に、判断の記録への参照を書く。

## 帰属の保留の容量とFixed-Shareの時間尺度（FedSDA）

FedSDAのclientは、観測した標本を帰属の保留（FIFO）に置き、あふれた古い標本から、現在のモデルの学習データへ確定する。`--fedsda-pending-assignment-fifo-capacities`で、この容量（標本数）を並べて比べられる。

Fixed-Shareの時間尺度Tは、予測の重みを他のモデルへ分け直す速さ（最良のモデルが入れ替わる見込みの間隔）。既定の混ぜる予測（`fedsda`、`fedsda-synchronized-local-mixture-prediction`）の、同期の時点と局所の重みの分け合いの割合も1/Tになる。`--fedsda-fixed-share-time-scales`で並べられる。

- 2つの軸は、それぞれの値だけを替える（指定しない側は構成の値のまま）。両方を指定すると全部の組を作る（容量の順の中で、Tの順）。FedSDAの構成ごとに、組の数だけrunが増える。
- 掃引の記録の形式の版6まで（2026-10-16より前）は、容量の軸がTも同じ値に替えていた（設定の検査が同じ値を求めた。IMPROVE-015で分けた）。その結果の「容量60」は、T＝60でもある。
- 既定は容量30・T120。容量30とT120を明示したrunは、指定しないrunと同じ設定・同じ識別子になる。
- 制約: 容量は、警報を待つ扱いの構成（既定）では変化区間の最小件数（5）以上（警報を即時にする構成では1以上）。Tは2以上。満たさない値は、引数の誤り（終了コード2）。
- 現在のモデルだけで予測する構成（`fedsda-single-model-prediction`）では、Tを替えても予測は変わらないが、計画からは除かない。AdaHedgeの構成ではモデル別の重みにはTを使わないが、混ぜる予測の同期・局所間の分け合いにはTを使う。
- 目安（手元の診断、混ぜる予測、研究バックログの8）: 容量を60にすると、circle2の間隔50は学習データの混ざりが減って上がるが、sine2の間隔200は新しい概念の学習が遅れて下がる（データによって得失が割れる）。Tだけ・混ぜる分け合いだけを、それぞれ60にすると、circle2で揃って上がり、sine2はほぼ同じ（どちらもTで決まるので、今は一緒に替わる）。
- 一覧の表の`pending_assignment_fifo_capacity`と`fixed_share_time_scale`の列に、runの値が入る。

例: `--fedsda-pending-assignment-fifo-capacities 30 --fedsda-fixed-share-time-scales 30 60`。[容量の軸のspec](../../.kiro/specs/fedsda-pending-assignment-fifo-capacity-axis/design.md)、[時間尺度の軸のspec](../../.kiro/specs/fedsda-fixed-share-time-scale-axis/design.md)。

## 途中で止まったとき

出力先を省略して同じ条件の掃引コマンドを対話端末で実行すると、過去掃引の保存先と進捗（例: `67/240 run`）を表示する。番号を選ぶと完了分を飛ばし、未完了runを最初から計算する。workersは7から14などへ変更できる。全run完了の候補を選ぶと、追加計算や記録作成をせず終了する。「新しいディレクトリで実行」とキャンセルも選べる。

一致判定は構成名と完全なrun設定の集合・動作基準版による。workers、引数順序、目的、study、labelは判定に含めない。探索範囲はこのcheckoutの`results/`配下。補助計画がない過去掃引も、既存READMEに条件と同じ基準版が記録されていれば候補になる。読めない・異なる来歴は自動選択しない。元READMEは初回の記録として維持し、再開時の目的や引数は今回のsweep_recordへ従来どおり記録する。

非対話端末で候補がある場合は入力を待たず、`--output-directory`で再開するか`--new-output`で新規実験を明示するよう案内して終了する。候補がなければ通常の新規実行になる。`--print-plan`は質問や復旧をせず候補と状態を表示するだけ。

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
比較script（`run_behavior_fix_comparison.py`）も、同じ扱い。比較の土台は新しい既定（混ぜる予測・T120、2026-10-20からは候補検証なし＋候補の共有部の固定）であり、過去の局所予測・T30や候補検証ありへ固定しない。

新しい実行では、補助領域`.resume/`へ掃引計画と各run開始時の完全設定・コード来歴・環境を保存する。保存中断でNPZだけが残っても、同じclean commit・実行ソースの内容hash・動作基準版・完全設定・数値環境の開始記録がある場合は、元NPZを`.resume/recovered/`へ退避して、そのrunを自動で再実行する。内容hashは未追跡のものも含めた`src/`とルート直下のPythonを対象とする。学習状態の途中再開ではない。

補助記録がない旧NPZ単独、異なる・不正な開始記録、JSONのみ、壊れたJSON/NPZ組、設定不一致は変更せず要確認とする。途中ファイルの有無と、計算中にCtrl+Cで止まった未完了runとは区別する。正常完了した結果は上書きしない。開始記録だけが残っていても、不正・不一致なら黙って置き換えず停止する。

同じ出力先の実行・復旧・集計は`.sweep.lock`のOS排他で保護する。元processとworkerの終了を待ってから再開する。lockファイルは残るが、存在だけで実行中とは判断せず、process終了時にはOSが排他を解放する。共有ファイルシステムではOSのファイルロックが正しく働く環境が必要。

NPZ・run JSON・一覧CSV・README・sweep_recordの既存の形式と版は変更しない。追加されるのは上記の再開補助ファイルだけ。

## 出力

```
<出力先>/
  README.md                         目的と出どころ（最初の実行で作る。結論は人が書く）
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
- `sweep_record_*.json`は、目的、研究の問い、計画、コマンドの引数、コードの版（gitのcommitと、未コミットの変更の有無。runの要約と同じ値）、
  期待する動作の基準版、実行環境、
  開始と終了の時刻、runごとの結果（`completed`・`skipped_existing`・`failed`と、失敗の理由）を持つ。
  記録の形式の版2から基準版、版3から目的と研究の問い、版4から学習batchの抜き出し方の軸、版5から帰属の保留の容量の軸を持つ。版6で、学習batchの抜き出し方の軸を外した（2026-10-15）。版7から、Fixed-Shareの時間尺度の軸を持ち、容量の軸が時間尺度を替えない（2026-10-16）（記録の形式の版は、動作の基準版とは別の番号）。版1の記録の未コミットの変更の有無は、追跡しないファイルも数えていたので、版2の値とは比べられない。
- 指標と、表の列の意味、旧実装の名前との対応は、[旧実装の出力と、新実装への移行の一覧](../research/legacy-run-output-migration.md)。

## 構成

2026-10-16のユーザー判断により、既定は混ぜる予測・容量30・T120となった。根拠は[研究バックログの8](../research/research-backlog.md)。`fedsda-synchronized-local-mixture-prediction`は既定と同じ設定として残す。過去の同名構成ではT30などを使っていたため、結果は構成名だけで比べず、保存した完全設定と来歴を確認する。

2026-10-20のユーザー判断により、既定は、候補を検証せずにその場で採用し、候補の学習では共有部を固定する（概念固有部——adapterと出力層——だけを、警報の時点の共有部の上で学習する）構成になった（基準版7）。根拠は[研究記録](../research/fedsda-provisional-candidate-expert-20261019.md)の最後の2節（本規模、6データ・4間隔・seed 0〜9で、24条件中18条件で候補検証ありより高精度、有意の目安で下回る条件なし。下りの通信は最大で約25%増える）。候補検証ありの構成と、候補検証だけの設定・組合せの制約は削除した。`fedsda`の2026-10-19までの結果と、これからの結果は、同じ名前でも別の設定なので、保存した完全設定と来歴で区別する。

| 名前 | 内容 | 旧実装（`tools/run_main_ablation_suite.sh`）での名前 |
|---|---|---|
| `fedsda` | 提案構成。同期時点と局所更新の結合予測を混ぜる。FIFO容量30・T120。候補を検証せずに採用し、候補の学習では共有部を固定する（基準版7） | `reference`（現在は予測方式・T・候補の採否などに意図した差がある） |
| `fedsda-every-sample-change-interval-search` | 基準版16から既定の`fedsda`と同じ設定。`change_interval_search=every_sample_with_reverse_prequential_code`で、新しい順の1本の流れの途中和から、最小変化区間以上の全部のkを1件刻みで測る。これまでのコマンドと結果の記録のために残し、識別子は実行済みの結果と同じ（`fedsda`の識別子とは異なる） | 対応なし（[既定を替えるspec](../../.kiro/specs/fedsda-every-sample-change-interval-search-default/design.md)） |
| `fedsda-minimum-interval-multiples-change-interval-search` | 提案構成の`change_interval_search`だけを`minimum_interval_multiples_with_forward_prequential_code`へ戻す（基準版15までの既定）。kを最小変化区間の倍数に限り、kごとに末尾k件を古い方から一から学習して測る | 対応なし（[spec](../../.kiro/specs/fedsda-every-sample-change-interval-search-default/design.md)） |
| `fedsda-without-warm-started-learner` | 提案構成の`new_model_hypothesis_learners`だけを`reinitialized_learner`へ替える。「新しいモデル」の仮説を初期化し直した学習者だけで測り、学習者の選択の符号長を足さない。変化区間の探索は既定のまま | 対応なし（[spec](../../.kiro/specs/fedsda-without-warm-started-learner-configuration/design.md)） |
| `fedsda-without-prediction-weight-recalibration` | 提案構成から、サーバの集約の後の、予測重みの再較正を外す（集約の後も、予測重みと診断重みを、作り直さずに使う） | `no-recalibration` |
| `fedsda-without-candidate-validation` | 候補モデルの検証をせず（警報の区間に合う保有モデルが無いとき、区間で学習した候補を、警報の後の標本で検証せずに、その場で採用する）、候補の学習で共有部も更新する。2026-10-20からの提案構成からは、候補の共有部の固定を外した構成に当たる（設定は、それまでと同じ） | `immediate-creation` |
| `fedsda-loss-distance-clustering` | 提案構成の、サーバのクラスタリングの、モデル対の比べ方を替える（クラス別の正誤の信頼下限の代わりに、互いの標本での平均損失の増加量で判定する。閾値とlinkageは同じ） | `distance-average` |
| `fedsda-overall-loss-esr-detection` | 提案構成の、損失の監視の対象を、全体の損失の系列だけにする（正解クラス別の系列を混合しない） | `overall-esr` |
| `fedsda-adahedge-prediction-weights` | 提案構成の、予測に使う重みを、Fixed-Shareの重みから、AdaHedgeの重みへ替える（Fixed-Shareの重みも更新し続け、比較用に記録する） | `global-routing` |
| `fedsda-single-model-prediction` | 提案構成から、モデルの予測の結合を外す（現在の学習帰属のモデルだけで予測する。重みを持たないので、集約後の重みの再較正も行わない） | `hard-routing` |
| `fedsda-classwise-adwin-detection` | 提案構成の、損失の監視の検出器を、e-SRから、ADWINへ替える（全体と、正解クラス別の損失の窓を監視する。ADWINが検知しないときも、直近の損失の悪化で警報を出す補助の判定を使う） | `class-adwin` |
| `fedsda-overall-loss-adwin-detection` | 同じく、ADWINで、全体の損失の窓だけを監視する | `overall-adwin` |
| `fedsda-without-concept-specific-adapter` | 提案構成のモデルから、概念固有の残差adapterを外す（概念固有部は、出力層だけ） | `shared-backbone` |
| `fedsda-independent-models` | 提案構成のモデルを、共有部を持たないモデルへ替える（隠れ層と分類層の全部を、モデルごとに独立に持つ。ローカル学習は、モデルごとに順に行い、集約後の重みの再較正は行わない） | `independent` |
| `fedsda-synchronized-model-prediction` | 提案構成の、予測（と予測重みの更新）を、最後の同期の時点のモデルで行う（配布を反映した直後に保有モデル全部を写し、次の同期まで、その写しで予測する。同期の後に作ったモデルは、共有部を写しの値、モデル固有部を現在の値で評価する。損失の監視・学習・統計・アップロードは、局所更新した現在のモデルのまま） | なし（新実装だけ。旧とは照合せず、手元の診断の差し替えと、予測の記録が一致することを確かめた） |
| `fedsda-synchronized-local-mixture-prediction` | 提案構成の予測を、同期の時点のモデルの結合予測（上の構成と同じ）と、局所更新した今のモデルを同じ重みで結合した予測の、2つを混ぜて行う。混ぜる重みはclientごとに(同期, 局所)＝(0.5, 0.5)から始め、予測の後に、それぞれの損失（1−真のクラスの確率）で`exp(−2·損失)`を掛けて正規化し、Fixed-Shareの時間尺度Tの逆数1/Tを分け合う（学習率2は固定）。予測の記録の`combined_prediction_is_correct`と`any_model_or_combined_prediction_is_correct`は混ぜた予測の側、ほかの項目は同期の側。現在のモデルだけで予測する設定とは組み合わせられない。学習・検出ほかは、局所更新した現在のモデルのまま | なし（新実装だけ。手元の診断の差し替えと、予測の記録が一致することを確かめた） |
| `fedsda-local-parameter-prediction` | 提案構成から、予測に使うパラメータだけを局所更新した現在の値へ替える。全モデルの結合は維持し、Tは120。現在の学習帰属モデルだけで予測する構成とは異なる | なし（新実装の専用testで保護） |

2026-10-15に予測の設定へ項目（`prediction_model_parameter_source`）を足したので、それより前のFedSDAの結果とは、runの識別子が一致しない（同じ出力先へ実行すると、前のrunは済んだrunとして扱われない）。比べるときは、別の出力先を使う。

2026-10-20に、参加者の設定へ、クロス評価のモデルの送り方（`cross_evaluation_model_transfer_policy`）を足し、既定を、配布するモデルを使う扱いにした（基準版8。精度は同じで、FedSDAの下りの通信量だけが減る。定義は[指標](metrics.md)）。それより前のFedSDAの結果とは、runの識別子と通信量の数え方が違う。同じ日に、参加者の設定へ統合での共有部の扱い（`consolidated_shared_feature_policy`）も足し、既定を、同じ共有部を平均せずに保つ扱いにした（基準版9。統合が起きたrunでは、値がわずかに変わる）。

2026-10-21に、帰属の設定へ、警報の後の変化点と帰属先の選び方（`alarm_interval_assignment_policy`）を足し、既定を、最小記述長の原理で選ぶ扱い（`minimum_description_length`）にした（基準版10。[説明](../components/model-reuse.md#最小記述長の原理による選択既定)）。それより前のFedSDAの結果とは、runの識別子が一致しない。`fedsda-with-candidate-validation`は今の扱いを明示するので、値は前と同じだが、識別子は変わる。

2026-10-22に、帰属の設定へ「新しいモデル」の仮説の学習者の組（`new_model_hypothesis_learners`）を、参加者の設定へ初期モデルの損失の統計の推定の仕方（`initial_model_loss_statistics_estimation`）を足し、既定を`reinitialized_and_warm_started_learners`と`five_fold_cross_validation`にした（基準版11。[説明](../components/model-reuse.md#最小記述長の原理による選択既定)、[検出器の基準](../components/drift-detection.md#fedsda実装での基準平均)）。それより前のFedSDAの結果とは、runの識別子が一致しない。`fedsda-with-candidate-validation`は基準版10までの扱いを明示するので、値は前と同じだが、識別子は変わる。研究ツール（`tools/research_provisional_candidate_expert/run.py`）の変種は、基準版10の既定を基点に評価してきたので、`product-default`以外の変種は、この2つの項目を基準版10の値に戻してから差し替える（既定を変える前と同じ値になることを確かめた）。

提案構成以外の構成は、提案構成の、1つの要素だけを変える。2026-10-20に候補の学習の設定へ項目（`candidate_shared_feature_update_policy`）を足したので、それより前のFedSDAの結果とは、runの識別子が一致しない。小さい条件（sine2）で、旧実装の、同じ指定の全体runと、指標と離散列が一致することを確かめている。

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
python run_experiment_sweep.py --configurations fedsda feddrift --datasets sine2 --seeds 0 1 --client-count 10 --per-client-sample-count 5000 --aggregation-intervals 50 --feddrift-loss-increase-thresholds 0.05 0.1 --workers 14 --study fedsda-vs-feddrift --label threshold --purpose "FedDriftの損失増分閾値を替えて、FedSDAと比べる"
```

この例はFedSDA 2 run、FedDrift 4 run。同じdataset・seed・規模・間隔で比較し、閾値軸はFedSDAを複製しない。FedDriftのRは既定1で、client/serverの閾値を同時に設定する。対応datasetは上の`--datasets`の10データで共通。新しい4データのδの既定はMNISTと同じ0.10（新データ固有の公式値ではない）。画像の配置は上の手順に従う。

保存先は`<出力先>/feddrift/runs/`。要約JSONは完全条件と共通指標、NPZは記録表。`feddrift/run_summary.csv`は検出バッチ件数と損失増分閾値も含む。再実行は同じ条件の結果を飛ばし、破損・片方だけ・手法/条件不一致は上書きせず失敗とする。


## 単一モデル比較のFedAvg

`--configurations fedavg`で、旧Obliviousに対応する単一モデルの比較手法を実行できる。全到着標本を保持してオンライン学習し、累積保持件数でFedAvgする。同期後のoptimizerは初期化する。検出・モデル生成・切替・クラスタリングは行わない。正式名は`fedavg`で、旧名aliasは用意しない。

FedSDA・FedDriftと同じdataset/seed/client数/標本数/集約間隔で比較でき、精度・定常精度・通信量・計算量は共通定義。切替検出の列は空、モデル数は常に1。候補検証・routing・クロス評価などの固有指標は生成しない。`--feddrift-loss-increase-thresholds`はFedDriftだけを増やす。

```powershell
python run_experiment_sweep.py --configurations fedsda feddrift fedavg --datasets sine2 --seeds 0 --client-count 10 --per-client-sample-count 5000 --aggregation-intervals 100 --workers 1 --study three-methods --label sine2 --purpose "3手法を同じ条件で比べる"
```

FedAvgの成果物は`fedavg/runs/`配下の完全設定・共通指標・環境のJSONと名前付き記録のNPZ。再開と競合拒否、`run_summary.csv`の生成は他構成と同じ。準備の計算は本実行へ含めず、処理されない末尾は生成概念列と進行件数へ残す。

### 局所更新を集約ごとにまとめるFedAvg

`--configurations fedavg-local-updates-per-aggregation`は、`fedavg`の局所更新の間隔（旧のτ、`local_update_interval_per_client_samples`）を、集約間隔と同じにした構成。ほかの設定は`fedavg`と同じ。元論文のFedAvg（roundのデータで学習してから集約し、配布されたモデルで予測する）に沿った形で、FedDriftの学習と予測の時機とも同じになる（FedDriftも、roundの全標本を配布されたモデルで予測し、roundの終わりに学習する。`runtime/feddrift_run_server.py`の`synchronize_timestep`）。

| | `fedavg` | `fedavg-local-updates-per-aggregation` |
|---|---|---|
| 局所更新の時機 | 標本を1つ予測するたびに1回（τ=1） | roundの最後の標本の処理で、集約間隔の回数をまとめて |
| roundの中の予測に使うモデル | 局所更新したモデル | そのroundの始めに配布されたモデル |
| 1 roundの更新の回数・batch・使うデータ・通信 | 同じ | 同じ |

どちらも、累積の保持標本がbatchの件数（32）に足りないときは、その時点の局所更新を学習せずに捨てる（既存の挙動）。集約間隔を32より小さくすると、この構成では、最初の数roundの更新がまとめて捨てられる。成果物は`fedavg-local-updates-per-aggregation/runs/`に、`fedavg`と同じ形で保存する（要約の`method_name`は`fedavg`）。旧実装に対応する構成は無いので、旧とは照合しない。goldenの事例にもせず、専用のtest（`tests/refactoring/test_fedavg_local_updates_per_aggregation.py`）で守る。

```powershell
python run_experiment_sweep.py --configurations fedavg-local-updates-per-aggregation --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 200 --workers 14 --study method-baselines --label fedavg-local-updates-per-aggregation-n5000 --purpose "局所更新を集約ごとにまとめるFedAvg（元論文に沿った形）の基準値を取る"
```

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
    --workers 14 --label legacy-vs-fixed --purpose "016と002の修正の効果を比べる"
python analyze_behavior_fix_comparison.py results/behavior-fix-comparison/<開始時刻>_legacy-vs-fixed --chains
```

比較scriptは、出力先を省略すると、`results/behavior-fix-comparison/<開始時刻>_<ラベル>`に置き（研究の問いは固定。`--label`の既定は`comparison`）、
`--purpose`（実行のとき必須）と出どころを書いた`README.md`を作る。組合せを後から足すときや、再開するときは、その出力先を
`--output-directory`で指定する。`results/behavior-fix-comparison`の直下にある組合せのディレクトリ（2026-10-13〜14の結果）は、
このしくみより前の平らな形で、新しい規約の対象外（移さない。[台帳](experiment-log.md)を参照）。

分析scriptは、結果を、入力のディレクトリの中の`analysis_<作成日時>.md`へ書く（同じ名前があれば`_2`・`_3`…）。先頭に、
出どころ（作成の日時、分析のコマンドの引数、分析のコードの版、入力のディレクトリ、組合せごとのrun数と、入力のrunのcommit・基準版。
前の形式のrunは「記録なし」）を書き、入力のrunの基準版が揃っていなければ、見出しの直後に警告を書く。`--stdout`では、同じ中身を
標準出力へ出し、ファイルを作らない。

出力先に済んだrunがあれば、飛ばして続ける（dataset・seed・集約間隔を、後から足せる）。集計は、datasetと集約間隔ごとに、
組合せの絶対値（平均±標準偏差）と、同じseedのrunどうしの差（平均±標準誤差。登録の扱いの効果、LEGACY-002の効果、
両方の効果、警報を待つ扱いの効果）と、短い警報の連鎖の数と長さを出す。種類の違うデータを越えて平均した行は出さない。

現行モデルとして待った後に登録する扱い（2026-10-14に一度既定にした扱い）を、短い警報の扱いの2水準と、警報を待つ扱いと組み合わせて
足すときは、次の2つを、上と同じ出力先（`<出力先>`）へ実行する（組合せの全部を作るので、2回に分ける）。

```bash
python run_behavior_fix_comparison.py --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 200 \
    --registration-levels waited-as-current --short-alarm-levels legacy fixed \
    --workers 14 --output-directory <出力先> --purpose "現行として待つ登録の効果を足す"
python run_behavior_fix_comparison.py --datasets sea2 sea4 circle2 sine2 mnist2 mnist4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50 200 \
    --registration-levels waited-as-current --short-alarm-levels fixed --alarm-wait-levels on \
    --workers 14 --output-directory <出力先> --purpose "現行として待つ登録の効果を足す"
```

推定変化区間が最小件数に達するまで警報を待つ扱い（IMPROVE-006。2026-10-14から既定。比較scriptの水準`off`は、すぐに警報とする扱い）の効果は、
`--alarm-wait-levels`で比べる。指定しなければ、既定と違う`off`（待たない。即時）だけを実行する（2026-10-13の480 runの計画を変えない）。上の結果へ後から足すときは、足す組合せだけを
指定する（設定の項目が増えたので、前のrunは識別子が違い、済んだrunとして飛ばされない。集計は、dataset・集約間隔・seedで
対にするので、前の結果もそのまま比べられる）。

```bash
python run_behavior_fix_comparison.py --datasets sine2 circle2 sea4 mnist2     --seeds 0 1 2 3 4 5 6 7 8 9 --aggregation-intervals 50     --registration-levels fixed --short-alarm-levels fixed --alarm-wait-levels on     --workers 14 --output-directory <出力先> --purpose "警報を待つ扱いの効果を足す"
```

待つ扱いの組合せは、名前の末尾に`__alarm-wait-on`が付く。集計は、結果がある組合せについて、待つ扱いの効果（登録と短い警報の
扱いを固定して、待たない→待つ）と、切替の検出の再現率・適合率、変化から切替までの遅れを出す。待つ扱いは、検出器がe-SRの
ときだけ選べる（ADWINとの組合せは、設定の検査で拒否する）。

## seedの列挙と範囲

```text
--seeds 0 1 2 3 4    # 従来の列挙
--seeds 0-9          # 0から9まで、終端を含む
--seeds 0-2 5 8-9    # 0 1 2 5 8 9と同じ
```

値と順序が同じなら、run設定・識別子・再開候補は同じになる。範囲の表記はシェルの展開に依存せず、CLIが解釈するので引用符は不要。これはこのCLIで定義した構文であり、別のプログラムにも通用する汎用構文ではない。

各seedは0以上2**32未満。逆順の範囲、重複（`0-2 2-4`を含む）、小数、カンマ/コロン区切りは拒否する。展開結果は100000個以内で、巨大範囲は展開前に拒否する。`--seeds`自体を繰り返した場合は、従来どおり最後の指定が有効。混在は1回の`--seeds`内にまとめることを推奨する。

通常掃引・`run_behavior_fix_comparison.py`・予測初期質量の研究CLI・構成要素の解析CLIで同じ規則を使う。seed以外の数値軸では範囲構文を追加していない。
