# Design Document

## Overview
手元のFedDriftの実装の上に、IFCA+Windowの2構成`feddrift-ifca-window`（全クラスタを乱数で初期化）と`feddrift-ifca-window-pretrained-cluster0`（ID 0だけ共通の事前学習の初期モデル）を足す。Oracleの構成（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`、承認済み）で決めた既存の規則への接続を使う。変える点は、(1) runの最初に $K$ 個のモデルを持ち、構成ごとの規則で初期化する、(2) 各時刻の終わりの割り当てを「そのバッチでの平均損失が最小のモデル」にする（検出・作成・統合なし）、(3) 学習の標本を今の時刻の標本だけにする（窓1時刻）、の3点。

## 論文のIFCA+Windowとの違い（文書と論文に書く）
- 準じる点: クラスタの数を与え、各clientが損失の最小のクラスタのモデルを選び、そのモデルを直近の1時刻のデータで学習する。`feddrift-ifca-window`は、クラスタのモデルを乱数で初期化する点でも元の論文に準じる。
- 違う点（FedDriftの実装への移植に伴うもの。どちらの構成にも共通）: (a) 逐次の評価（ラベルを見る前に予測する）で、選んだモデルで次の時刻の標本を予測する。論文は時刻 $\tau$ で学習した後に次の時刻の標本でテストし、変化した時刻を平均から除く。(b) 学習のround数・学習の対象・集約の重みは、手元のFedDriftの実装の規則（`docs/experiments/differences-from-feddrift.md`のD04・D05・D08・D11）。(c) $K$ を真のコンセプトの数として与える。(d) 選択はclientごと・時刻ごと（元のIFCAのとおり）。
- 違う点（`feddrift-ifca-window-pretrained-cluster0`だけ）: ID 0だけを全手法に共通の事前学習の初期モデルにする。全手法が同じ事前学習の初期モデルから始まる条件に合わせ、かつクラスタを分けるための派生版で、事前学習の利点がID 0に偏る。
- 論文の比較表には`feddrift-ifca-window`を「IFCA+Window」として載せ、上の(a)〜(d)を本文か脚注で書く。`feddrift-ifca-window-pretrained-cluster0`は初期化の影響を示す参考として載せるか付録に置く（ユーザーが決める）。

## Architecture（実装担当が既存の構造を調べて具体化する。Oracleの実装で足した設定の項目と分岐を再利用する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_IFCA_WINDOW_CONFIGURATION_NAME`（値`feddrift-ifca-window`）と`FEDDRIFT_IFCA_WINDOW_PRETRAINED_CLUSTER0_CONFIGURATION_NAME`（値`feddrift-ifca-window-pretrained-cluster0`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。構成`feddrift`の設定を元に、割り当ての方式の設定の項目（Oracleで足した項目）を新しい値（例: `minimum_loss_cluster_assignment`）にし、学習の窓の項目（新しい項目。既定は今の「全部の時刻」）を「今の時刻だけ」にし、クラスタのモデルの初期化の項目（新しい項目。値は全部を乱数／ID 0だけ事前学習）を構成ごとに決める。設定の項目は`experiment_spec`の規約で登録する。
- モデルの登録と初期値: runの組立てで $K$ 個のモデルを登録する（Oracleと同じ）。乱数の初期化は、runのseedとモデルのIDから決めた分けた乱数の状態で、モデルの構造の既定の初期化（`reset_parameters`）を行う。runの乱数を消費しない。
- 割り当て: 各時刻の終わり（検出バッチがそろったとき）に、そのclientの $K$ 個のモデルのそのバッチでの平均損失（今の`detection_batch_assessment`の評価）を求め、最小のモデル（同値はIDの小さい方）を、そのバッチの学習の行き先と次の時刻の予測のモデルにする。最初の検出バッチがそろうまでは、ID 0で予測する。サーバの時刻の準備のクラスタリング・統合は行わない。
- 学習の対象と既存の規則への接続（Oracleのspecの「仕様レビュー2回目の反映」と同じ接続）: 各clientがその時刻に選んだモデルは1つなので、学習の対象は、今の既定`training_model_eligibility="models_assigned_in_current_timestep"`のとおり、各clientの選んだモデルIDの和集合。各時刻の学習の前に、各clientの全モデルの学習の標本の集合（`ModelTrainingSampleStore`）を空にしてから、その時刻のバッチを選んだモデルへ追加する（窓1時刻）。各clientは、今の`train_feddrift_held_models_for_round`の規則（そのclientのモデルごとの学習の標本の件数が`batch_sample_count`以上のとき参加、R回のroundのそれぞれで同じ標本の集合を使って更新）で学習する。したがって、そのclientで学習に参加するのは選んだモデルだけで、件数はその時刻の標本数（$I$）。集約の重みは、今の`_aggregate_and_distribute_models`の規則（clientごと・モデルごとの学習の標本の件数＝その時刻の件数）。割当のIDの報告の通信は、今と同じくclient 1つにつき1通。
- 記録: 各時刻のclientごとの選択（選んだモデルIDと、$K$ 個のモデルの平均損失）を記録する表を足す（既存の表に無いため。表の名前と列は実装担当が既存の記録の規約に合わせて決め、naming.mdへ追記する）。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_ifca_window_configuration.py`。
- 部品のtest（要求2.2）: 2構成の初期値（`feddrift-ifca-window`の $K$ 個が互いに異なり事前学習の初期モデルとも異なる、`-pretrained-cluster0`のID 0が事前学習の初期モデルと同じでID 1〜$K-1$が前者と同じ、同じseedで同じ、runの乱数の状態が変わらない）。
- 小さい全体run（要求2.3）: 2構成それぞれで、SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本に固定し、(a) 少なくとも1つのclientでコンセプトが変わる、(b) 各時刻の各clientの選択が記録した $K$ 個の平均損失の最小（同値はIDの小さい方）、(c) 選ばれたモデルだけが、その時刻の標本で学習される（学習の直前のownerの状態のモデルごとの学習の標本の標本番号がその時刻の範囲に入り、選ばれていないモデルの集合は空）、(d) 各clientの保有モデルIDが毎時刻 $\{0,\dots,K-1\}$、サーバの時刻ごとのモデル数（`snapshot_synchronized_timestep_model_counts`）が $K$、登録の記録が最初の $K$ 個だけ、統合の記録が0件、を確かめる。条件を満たさなければ標本数を5000に増やして固定し、Implementation Notesに書く。ID 1〜$K-1$が実際に選ばれるかはIFCAの結果であって要求ではないので、testでは求めず、選ばれた回数を記録して報告する。
- 構成・掃引・識別子・依存境界のtest、FedDriftとOracleのtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは主担当が流す。

## 本規模の実行（要求3.2）
- `--configurations feddrift-ifca-window feddrift-ifca-window-pretrained-cluster0 --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study method-comparison-ifca-v16 --label ifca-window-seeds0-9 --purpose "FedDriftの学習・集約の上で、IFCA+Window（クラスタ数は真のコンセプト数）を2つの初期化で10データ評価する"`（800 run）。

## 本規模の集計
- 失敗したrunがあれば集計を止めて報告する。構成ごとに、データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度、FedHASP（基準版16）・FedDrift・FedAvg（集約ごとの形）・Oracleとの差（勝ち/負け）、通信、MAC、実行時間、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`と同じ定義）、IDごとに選ばれた回数。比べる相手の結果は論文の表1と同じもの。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna、effort medium）FAIL: 初期化の条件、既存の規則への接続、test、集計。反映。
- 2回目FAIL: 初期化が論文のIFCA+Windowと違う（Major）、同値の扱い（Minor、要求1.3に既にあるので変えない）。初期化を2構成に分けた。
- 3回目FAIL: 文書間の食い違い（要件は旧の初期化、設計は追記で2構成）、論文に準じる範囲の書き方。要件・設計・task・命名を2構成で書き直し、「論文のIFCA+Windowとの違い」の節を設けた。
