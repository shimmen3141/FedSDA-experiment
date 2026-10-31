# Design Document

## Overview
手元のFedDriftの実装の上に、IFCA+Window（FedDriftの論文の比較手法）の構成`feddrift-ifca-window`を足す。Oracleの構成（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`）で決めた既存の規則への接続を、そのまま使う。変える点は、(1) runの最初に $K$ 個のモデルを持ち、ID 1〜$K-1$を乱数で初期化する、(2) 各時刻の終わりの割り当てを「そのバッチでの平均損失が最小のモデル」にする（検出・作成・統合なし）、(3) 学習の標本を今の時刻の標本だけにする（窓1時刻）、の3点。

## IFCAとFedDriftの論文のIFCA+Windowとの関係
- IFCA（Ghosh et al., NeurIPS 2020）は、クラスタの数 $K$ を与え、各clientが損失の最小のクラスタのモデルを選び、選んだモデルを学習する。クラスタのモデルは乱数で初期化する。
- FedDriftの論文は、時間とともに変わる設定へ、学習のデータを直近の1時刻に限る窓を足した版（IFCA+Window）を比較手法にした。本構成はこれに当たる。
- 違い: (a) 本稿は逐次の評価で、選んだモデルで次の時刻の標本を予測する（FedDriftの割当のモデルで予測するのと同じ）。(b) 全手法に共通の事前学習の初期モデルを使うので、ID 0だけを事前学習のモデルにし、ほかを乱数で初期化する（全部を同じ初期モデルにするとクラスタが分かれない）。(c) $K$ を真のコンセプトの数として与える（IFCAに有利な条件）。

## Architecture（実装担当が既存の構造を調べて具体化する。Oracleの実装で足した設定の項目と分岐を再利用する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_IFCA_WINDOW_CONFIGURATION_NAME`（値`feddrift-ifca-window`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。構成`feddrift`の設定を元に、割り当ての方式の設定の項目（Oracleで足した項目）を新しい値（例: `minimum_loss_cluster_assignment`）にし、学習の窓の項目（新しい項目。既定は今の「全部の時刻」）を「今の時刻だけ」にする。設定の項目は`experiment_spec`の規約で登録する。
- モデルの登録と初期値: runの組立てで $K$ 個のモデルを登録する（Oracleと同じ）。ID 1〜$K-1$は、事前学習をせず、runのseedとモデルのIDから決めた分けた乱数の状態（runの乱数を消費しない）で、モデルの構造の既定の初期化（`reset_parameters`）を行ったもの。共有の特徴抽出器を持たないFedDriftのモデルでは、モデル全体を初期化する。
- 割り当て: 各時刻の終わり（検出バッチがそろったとき）に、そのclientの $K$ 個のモデルのそのバッチでの平均損失（FedDriftの検出と同じ損失。今の`detection_batch_assessment`の評価）を求め、最小のモデルを、そのバッチの学習の行き先と、次の時刻の予測のモデルにする（同値はIDの小さい方）。検出・作成・統合（サーバの時刻の準備のクラスタリング）は行わない。
- 学習の窓: 各時刻の学習の前に、各clientのモデルごとの学習の標本の集合を、その時刻に帰属した標本だけにする（過去の時刻の標本を外す）。学習の対象は、今の既定の規則（その時刻にいずれかのclientに選ばれたモデルの和集合）。件数の条件、R回のround、集約の重み（clientごと・モデルごとの学習の標本の件数＝その時刻の件数）は今の規則のまま。
- 最初の時刻の予測: 最初の検出バッチがそろうまでは、ID 0（事前学習のモデル）で予測する。
- 記録: 各時刻のclientごとの選択（モデルIDと、$K$ 個のモデルの平均損失）を記録する（既存の表に無ければ足す）。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_ifca_window_configuration.py`（要求2.1・2.2）。小さい全体runの条件はSINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本に固定する。(a) モデルの数・登録・統合はOracleと同じ記録で確かめる。(b) ID 1〜$K-1$の初期値がID 0と異なり、同じseedで同じになることは、runの組立ての部品のtestで確かめる。(c) 選択が平均損失の最小と一致することは、記録した $K$ 個の平均損失から確かめる。(d) 学習の標本が今の時刻だけであることは、学習の直前のownerの状態（モデルごとの学習の標本の標本番号が、その時刻の範囲に入る）で確かめる。
- 構成・掃引・識別子・依存境界のtest、FedDriftとOracleのtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは主担当が流す。

## 本規模の実行（要求3.2）
- `--configurations feddrift-ifca-window --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study method-comparison-ifca-v16 --label ifca-window-seeds0-9 --purpose "FedDriftの学習・集約の上で、IFCA+Window（クラスタ数は真のコンセプト数）を10データで評価する"`（400 run）。集計はOracleのspecと同じ項目。

## 仕様レビュー1回目（GPT-6 Luna、effort medium）の反映（2026-11-01。上と食い違うときは、こちらを優先する）
- 初期化（Major 1）: この構成は、FedDriftの論文のIFCA+Windowそのものの再現ではなく、本稿の全手法に共通の事前学習の条件に合わせた派生版と位置づける。ID 0だけが事前学習の初期モデル、ID 1〜$K-1$は乱数の初期化。論文のIFCA（全部のクラスタを乱数で初期化、FMoWでも事前学習の代わりに乱数）との違いと、その理由（全手法が同じ事前学習の初期モデルから始まる条件を保ち、かつクラスタを分けるため。ID 0に事前学習の利点が偏る）を、`differences-from-feddrift.md`と論文に書く。全部を乱数で初期化する版は、この構成では作らない（必要ならユーザーの判断で別に足す）。
- 学習の対象と既存の規則への接続（Major 2）: Oracleのspec（design.mdの「仕様レビュー2回目の反映」）と同じ接続を使う。各clientがその時刻に選んだモデルは1つなので、学習の対象は、今の既定`training_model_eligibility="models_assigned_in_current_timestep"`のとおり、各clientの選んだモデルIDの和集合。各clientは、学習の対象のうち自分が学習の標本を持つモデルを、今の`train_feddrift_held_models_for_round`の規則（そのclientのモデルごとの学習の標本の件数が`batch_sample_count`以上のとき参加、R回のroundのそれぞれで、同じ標本の集合を使って更新）で学習する。窓1時刻なので、各時刻の学習の前に、各clientの全モデルの学習の標本の集合を空にしてから、その時刻のバッチを選んだモデルへ追加する。したがって、そのclientで学習に参加するのは選んだモデルだけで、件数はその時刻の標本数（$I$）。集約の重みは、今の`_aggregate_and_distribute_models`の規則（clientごと・モデルごとの学習の標本の件数＝その時刻の件数）。
- test（Minor 3）: 小さい全体runで、(a) 少なくとも1つのclientでコンセプトが変わる、(b) 各時刻の各clientの選択が記録した $K$ 個の平均損失の最小（同値はIDの小さい方）、(c) 選ばれたモデルだけが、その時刻の標本で学習される（ownerの状態か計算回数で）、(d) 各clientの保有モデルIDが毎時刻 $\{0,\dots,K-1\}$、サーバの時刻ごとのモデル数が $K$、登録の記録が最初の $K$ 個だけ、統合の記録が0件、を確かめる。ID 1〜$K-1$が実際に選ばれるかはIFCAの結果であって要求ではないので、testでは求めず、選ばれた回数を記録して報告する。
- 本規模の集計（Minor 4）: 失敗したrunがあれば集計を止めて報告する。データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度、FedHASP（基準版16）・FedDrift・FedAvg（集約ごとの形）・Oracleとの差（勝ち/負け）、通信、MAC、実行時間、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`と同じ定義）、IDごとに選ばれた回数。比べる相手の結果は論文の表1と同じもの。

## 仕様レビュー2回目の反映（2026-11-01。上と食い違うときは、こちらを優先する）
- 初期化（Major、2回目も同じ指摘）: 2つの構成に分け、報告でも区別する。
  - `feddrift-ifca-window`: 論文に準じる版。$K$ 個のモデルを全部、事前学習をせずに乱数で初期化する（runのseedとモデルのIDから決めた分けた乱数の状態。runの乱数を消費しない）。共通の事前学習の初期モデルは使わない。
  - `feddrift-ifca-window-pretrained-cluster0`: 共通の事前学習の条件に合わせた派生版。ID 0だけが事前学習の初期モデルの写し、ID 1〜$K-1$は上と同じ乱数の初期化。
  - 論文の比較表には前者を「IFCA+Window」として載せ、後者は初期化の影響を示す参考として載せるか付録に置く（ユーザーが決める）。どちらも`differences-from-feddrift.md`に書く。初期化の設定の項目は1つ（例: `cluster_model_initialization`、値は全部を乱数／ID 0だけ事前学習）にし、ほかの設定は同じ。
  - 最初の検出バッチがそろうまでの予測は、どちらもID 0のモデルで行う。
  - 本規模の実行は2構成で800 run（`--configurations feddrift-ifca-window feddrift-ifca-window-pretrained-cluster0`、ほかは上のコマンドと同じ）。
- 同値の扱い（Minor）: 要求1.3に「同値はIDの小さい方」と既に書いてあるので、変えない。
