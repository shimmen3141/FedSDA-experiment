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
