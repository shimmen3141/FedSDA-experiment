# Design Document

## Overview
手元のFedDriftの実装の上に、割り当てを真のコンセプトで行う構成`feddrift-oracle-concept-assignment`を足す。FedDriftの複数モデルの学習・集約・通信はそのまま使い、(1) runの最初に $K$ 個のモデルを持つ、(2) 学習の標本を真のコンセプトで振り分ける、(3) 予測は直前にラベルを見た標本のコンセプトのモデルで行う、(4) 検出・作成・統合を行わない、の4点だけを変える。

## FedDriftの論文のOracleとの関係
- 論文のOracleは「学習の時点で真のコンセプトのIDを知り、真のクラスタリングでFedDriftの学習（Algorithm 1）を行う」もので、次の時刻の標本でテストする（変化した時刻は平均から除く）。
- 本稿は逐次の評価（ラベルを見る前に予測する）なので、予測の時点ではまだその標本のコンセプトを知らないとし、直前にラベルを見た標本のコンセプトのモデルで予測する。変化の直後の1標本だけは古いコンセプトのモデルで予測することになる。真のコンセプトを予測の時点でも使うと、手法の比較ではなく、問題の上限（変化が起きた瞬間に分かる）になるため、使わない。
- 学習の時期（各時刻の終わりに、その時刻の標本でR回のround）と集約は、FedDriftの構成`feddrift`と同じ。そのため、この構成は「割り当てを誤らないFedDrift」の上限であり、FedHASPの局所の学習（標本ごと）とは学習の時期が違う。論文ではこの違いを書く。

## Architecture（実装担当が既存の構造を調べて具体化する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_ORACLE_CONCEPT_ASSIGNMENT_CONFIGURATION_NAME`（値`feddrift-oracle-concept-assignment`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。組立ては、構成`feddrift`の設定を元に、割り当ての方式を新しい値（例: `true_concept_assignment`）にする設定の項目で切り替える。設定の項目は`experiment_spec`の規約（`OptionSpec`・`ActivationRule`など。AGENTS.md）に従って登録する。
- `runtime/feddrift_*`・`methods/feddrift`: 割り当ての方式が真のコンセプトのとき、(a) runの組立てで $K$ 個のモデルを登録する（IDは0〜K−1、全部が初期モデルの写し）、(b) 検出バッチの判定（`FeddriftDetectionBatchProcessor`）の代わりに、標本ごとに真のコンセプトで学習の標本を振り分ける、(c) 予測のモデルを、直前の標本の真のコンセプトのモデルにする、(d) サーバの時刻の準備で統合を行わない。真のコンセプトは観測の列（`observed_concept_id`）から取る。
- 記録: 予測の記録（`sample_prediction.prediction_model_id`）と、学習の行き先が追える記録（既存の表に無ければ、標本ごとの学習の行き先の列を足す）。新しい列を足すなら、記録の版の扱いは既存の規約に従う。
- 既存の構成への影響: 既存の構成の設定の値・runの識別子・goldenは変えない（新しい項目の既定は、今の`feddrift`の挙動）。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_oracle_concept_assignment_configuration.py`（要求2.1・2.2）。2.2の小さい全体runの条件は、SINE2（2コンセプト）とMNIST4（4コンセプト）、間隔50、seed 0、10 client、各client 1000標本程度に固定する。各条件で、少なくとも1回のコンセプトの変化が起きることを先に確かめる（起きなければ標本数を増やして固定し、Implementation Notesに書く）。
- 構成・掃引・識別子・依存境界のtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。

## 本規模の実行（要求3.2）
- `--configurations feddrift-oracle-concept-assignment --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study method-comparison-oracle-v16 --label oracle-seeds0-9`（400 run）。
