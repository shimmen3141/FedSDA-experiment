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

## 仕様レビュー1回目（GPT-6 Luna、effort medium）の反映（2026-11-01。上と食い違うときは、こちらを優先する）
- 学習の対象と標本の集合（Major 1）: FedDriftは検出バッチ（$I$ 標本）を1つのモデルへ帰属させ、帰属の後の学習（学習の対象のモデル、モデルごとの件数の条件`batch_sample_count`、R回のround、標本数の重み）を行う。Oracleでは、この「検出バッチを1つのモデルへ帰属させる」処理だけを、「検出バッチを標本の真のコンセプトで分け、分けた部分をそれぞれのコンセプトのモデルへ帰属させる」処理に置き換える。帰属の後の学習は、FedDriftの規則をそのまま適用する（モデルごとの学習の標本の集合には、そのモデルへ帰属した標本が入る。件数の条件もモデルごとに今の規則で判定する。学習の対象のモデルは、その時刻にいずれかのclientで標本が帰属したモデル）。集約の重みも、今の規則どおりモデルごとの帰属した標本数。
- 最初の予測（Major 2）: 全部のモデルは同じ初期モデルの写しから始まるので、ラベルを見る前の最初の標本の予測には、モデルのIDが最小のモデル（ID 0）を使うと定める。このとき、どのモデルで予測しても値は同じで、真のコンセプトの情報は使わない。2標本目からは、そのclientが直前にラベルを見た標本の真のコンセプトのモデルで予測する。
- testの固定条件（Major 3）: 小さい全体runは、SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本に固定する。testは次を記録から確かめる。(a) 少なくとも1つのclientでコンセプトが変わる。(b) 各標本の学習の行き先が、その標本の真のコンセプトのモデル。(c) 各標本の予測のモデルIDが上の規則（最初はID 0、以後は直前の標本の真のコンセプトのモデル）。(d) 現れた各コンセプトのモデルが、少なくとも1回学習される（計算回数か、モデルごとの学習の標本数で）。(e) 各clientが持つモデルの数と、サーバに登録されたモデルの数が、runの最初から最後まで $K$。作成と統合の記録が0件。条件を満たさなければ標本数を5000に増やして固定し、Implementation Notesに書く（ほかの条件は探さない）。
- 本規模の集計（Minor）: 失敗したrunがあれば集計を止めて報告する。データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度、FedHASP（基準版16）・FedDrift・FedAvg（集約ごとの形）との差（勝ち/負け）、通信、MAC、実行時間、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`と同じ定義）。比べる相手の結果は、論文の表1と同じもの（`docs/research/feddrift-post-change-accuracy-20261101.md`に書いたパス。新しい4データのFedDriftは、調整したδの結果が出ていればそれ）。

## 仕様レビュー2回目の反映（2026-11-01。上と食い違うときは、こちらを優先する）
- 学習の対象と既存の規則への接続（Major）: FedDriftの既定の学習の対象は`training_model_eligibility="models_assigned_in_current_timestep"`で、サーバが各clientの現在の割当のモデルID（client 1つにつき1つ）の和集合を、その時刻の学習の対象にする（`feddrift_run_server.py`の`_current_timestep_training_model_ids`、`synchronize_timestep`）。Oracleでは、各clientの「その時刻に標本を1件以上帰属させたモデルIDの集合」（client 1つにつき複数あり得る）の和集合を、その時刻の学習の対象にする。各clientは、学習の対象のうち自分が学習の標本を持つモデルを、今の`train_feddrift_held_models_for_round`の規則（そのclientのモデルごとの学習の標本の件数が`batch_sample_count`以上のとき参加、R回のroundのそれぞれで更新）で学習する。学習の標本の集合は、今の`ModelTrainingSampleStore`のモデルごとの集合で、Oracleではそこへ標本ごとに真のコンセプトのモデルへ追加する（今は検出バッチ全体を割当のモデルへ追加する）。集約の重みは、今の`_aggregate_and_distribute_models`の規則（clientごと・モデルごとの学習の標本の件数）のまま。その時刻に標本が帰属しなかったコンセプトのモデルは、どのclientでも学習の対象にならない（今の既定で、どのclientにも割り当てられていないモデルが学習されないのと同じ）。割当のIDの報告の通信は、今と同じくclient 1つにつき1通（IDの集合を1通で送る）と数える。
- 記録（Minor）: モデルの数と作成・統合が無いことは、既存の記録で確かめる。サーバの時刻ごとのモデル数（`snapshot_synchronized_timestep_model_counts`）がすべての時刻で $K$、モデルの登録の記録（`model_registration`の表）が最初の $K$ 個だけ、統合の記録が0件、各clientの保有モデルIDが毎時刻 $\{0,\dots,K-1\}$。モデルのIDとコンセプトの対応は「ID $k$ ＝コンセプト $k$」に固定し、写像の表は作らない。
