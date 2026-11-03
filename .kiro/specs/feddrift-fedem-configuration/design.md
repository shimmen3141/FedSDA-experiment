# Design Document

## Overview
手元のFedDriftの実装の上に、FedEM（Marfoq et al., NeurIPS 2021）の4構成を足す。IFCA+Window（`.kiro/specs/feddrift-ifca-configuration/`、承認・実装済み）で足した $K$ 個のモデルの登録・初期化・学習の窓と、1時刻のround数（`.kiro/specs/feddrift-window-multiple-training-rounds/`、実装済み）を再利用する。IFCA+Windowとの違いは、(1) 学習の割り当てが固い選択ではなく標本ごとの事後の重み $q$ で、全 $M$ 個のモデルが毎round学習する、(2) 予測がclientごとの成分の重み $\pi_c$ による混合、(3) 統合・検出・新しいモデルの作成を行わない（IFCAと同じ）、の3点。

## 公式のFedEMの手順（`C:\Users\yshin\Research\FedSDA-experiment\FedEM`、`996fadd`）
- client（`client.py`の`MixtureClient.step`）: 各roundで、(i) `update_sample_weights`: 学習のデータの全標本で各成分の損失（`gather_losses`、交差エントロピー、`reduction="none"`）を求め、`samples_weights = softmax(log(learners_weights) − all_losses.T, dim=1).T`。(ii) `update_learners_weights`: `learners_weights = samples_weights.mean(dim=1)`。(iii) 各成分を`fit_epochs`（または`fit_batch`）で、重み付きの損失`(loss_vec.T @ weights[indices]) / loss_vec.size(0)`で学習する（`learners/learner.py`）。E-stepのデータ（`val_iterator`）は、学習と同じ`train`のファイルを`train=False`（シャッフル・`drop_last`なし）で読むもので、学習のデータの全標本と同じ（`utils/utils.py`の`get_loaders`の238〜246行と説明の204行）。round内の順序は、E-step（`update_sample_weights`）→$\pi$ の更新（`update_learners_weights`）→M-step→サーバの集約（`mix`）→配布。
- server（`aggregator.py`の`CentralizedAggregator.mix`）: 各成分を、clientの学習の標本の件数の割合（`clients_weights`）で平均して配る。
- 予測（`learners_ensemble.py`の`evaluate_iterator`）: `Σ_m learners_weights[m]·softmax(model_m(x))`。
- 初期化: 各成分のモデルを独立に乱数で初期化し、`learners_weights`は一様（$1/M$）。

## 手元への接続（実装担当が既存の構造を調べて具体化する）
- 構成: `sweep/experiment_configurations.py`に4構成の定数を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。構成`feddrift-ifca-window`の設定を元に、割り当ての方式の設定の項目を新しい値（例: `fedem_responsibility_assignment`）にする。学習の標本の範囲の項目（新しい値。`-window`は既存の「今の時刻だけ」、`-history`は「runの初めからの全部」）、初期化の項目（IFCAの既存の2値）を構成ごとに決める。設定の項目は`experiment_spec`の規約（OptionSpec・ActivationRule・ChoiceConstraint・ParameterSpec）で登録する。
- モデルの登録と初期値: IFCA+Windowの既存の処理をそのまま使う（$K$ を $M$ と読む）。`-pretrained-component0`の付かない2構成は、$M$ 個とも独立に乱数で初期化する（公式と同じ）。付く2構成は、ID 0だけを全手法に共通の事前学習の初期モデルの写しにし、ID 1〜$M-1$は前者と同じ乱数の初期化にする（公式との違い(c)）。4構成とも初期の $\pi_c$ は一様（$1/M$）。
- 学習の標本の範囲: `-window`はIFCA+Windowと同じく時刻の初めに空にする。`-history`は空にしない。どちらも、clientの全標本を全 $M$ 個のモデルの学習の標本にする（モデルごとに分けない。標本の重みは $q$ で与える）。
- E-step: 各roundの学習の直前に、各clientが、そのroundの学習の標本の全部について、保有する $M$ 個のモデル（直前に配布されたモデル）の交差エントロピーの損失を求め、公式の式で $q$ を求める。$q$ はM-stepの重みに使う。$\pi_c$ は、その時刻に届いた標本（$I$ 件）の $q$ の平均にする。`-window`では学習の標本がその時刻の標本だけなので、公式（学習のデータの全標本の平均。公式の`val_iterator`は学習のデータの全標本）と同じになる。`-history`では、学習は初めからの全標本で行うが、$\pi_c$ を全標本の平均にすると古いコンセプトの割合が残り、clientのいまのコンセプトの成分を予測で重く使えない。固定のデータを前提にした公式の平均を、そのまま非定常のストリームへ延ばさず、$\pi_c$ をその時刻の標本で推定する（ストリームへ移すための違い(g)。FedEMを不当に弱くしないため）。E-stepの計算は計算量として計上する（推論の標本数）。
- M-step: 各roundで、全 $M$ 個のモデルを、今の局所学習の規則（`train_feddrift_held_models_for_round`の件数の条件・更新の回数・ミニバッチの作り方・学習率・optimizer）で学習する。ただしミニバッチの損失は、公式の式 $\sum_{i\in B} q_{i,m}\,\ell_m(x_i,y_i)/|B|$ にする。更新の回数は、FedDriftの1つのモデルの学習と同じ予算を成分ごとに使う。したがって局所の学習の計算量はおおむね $M$ 倍、パラメータの通信は $M$ 倍になる（公式も各成分を同じ`local_steps`で学習する）。計算量をそろえるための予算の調整はしない。精度と通信・計算量は、論文の図3と同じく対で示す。
- 集約: 全 $M$ 個のモデルを学習の対象にする（全clientが毎round全モデルを送る）。各モデルの集約の重みは、clientの学習の標本の件数（`-window`はその時刻の件数、`-history`は累積の件数）。$q$ の和は使わない（公式の`clients_weights`）。
- 時刻の中の順序と予測: 時刻 $t$ の各標本は、(1) ラベルを見る前に、直前の配布のモデル（時刻の途中では学習しない。既存のFedDrift系の構成と同じ）の予測確率を、直前の時刻 $t-1$ の最後のroundのE-stepの $\pi_c$ で重み付けて混ぜて予測し、(2) ラベルを見て学習の標本へ加える。$I$ 件がそろったら、round $r=0,\dots,R-1$ のそれぞれで、E-step（そのroundの学習の標本の全部の $q$、その時刻の標本の $\pi_c$）→M-step→集約→配布を行う。時刻 $t+1$ の予測は、最後の配布のモデルと最後のroundの $\pi_c$ を使う。最初の時刻の予測は、初期のモデルと一様な $\pi_c$。予測の後のFedDriftの検出・新しいモデルの作成・サーバの統合は行わない。
- round数: 1時刻のround数の掃引の値を4構成でも使えるようにする（`feddrift_run_participant_factory.py`の検査と掃引の計画の制約）。各roundで、E-step→M-step→集約・配布を行う（公式の`step`と同じ順）。
- 記録: 表`fedem_client_mixture`（列: `timestep_index`、`client_id`、`model_id`、`mixture_weight`。各時刻の最後のroundのE-stepの後の $\pi_c$）と、表`fedem_sample_responsibility`（列: `timestep_index`、`client_id`、`sample_index`、`max_responsibility_model_id`、`max_responsibility`。各時刻の最後のroundのE-stepでの、その時刻に届いた標本だけの値。`-history`ではE-stepの対象は初めからの全標本だが、記録するのはその時刻に届いた標本だけ）。名前と列は既存の記録の規約に合わせて実装担当が決めてよく、naming.mdへ追記する。保存形式の版を上げる。

## 公式のFedEMとの違い（文書と論文に書く）
- 準じる点: E-stepとM-stepの式、全成分の学習、集約の重み、混合による予測、乱数の初期化と一様な初期の $\pi$。
- 全手法に共通の条件から来る違い: (a) 逐次の評価（ラベルを見る前に予測する）。公式は固定の学習・テストのデータ。(b) 局所学習の予算・学習率・optimizer（Adam、D04・D05・D11）。公式はSGDとエポック数。(c) 事前学習の初期モデル（`-pretrained-component0`だけ）。
- ストリームへ移すための違い: (g) 予測に使う $\pi_c$ は、その時刻の標本の $q$ の平均（`-window`では公式と同じ。`-history`では学習の標本の全部の平均ではない）。(d) 学習のデータは、届いた標本のうちその時刻だけ（`-window`）か全部（`-history`）。公式は固定のデータ。(e) 1時刻のround数 $R$（既定1）。公式は固定のデータで多数のround。`-history`ではE-stepとM-stepが時刻をまたいで繰り返される。(f) $M$ をコンセプトの数として与える（FedEMに有利）。公式では $M$ はハイパーパラメータ。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_fedem_configuration.py`。
- 公式の式との照合（要求3.1）: 公式の`MixtureClient.update_sample_weights`・`update_learners_weights`、`learner.fit_batch`の損失、`clients_weights`、`evaluate_iterator`の混合の式の抜粋（式だけを抜き出したもので、関数全体の写しではない。対応する公式のファイル・行は抜粋の各関数の説明に書く）を`results/_sol_prompts/fedem-official-excerpt.py`へ置き（主担当が作った）、固定の損失の行列・重み・ミニバッチ・件数・確率で、手元の $q$・$\pi$・重み付きの損失・集約の重み・混合の予測が一致すること（許容誤差1e-6）を確かめる。初期の $\pi_c$ が一様であることと、round内の順序（E-step→$\pi$ の更新→M-step→集約→配布）を、手元の処理を記録する差し込み（呼び出しの順の記録）で確かめる。
- 初期化（要求3.2）: IFCA+Windowの既存のtestと同じ観点を4構成で。
- 小さい全体run（要求3.3）: `feddrift-fedem-window`と`feddrift-fedem-history`で、SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本。(a) 学習の直前の学習の標本が、`-window`ではその時刻の標本番号だけ、`-history`では初めからの全部。(b) 毎roundで全 $M$ 個のモデルが学習・集約され、集約の重みがclientの件数の割合。(c) 予測の確率が、記録した $\pi_c$ による $M$ 個の予測の混合と一致する（数標本を照合）。(d) 上りと下りのモデルの転送数がround×client×$M$。(e) $\pi_c$ の各行の和が1で、ある時刻の予測に使った $\pi_c$ が直前の時刻の最後のE-stepの値と同じ（集約の間は変わらない）。`-history`の $\pi_c$ が、その時刻の標本の $q$ の平均である。(h) 各標本の予測が、その標本のラベルを学習の標本へ加える前に行われている（予測の記録とE-stepの対象の標本番号で照合）。(g) `-history`の`fedem_sample_responsibility`の行が、その時刻に届いた標本だけである。(f) 統合・新しいモデルの登録が0件、モデル数が毎時刻 $M$。$R=3$ でも(b)(d)を確かめる。
- 既定の不変: 既存の構成の保存設定・識別子が変わらない。構成・掃引・識別子・依存境界のtest、IFCA・Window・CFL・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`、Ruff・Pyright。全pytestは主担当が流す。

## 予備の実行（要求4.2）
- `--configurations feddrift-fedem-window feddrift-fedem-window-pretrained-component0 feddrift-fedem-history feddrift-fedem-history-pretrained-component0 --datasets sine2 circle2 mnist4 rotmnist4 --seeds 100-102 --aggregation-intervals 50 500 --client-count 10 --per-client-sample-count 5000 --study fedem-pilot-v16 --label fedem-4configs-seeds100-102 --purpose "FedEMの4構成（初期化×学習の標本の範囲）を、評価に使わないseedで比べて本規模の1構成を選ぶ"`（96 run、$R=1$、uranium）。開始と結果を実験の台帳に記録する。4データとも、各clientが独立に300標本の後から標本ごとに確率0.0015でほかのコンセプトへ移り、コンセプトは何度も再来する（論文の5.1節の設定）。
- 集計: 構成×データ×間隔ごとの累積の精度・通信・局所の計算量、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`と同じ定義）、各成分の $\pi$ の推移、$q$ の最大の成分とコンセプトの一致の割合（純度）。参考として、IFCA+Window・FedAvg+Window・FedHASPのseed 0〜9の結果を並べる（seedが違うので構成の選択には使わない）。
- 判断: 選択に使うのは予備の実行（seed 100〜102、評価に使わない）だけで、評価はseed 0〜9（本規模）で行う。指標は、構成ごとの、4データ×2間隔×3 seedの累積の精度の平均（変化の直後の精度は判断の根拠の説明に使う）。4構成のパラメータの通信は同じ（全clientが毎round $M$ 個）で、`-history`は局所の計算量（E-stepの推論）が多いので、計算量の差も示す。この値が最も高い構成を本規模に使う案をユーザーに示す。選ぶ手順: (1) 4構成のこの値の最大値を求める。(2) 最大値との差が0.1ポイント以内の構成を候補にする（最大の構成自身を含む）。(3) 候補に`-window`の構成があれば`-window`の構成だけに絞る（局所の計算量が少ない）。(4) 残りに乱数の初期化の構成があればそれを選び、無ければ`-pretrained-component0`の構成を選ぶ。成分がコンセプトに分かれない（純度が一様の割り当てと同程度）ときは、$R=10$ の予備の実行（同じ予算でroundを増やす）を提案する。本規模は10データ×間隔50・100・200・500×seed 0〜9（400 run）。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna medium）FAIL（Major 4: E-stepの対象の説明、予備の実行のseedと選択の規則、予算の意味、testの範囲。Minor 1: 写しの呼び方）。公式の`val_iterator`が学習のデータの全標本であることを確かめて説明を直し、選択の規則・予算・testを明記した。
- 2回目FAIL（Major 3: `-history`の $\pi_c$ が全標本の平均で追従しにくい、予備の実行の追従の指標、時刻の中の順序。Minor: 掃引の`--label`・`--purpose`）。`-history`の $\pi_c$ をその時刻の標本で推定する形にし（違い(g)）、時刻の中の順序と照合のtest、変化の直後の精度の集計、`--label`・`--purpose`を足した。
- 3回目FAIL（Major 2: 初期化とE-stepの範囲の書き方がrequirementsとdesignでそろっていない。Minor: 同等とみなす規則の対象）。処理は変えず、各構成の初期化とE-step・M-step・$\pi_c$ の範囲をrequirementsとdesignに明記し、選択の規則の文面を直した。
- 4回目FAIL（Minor 1だけ: 選択の手順が一意でない。1〜3回目の指摘は解消、FedEMの手順の写し・ストリームへの合わせ方に重大な不公平なしと判定）。選択の手順を(1)〜(4)の段に書き直した。Minorの文面の修正なので再レビューは省き（2026-10-27の決定）、承認とする。
