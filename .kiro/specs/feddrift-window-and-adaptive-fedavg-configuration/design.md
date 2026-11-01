# Design Document

## Overview
手元のFedDriftの実装の上に、Window（単一のモデル、窓1時刻）の構成`feddrift-window`と、Adaptive-FedAvg（窓1時刻）の構成`feddrift-adaptive-fedavg-window`を足す。Oracleの構成（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`、実装`e2dae549`）とIFCAの構成（`.kiro/specs/feddrift-ifca-configuration/`、実装`7874c449`）で足した設定の項目・分岐・記録の作りと、FedDriftの学習・集約・通信の規則を再利用する。2構成に共通して変える点は、(1) モデルを1個だけ持ち、全標本をそれに帰属させる（検出・作成・クラスタリング・統合なし）、(2) 学習の標本を今の時刻だけにする（IFCAの学習の窓）。Adaptive-FedAvgだけ、(3) clientの局所の学習をSGDにし、学習率をサーバが時刻ごとに決める、(4) サーバが集約の後に公式の`AdaState`の式で次の学習率を求める。

## 公式の実装（移植の元）
- Window: 公式の`win-1`（単一のモデル、直近1時刻のデータで学習）。FedDriftの論文の主表の`Window`。optimizerはほかの手法と同じAdam（付録`appendix-expt-setup.tex`）。
- Adaptive-FedAvg: `FedDrift/fedml_api/distributed/fedavg_ens/FedAvgEnsDataLoader.py`の`AdaState`（`init_lr`、`beta1=beta2=beta3=0.5`）。`update(theta, t)`は下の式。`current_lr()`は $\eta$。
- `FedAvgEnsAggregatorAda.py`: 標本数で重みづけた平均で集約し、集約したパラメータ（`state_dict`の全部の値を平らにして連結）で`AdaState.update`を呼ぶ。設定`win-1_iter`では、時刻（iteration）ごとに1回、そのiterationの番号（0始まり）で更新する（`round_idx == comm_round - 5`のround）。学習率は`extra_info`として、配布するモデルと一緒にclientへ送る。
- `FedAvgEnsTrainerAda.py`: optimizerは`torch.optim.SGD(lr=args.lr)`に固定（momentum・weight decayなし）。学習の前に、サーバから受け取った学習率をoptimizerの`lr`に書く。学習の標本は窓1時刻（`win-1`）。
- FedDriftの論文の付録（`appendix-expt-setup.tex`）: Adaptive-FedAvgはSGDと独自の学習率の調整を使うので別に扱い、初期の学習率は $10^{-2}$（SINE-2だけ $10^{-1}$）が最良だった。ほかの手法は $10^{-a}$（$a=1,\dots,4$）から選んだ同じ学習率。

## 学習率の更新の式（公式の`AdaState.update`の移植）
集約の回数を $t=1,2,\dots$（その時刻の集約が何回目か）、集約したモデルのパラメータを1本に連結したベクトルを $\theta_t$（float64、長さ $P$）とする。状態の初期値は $\mu_0=\mathbf{0}$、$s_0=0$、$\gamma_0=0$、$\eta=\eta_0$。
- $\hat\mu_{t-1}=\mu_{t-1}/(1-\beta_1^{t-1})$、$\hat s_{t-1}=s_{t-1}/(1-\beta_2^{t-1})$（$t=1$ では両方0）
- $\mu_t=\beta_1\mu_{t-1}+(1-\beta_1)\theta_t$
- $s_t=\beta_2 s_{t-1}+(1-\beta_2)\cdot\frac{1}{P}\sum_p(\theta_{t,p}-\hat\mu_{t-1,p})^2$、$\hat s_t=s_t/(1-\beta_2^{t})$
- $r_t=\hat s_t/\hat s_{t-1}$（$\hat s_{t-1}=0$ なら $r_t=1$）
- $\gamma_t=\beta_3\gamma_{t-1}+(1-\beta_3)r_t$、$\hat\gamma_t=\gamma_t/(1-\beta_3^{t})$
- $\eta\leftarrow\min(\eta_0,\ \eta_0\hat\gamma_t/t)$

時刻 $\tau$ の集約の後に求めた $\eta$ を、時刻 $\tau+1$ の学習に使う。最初の時刻の学習は $\eta_0$。$\theta_t$ はモデルの`state_dict`の全部の値を、`state_dict`の順に平らにして連結する（今のモデルにbufferは無いが、あれば含める。公式と同じ）。式の値が有限でないときは例外で止める。

## 論文・公式の実装との違い（文書と論文に書く）
- 2構成に共通して準じる点: 単一のモデル、窓1時刻、標本数で重みづけた平均の集約。
- 2構成に共通する違い（FedDriftの実装への移植に伴うもの。IFCAの「論文のIFCA+Windowとの違い」と同じ）: (a) 逐次の評価（D12）。(b) 初期のモデルは全手法に共通の事前学習の初期モデル（D06）。(c) 局所の学習の規則は手元のFedDriftの実装（D04・D05・D11。バッチ32、更新の予算、round数1）。公式は時刻ごとに複数round・各roundで`epochs`回の乱択バッチ。
- Window: ほかの手法と同じoptimizer・学習率（`feddrift`の既定）。論文に準じる。
- Adaptive-FedAvg: 準じる点は、SGD（momentum・weight decayなし）、時刻ごとに1回の学習率の更新と上の式、$\beta_1=\beta_2=\beta_3=0.5$、$\eta_0$ を $10^{-a}$ から選ぶこと。違う点は、(d) 更新に使う $\theta_t$ の時点。公式の`win-1_iter`は、1時刻（iteration）の`comm_round`回のroundのうち、`round_idx == comm_round - 5`のroundの集約の後のパラメータで、時刻ごとに1回更新する（引数は0始まりのiterationの番号で、`update`の中で1を足すので、式の $t$ は1始まりの時刻の番号）。更新の頻度（時刻ごとに1回）と $t$ の数え方は手元も同じ。手元はround数1（D11）で1時刻の集約が1回なので、その集約の後のパラメータを使う。公式で最後から5つ目のroundを選んでいる意図はコードと論文に書かれておらず、round数1の手元では「その時刻の学習を反映した、時刻の中で唯一の集約」に置き換える。この置き換えを比較手法の説明に書く。(e) $\eta_0$ はデータ×間隔ごとに選ぶ（FedDriftのδと同じ単位）。論文はデータごと。
- 割り当てのIDの報告の通信は、FedDriftの規則のまま（client 1つにつき1通）。Adaptive-FedAvgの学習率は配布するモデルと一緒に送るスカラー1つで、通信量に数えない（文書に書く）。

## Architecture（実装担当が既存の構造を調べて具体化する。Oracle・IFCAの実装で足した設定の項目と分岐を再利用する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_WINDOW_CONFIGURATION_NAME`（値`feddrift-window`）と`FEDDRIFT_ADAPTIVE_FEDAVG_WINDOW_CONFIGURATION_NAME`（値`feddrift-adaptive-fedavg-window`）を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へこの順で加える。2構成とも、構成`feddrift`の設定を元に、割り当ての方式の項目（Oracleで足した`model_assignment_policy`）を新しい値（例: `single_global_model_assignment`）、学習の窓（IFCAで足した`training_sample_window`）を`current_timestep`にする。Adaptive-FedAvgだけ、学習率の調整の設定（新しい項目。例: `adaptive_learning_rate_settings`に $\eta_0$ と $\beta$ を持たせ、局所の学習をSGDにする）を足す。設定の項目は`experiment_spec`の規約（`OptionSpec`・`ActivationRule`・`ChoiceConstraint`、数値は`parameters.py`）で登録する。既定値は保存設定から省き、既存の識別子を変えない。
- 掃引: `--adaptive-fedavg-initial-learning-rates`を、`--feddrift-loss-increase-thresholds`と同じ作り（`experiment_sweep_plan.py`の検査、`experiment_sweep_command.py`）で足す。`feddrift-adaptive-fedavg-window`を含まない掃引で指定したら例外。値は有限で正。
- モデル: runの組立てでモデルを1個（ID 0）登録し、共通の事前学習の初期モデルの写しにする（IFCAの $K$ 個の登録の作りで $K=1$、初期化は既定の`all_pretrained`）。
- 割り当て: 各時刻の終わり（検出バッチがそろったとき）に、そのバッチの全標本をID 0へ帰属させ、予測のモデルをID 0のままにする。検出の評価・新しいモデルの作成をしない。サーバの時刻の準備のクラスタリング・統合をしない（IFCAと同じ分岐）。
- 学習の窓と学習の対象: IFCAと同じく、各時刻の学習の前に全モデルの学習の標本の集合を空にしてから（`ModelTrainingSampleStore.clear_model_training_samples`）、その時刻のバッチを追加する。学習の対象・件数の条件・更新の予算・集約の重みは今のFedDriftの規則へ接続する（IFCAのdesign.mdの「学習の対象と既存の規則への接続」と同じ）。
- 局所の学習: Windowは`feddrift`の既定のoptimizer（Adam）のまま。Adaptive-FedAvgは、学習の対象のモデルのoptimizerをSGD（momentum 0、weight decay 0）で作り、その時刻の $\eta$ を学習率にする（既存の`SgdParameterOptimizerSettings`を使う）。D05のとおり配布でoptimizerを作り直してよい（SGDは状態を持たない）。
- サーバの学習率の更新（Adaptive-FedAvgだけ）: 集約の後（モデルを配布する前）に、集約したモデルの $\theta_t$ で状態を更新し、次の時刻の $\eta$ を決める。状態はサーバのrunの中だけで持つ。学習率の更新の部品は、runtimeから独立した小さい関数・クラスにし、部品のtestで参照の実装と照合できるようにする。
- 記録: Adaptive-FedAvgだけ、時刻ごとの学習率の記録の表を足す（IFCAの`cluster_selection`表と同じ作り。列の例: `timestep_index`（0始まりの時刻の番号）、`aggregation_count`（式の $t$ ＝`timestep_index`＋1。1始まり）、`variance_ratio`（$r_t$）、`smoothed_ratio`（$\hat\gamma_t$）、`learning_rate_for_next_timestep`（$\eta$））。表の名前と列は実装担当が既存の記録の規約に合わせて決め、naming.mdへ追記する。保存形式の版を上げる必要があれば上げ、既存の構成の表と列は変えない。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_window_and_adaptive_fedavg_configuration.py`。
- 部品のtest（要求4.2）: 公式の`AdaState`（`__init__`と`update`、`current_lr`）をtestの中にそのまま写した参照の実装と、移植の部品を、(i) 固定のseedの乱数の列（$P=7$、$T=30$）、(ii) 途中で $\theta$ が大きく跳ぶ列（変化を模す）、(iii) 一定の $\theta$ の列（$\hat s$ が0になる場合を含む）、(iv) $\eta_0$ を0.1と0.001、で照合し、各 $t$ の $\eta$ が相対誤差 $10^{-12}$ 以内で一致することを確かめる。
- 構成のtest（要求4.1）: 2構成の一覧、Windowの設定が`feddrift`から割り当ての方式と学習の窓だけ違うこと、Adaptive-FedAvgの設定がWindowから学習率の調整の設定だけ違うこと、組立て、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、依存境界の登録。掃引のオプションの検査（構成を含まないときの例外、非正・非有限の拒否、runの識別子が $\eta_0$ で変わること、既定の $\eta_0$ が保存設定から省かれること）。
- 小さい全体run（要求4.3）: 2構成それぞれで、SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本に固定し、(a) 少なくとも1つのclientでコンセプトが変わる、(b) Adaptive-FedAvgでは、各時刻の各clientの学習の直前のoptimizerがSGD（momentum 0、weight decay 0）で、学習率が、最初の時刻は $\eta_0$、以後は前の時刻にサーバが記録した $\eta$ に等しい。Windowでは、optimizerが`feddrift`の既定のAdamで、学習率がそのデータの`feddrift`の既定のまま、(c) Adaptive-FedAvgでは、記録した $\eta$ の列が、そのrunで観測した各時刻の集約後のパラメータを参照の実装に入れた値と一致する、(d) 学習の直前のモデルの学習の標本が、その時刻の標本だけ、(e) 各clientの保有モデルIDが毎時刻 $\{0\}$、サーバの時刻ごとのモデル数が1、登録の記録が最初の1個だけ、統合の記録が0件、を確かめる。条件(a)を満たさなければ標本数を5000に増やして固定し、Implementation Notesに書く。Adaptive-FedAvgの $\eta$ の最小・最大と、変化の後に $\eta$ が上がった時刻の数を報告する（要求ではない）。
- 構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCAのtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは主担当が流す。

## 調整（要求5.2）
- Adaptive-FedAvgだけ。Windowは調整しない（ほかの手法と同じ学習率）。FedDriftの論文の付録に合わせ、$\eta_0\in\{10^{-1},10^{-2},10^{-3},10^{-4}\}$ を、評価に使わないseed 100〜102で、データ×間隔ごとに選ぶ（10データ×4間隔×4値×3 seed、480 run）。
- 選び方: 3 seedの平均の全体の累積精度が最良の $\eta_0$（通信・計算は $\eta_0$ によらず同じなので、FedDriftのδの0.1ppの幅の規則は使わない）。最良が格子の端（$10^{-1}$ または $10^{-4}$）でも、論文の付録の候補の範囲に固定した選択として広げない（FedDriftのδの上端と同じ扱い、2026-11-01のユーザー判断）。範囲の外が最良である可能性は調べないので、結果の記録と論文では「論文の候補の範囲から選んだ」と書き、最良の学習率とは書かない。端を選んだセルの数を記録する。seedがそろわない値は外す。
- コマンドの形: `--configurations feddrift-adaptive-fedavg-window --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 100-102 --aggregation-intervals 50 100 200 500 --adaptive-fedavg-initial-learning-rates 0.1 0.01 0.001 0.0001 --client-count 10 --per-client-sample-count 5000 --workers 64 --study adaptive-fedavg-lr-tuning-v16 --label lr-grid-seeds100-102`。

## 本規模の実行と集計（要求5.2）
- Window: 10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本（400 run、`--study method-comparison-window-v16`）。調整を待たずに流せる。
- Adaptive-FedAvg: 選んだ $\eta_0$ で同じ条件（400 run、`--study method-comparison-adaptive-fedavg-v16`）。$\eta_0$ ごとにコマンドを分ける（FedDriftのδの本評価と同じ）。
- 失敗したrunがあれば集計を止めて報告する。構成ごとに、データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度、FedHASP（基準版16）・FedDrift・FedAvg（集約ごとの形）・Oracle・IFCAとの差（勝ち/負け）、WindowとAdaptive-FedAvgの差（学習率の調整の効果）、通信、MAC、実行時間、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`と同じ定義）。
- seed: 手元の全手法の評価と同じseed 0〜9の10個（FedDriftの論文の5 trialsではなく、手元の論文の表1の条件に合わせる）。セルごとに10 seedの平均と標本標準偏差、差は同じseedの対で求める。
- 比べる相手（全部seed 0〜9、10 client×5000標本、同じデータ・間隔。論文`FedHASP-paper/ja/sections/05-experiments.tex`の表1の根拠のコメントと同じ結果）: FedHASPは基準版16の既定（6データ`results/fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda`、新しい4データ`results/fedsda-every-sample-change-interval-search/20261029_172159_new-datasets-seeds0-9`）。FedDriftは調整したδ（6データ`results/feddrift-tuned-delta-v15`、新しい4データは調整したδの本評価の結果。届くまでは既定のδの`results/new-datasets-baselines/feddrift`で仮に集計し、その旨を書く）。FedAvgは集約ごとの形（`results/method-comparison-v15/20261029_204550_fedavg-seeds0-9`と`results/new-datasets-baselines`の`fedavg-local-updates-per-aggregation`）。Oracleは`method-comparison-oracle-v16`の`oracle-seeds0-9`、IFCAは`method-comparison-ifca-v16`の`ifca-window-seeds0-9`。表1の根拠が後で差し替わったら、差し替えた後のものを使う。相手のrunが1つでも欠ければ止める。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna、effort medium）FAIL。式・時刻の対応・2構成の定義・既存の構造への接続は整合と判定。Major 2: (1) 公式の`win-1_iter`の更新の時点（最後から5つ目のround）と、手元のround数1の対応の説明が足りない——「論文・公式の実装との違い」の(d)に、更新の頻度と $t$ の数え方は同じで、使う集約の時点だけを置き換えることを書いた。(2) 本規模のseed数と比べる相手の結果が特定されていない——seed 0〜9の理由と、比べる相手の結果のフォルダを書いた。Minor 2: 記録の列と式の $t$ の対応、格子の端を広げない理由と結果の書き方——反映。処理・契約を変えない文面と計画の記述の修正なので、再レビューは省く（2026-10-10の決定）。
- 実装の後（2026-11-01）: 要求2.1・Validationの(b)にWindowの学習率を「0.001」と書いていたが、`feddrift`の既定の学習率はデータごとに違う（SINE2は0.01、MNIST4は0.001）。実装担当は「`feddrift`の既定のまま」の要求に従い、既存の値を保った。文面を直した。処理を変えない文面の修正。
