# Design Document

## Overview
FedAvg+Window（`feddrift-window`）とCFL+Window（`feddrift-cfl-window`）で、1時刻あたりの学習・集約のround数 $R$ を1より大きくできるようにする。サーバの`synchronize_timestep`は既に`local_training_round_count`回「全clientの`train_local_round`→`aggregate_and_distribute_training_round`」を繰り返す作りで、FedDrift本体では $R=2,3$ がgoldenに入っている。今は`feddrift_run_participant_factory.py`の検査が、`minimum_loss_cluster_assignment`・`single_global_model_assignment`・`cfl_bipartition_assignment`で $R\neq1$ を拒否している。$R$ は`feddrift_default_run_settings.py`で1に固定され、掃引の値になっていない。

公式の実装（`results/_sol_prompts/cfl-official-excerpt.py`）のCFLは、毎roundの`aggregate`で`cluster_cfl`を呼び、分割したら`train_data_weights[curr_iter]`（その時刻のclientとモデルの対応）を書き換えて、そのroundの集約を省く。残りのroundでは、clientが新しい対応でその時刻の窓の標本を学習する（`num_sample returned by the client already factors in the sc_state weights`）。手元の`split_cfl_training_round`は $R=1$ の前提で、分割の後に割り当て（`current_training_model_assignment`）だけを変え、その時刻の学習の標本は元のモデルに残る。また学習の対象の集合は`synchronize_timestep`の最初に1回だけ作られ、clientの`train_local_round`へ渡される。

## 設定と掃引
- 掃引の値`--feddrift-local-training-round-counts`（正の整数の列）。DriftSurfのδ（`--driftsurf-accuracy-drop-thresholds`、`experiment_sweep_plan.py`の`driftsurf_accuracy_drop_thresholds`、`experiment_spec/parameters.py`の`driftsurf_accuracy_drop_threshold`）と同じ作りにする。`experiment_spec/options.py`のOptionSpec・ActivationRule・ChoiceConstraint、`parameters.py`、`sweep.py`のSweepAxis（AGENTS.mdの規約）へ登録し、`docs/reference/options.md`は`python -m tools.generate_option_docs`で再生成する。
- 許す構成は`feddrift-window`と`feddrift-cfl-window`だけ。ほかの構成に既定以外の値を指定したら、掃引の計画で拒否する。runの組み立ての検査も、上の2つの割り当ての方式（`single_global_model_assignment`、`cfl_bipartition_assignment`）では $R\ge1$ を許し、`minimum_loss_cluster_assignment`（IFCA）は1だけを許す。エラーの文言を直す。
- 既定は1。既定のrunの保存設定・識別子・goldenを変えない（既存の`local_training_round_count`は既に保存設定に入っているので、値が1なら今と同じになるはずである。実装担当が確かめる）。

## round ごとの挙動（2構成に共通）
- 各roundで、今と同じ`train_feddrift_held_models_for_round`（局所の予算`updates_per_sample`×`k_steps`、バッチ、学習率、件数の条件）を使う。集約の重みはその時刻の窓の件数。通信（上りのパラメータ、下りの配布）と計算の計数は毎round行う。学習の対象の集合（学習と集約に使うモデルIDの集合）と、その通知（メッセージの件数）は別のものとして扱う。学習の対象の集合は毎roundの学習の直前に作り直す（下の「CFLの時刻の途中の分割」）。FedAvg+Windowと、分割の無い時刻のCFLでは、毎round同じ集合になる。通知は今どおり時刻に1回、clientごとに1通と数え、CFLで時刻の途中に集合が変わったときだけ下のとおり足す。
- 窓の標本は時刻の初めにだけ空にする（今の`feddrift_client_stream_processing.py`の位置のまま）。その時刻の $R$ 回のroundは同じ窓の標本を使う。
- optimizerは既定のAdam（AMSGrad、weight decay 0.001、`feddrift_default_run_settings.py`）で、配布のたびに作り直す（D05）。公式の学習器はモデルごとのAdamを初めに1度作り、時刻の中のroundで状態を保つ。$R>1$ ではこの差が残るが、全手法に共通の学習規則であり、同じ $R$ のFedAvg+Windowにも同じく生じるので許容する（2026-11-03、実装時の確認で訂正）。

## CFLの時刻の途中の分割
- 1時刻の中の各round $r=0,\dots,R-1$ は、次の順に行う（公式の「clientの学習→`aggregate`の中の`cluster_cfl`→分割なら集約を省いて返す」と同じ順）。
  1. その時点の割り当てから学習の対象の集合 $E_r$ を作る（`_current_timestep_training_model_ids`）。
  2. 全clientが`train_local_round(eligible_model_ids=E_r)`で学習する。
  3. `aggregate_and_distribute_training_round`の中で、集約の対象の候補（clientの保有モデルIDのうち $E_r$ に入るもの。既存の絞り方）を作り、その学習の後の更新で、既存の`split_cfl_training_round`で分割を判定する（判定の規則・状態・2分割・上限・θ0・集約の省略・記録は今のまま。判定に使うのは、このroundの学習の後のパラメータと、このroundの初めに配布したパラメータの差）。
  4. 分割が無ければ、候補のモデルを通常どおり集約・配布する。分割があれば、そのroundの全モデルの集約を省き、分割したモデルと新しいモデルをθ0にして全モデルを配布し、`cl2`のclientの割り当てを新しいIDに変え、下のとおり標本を付け替える。
  5. 次のround $r+1$ は1.から行う。$E_{r+1}$ は変わった後の割り当てから作るので新しいIDを含み、新しいIDのモデルは全clientに配布済みで保有されているので、3.の候補にも入り、`cl2`のclientの更新で集約される。
  - $E_r$ は、同じroundの学習（2.）と集約の候補（3.）で同じ値を使う。2.と3.の間で割り当ては変わらない。
- 分割したら（分割したroundの集約の省略と配布、割り当ての変更の後に）、割り当てを変えたclient（`cl2`）ごとに、そのclientの`training_sample_store`で元のモデルIDの標本の列を、既存の`ModelTrainingSampleStore.reassign_model_training_samples_id(original_model_id=元のID, reassigned_model_id=新しいID)`で新しいIDへ付け替える。窓1時刻のCFLでは、このstoreは時刻の初めに空になり、その時刻の標本だけを持つので、移るのはそのclientのその時刻の標本だけで、順序も保たれる。新しいIDはそのrunで初めて払い出すIDなので、移す先に既存の列は無い（無いことを検査する）。`cl1`のclientと、ほかのクラスタのclientの列は変えない。
- 件数の記録`model_training_and_assignment_counts_store`は変えない。`trained_sample_counts_by_model_id`と`parameter_update_step_counts_by_model_id`は、実際に学習したroundでそのときのモデルIDに加算する累積の記録なので、分割の前のroundの元のIDへの加算はそのまま残り、分割の後のroundの学習は新しいIDに加算される。`assigned_sample_counts_by_model_and_concept_id`は、時刻の終わりの帰属（`assigned_training_sample_absorption.py`）の時点の診断の記録なので、分割で付け替えない（文書に書く）。
- 学習の対象の集合は上の1.のとおり毎round作り直す（`synchronize_timestep`のroundのループの中で作り、全clientの`train_local_round`と、そのroundの`aggregate_and_distribute_training_round`の両方に同じ値を使う。後者は今は中で`_current_timestep_training_model_ids`を呼び直しているが、同じ割り当てから作るので同じ値になる。引数で渡す形にしてもよい）。
- 時刻の途中で対象の集合が変わったときだけ、変わった後の対象の集合の通知をclientごとに1通数える（メッセージの件数。パラメータのbyteに入れない）。分割で変わった割り当ての通知は、既存の分割の配布に含める（今の $R=1$ と同じ）。
- 次の時刻の予測と割り当ての写し（`cfl_assignment`）は、その時刻の最後のroundの後の割り当てを使う（今の写しの位置のまま）。
- 記録`cfl_split_check`に列`round_index`（その時刻の中のroundの番号、0から）を足す。保存形式の版を7から8へ上げる。ほかの構成の表・列は変えない。naming.mdへ追記する。

## 公式との違い（文書と論文に書く）
- 合わせる点: 毎roundの分割の判定、分割したroundの集約の省略、時刻の途中の割り当て直しと、残りのroundでの新しい対応による学習、窓（その時刻の標本だけを $R$ 回学習）。
- 許容する違い（全手法に共通の条件から来るもの）: (a) 逐次の評価（D12）、(b) 局所の学習の予算・バッチ・学習率（D04・D11。公式の例は1 roundに5エポック）、(c) 分割で使うモデルの初期値をθ0にする（D06。公式は乱数の再初期化）、(d) $R$ の値（公式の実行例は200、論文は100。手元は予備の実行で決める。計算と通信は $R$ に比例して増え、図3で示す）。
- FedAvg+Windowの $R>1$ は、論文のWindowと同じく、その時刻の標本だけを $R$ 回学習する。

## Validation
- 新しいtestは`tests/refactoring/test_feddrift_window_multiple_training_rounds.py`（実装担当が既存の命名に合わせて決めてよい）。
- 既定 $R=1$: 全構成の設定と識別子（例: FedDriftの`sine2_interval50_seed0_1fbf192c89f9`）、基準版1〜16のgolden（Windows）が変わらない。CFL・Window・IFCAの既存のtestが通る。
- 掃引と制約: `--feddrift-local-training-round-counts 1 3`で2構成のrunが値ごとにでき、識別子が値で異なる。ほかの構成（`feddrift`、`feddrift-ifca-window`、`feddrift-driftsurf`など）に3を指定すると掃引の計画で拒否される。組み立ての検査でIFCAの $R=2$ が拒否され、文言が正しい。
- FedAvg+Window $R=3$ の小さい全体run（SINE2、間隔50、seed 0、10 client×2500標本）: 各時刻に学習・集約が3回、各roundの各clientの学習の標本がその時刻の窓の標本と一致、上りと下りのパラメータの通信の回数が同じ条件の $R=1$ の3倍、モデルは1つ。
- CFLのサーバの経路（必須。既存の分割が決定的に起きる入力のtestを $R=3$ に広げる）: 時刻の最初のroundで分割が起き、(1) そのroundの集約の省略、(2) 2・3 round目に`cl2`のclientが新しいIDのモデルを学習し、その学習の標本がそのclientのその時刻の窓の標本と一致し、元のIDの学習の標本に残らない、(3) 1 round目の学習の対象の集合に新しいIDが無く、2・3 round目の`train_local_round`に渡された対象の集合と集約の対象に新しいIDが入り、新しいモデルの集約がその件数の重みで行われる、(3'') 2 round目の集約で、新しいIDのモデルが`cl2`のclientの2 round目の学習の後のパラメータの重みつき平均（重みはその時刻の件数）になり、元のIDのモデルが`cl1`のclientの分だけの平均になる、(3') `model_training_and_assignment_counts_store`の学習の件数とstep数が、1 round目は元のIDに、2・3 round目は`cl2`の分が新しいIDに加算され、帰属の診断の件数は変わらない、(4) `cfl_split_check`の`round_index`、(5) 次の時刻の割り当て・予測のモデル・学習の標本の集合（空から始まる）、(6) 対象の集合の通知の件数、を確かめる。
- CFL $R=3$ の小さい全体run（SINE2とMNIST4、間隔50、seed 0、10 client×2500標本）: 既存の照合（割り当て、窓、予測、判定をノルムと閾値から再計算した結果との一致、上限、検出0・統合0）を、roundごとの判定と時刻の途中の割り当て直しに広げる。分割が起きたかは要求にしない。分割の回数・時刻・roundを報告する。
- 構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・Window・Adaptive-FedAvg・DriftSurf・CFLのtest、`tests/test_regression.py`、Ruff・Pyright。全pytestは主担当が流す。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna、effort medium）FAIL。Major 2: (1) 学習の対象の集合をroundごとに作り直すことと、時刻に1回の通知の区別——「round ごとの挙動」と「CFLの時刻の途中の分割」に明記した。(2) 分割の後の標本の付け替えと件数の記録の扱い——既存の`reassign_model_training_samples_id`で`cl2`のclientのその時刻の列を移し、件数の記録は変えないと定めた。testにも足した。
- 2回目FAIL。Major 2: (1) 判定とroundの順序（学習の後・集約の前に判定し、分割したroundは集約を省き、次のroundから新しい割り当てで学習する）——「CFLの時刻の途中の分割」に1〜5の手順を書いた。(2) 分割の後に新しいIDの更新が集約される経路——同じroundの学習と集約の候補に同じ対象の集合を使い、新しいIDは配布済みで保有されているので次のroundの候補に入ることを明記し、testに足した。

## 予備の実行（要求4.2）
- コマンドの例: `--configurations feddrift-cfl-window feddrift-window --datasets sine2 circle2 mnist4 rotmnist4 --seeds 100-102 --aggregation-intervals 50 500 --feddrift-local-training-round-counts 10 50 --client-count 10 --per-client-sample-count 5000 --workers <数> --study cfl-window-rounds-pilot-v16 --label rounds-10-50-seeds100-102 --purpose "1時刻R roundでCFL+Windowが分割するかと、同じRのFedAvg+Windowとの差を、評価に使わないseedで確かめる"`（96 run、uranium）。計算は $R$ に比例し、$R=1$ の約 $R$ 倍の時間がかかる。主担当が計画（run数と重複先）を内部で確かめてから渡す。
- 集計: runごとの分割の回数・時刻・round、最終のモデルの数、精度、通信量、同じ $R$・同じseedのFedAvg+Windowとの差、$R=1$ の予備の実行との差。間隔50では同じ50標本を $R$ 回学習するので、過学習（$R$ を増やして精度が下がるか）も見る。
- 判断（ユーザー）: CFLが分割し、同じ $R$ のFedAvg+Windowより精度が上なら、$R$ を調整用のseedで選んで本規模（400 run）へ進む。そうでなければ、「$R$ を増やせば分割するが、コンセプトの切り替わりを追えない」などの結果を論文に書く。
