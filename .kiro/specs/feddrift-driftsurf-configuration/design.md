# Design Document

## Overview
手元のFedDriftの実装の上に、DriftSurfの構成`feddrift-driftsurf`を足す。Oracle（`e2dae549`）・IFCA（`7874c449`）・FedAvg+Window／Adaptive-FedAvg（`f89a113b`）の実装で足した設定の項目・分岐・記録の作り（割り当ての方式、学習の窓、モデルの登録、検出・作成・統合を行わない分岐、版つきの記録の表）と、FedDriftの学習・集約・通信の規則を再利用する。変える点は、(1) サーバが役割`pred`・`stab`・`reac`の大域モデルを持ち、全clientが予測の役割のモデルで予測する、(2) 各時刻の終わりに、clientが役割のモデルの正解数を数え、サーバが合計した正解率で公式の状態遷移を行う、(3) 各役割のモデルを、その役割の時刻の窓の標本で学習する。

## 公式の実装（移植の元）
- `FedAvgEnsDataLoader.py`の`DriftSurfState`（`delta`、`r=3`、`wl=10`）。`models`は`pred`・`stab`・`reac`、`train_data_dict`は役割ごとの学習に使う時刻（iteration）の番号の列、`train_keys`は今の時刻に学習する役割、`model_key`は予測の役割、`acc_best`、`state`（`stab`／`reac`）、`reac_ctr`、`acc_dict`。`_score`は役割のモデルの、その時刻の全clientの学習データ（`win-1`の`train_data_global`）での正解率。
- `DriftSurf_data_loader`: 最初の時刻（`curr_train_iteration == 0`）は状態遷移をせず、`pred`と`stab`を時刻0のデータで学習する（`train_data_dict`の初期値は`pred: [0]`、`stab: [0]`）。2つ目の時刻からは、その時刻のデータで`run_ds_algo`を呼んでから、`train_keys`の役割ごとに、その役割の時刻の列のデータで学習する。δはデータごとの既定（sea 0.02、sine 0.10、circle 0.05）か引数。
- `FedAvgEnsAggregatorDriftSurf.py`: `train_keys`の順にモデルを並べ、前の時刻の役割のモデルを引き継ぐ（`None`の役割、つまり作り直した役割は、その時刻に組み立てた新しいモデル）。標本数で重みづけた平均で集約する。テスト（予測）は`model_key`の役割のモデル。
- FedDriftの論文の付録: DriftSurfのδは0.02〜0.20から最良を選び、$r=3$（元の論文の $r=4$ より良かった）。ほかは元の論文の既定。

## 状態遷移（公式の`run_ds_algo`の移植。時刻 $\tau\ge 1$ の終わり、学習の前）
正解率 $a_{\mathrm{pred}}$・$a_{\mathrm{stab}}$・$a_{\mathrm{reac}}$ は、その時刻 $\tau$ の全clientの全標本での正解率（clientごとの正解数と件数を合計して割る。公式の、全clientのデータを集めた`_score`と同じ値）。
1. $a_{\mathrm{pred}}$ を求め、$a_{\mathrm{pred}}>$`acc_best`なら`acc_best`を更新する。
2. 状態が`stab`のとき: `stab`の時刻の列が空なら $a_{\mathrm{stab}}=0$、そうでなければ求める。$a_{\mathrm{pred}}<$`acc_best`$-\delta$ または $a_{\mathrm{pred}}<a_{\mathrm{stab}}-\delta/2$ なら、状態を`reac`にし、`reac`を作り直し（モデルなし・時刻の列を空）、`reac_ctr`$=0$、`acc_dict`を長さ $r$ の0の列にする。そうでなければ、$\tau$ を`pred`と`stab`の時刻の列へ足し、`train_keys`$=$[`pred`,`stab`]。
3. 状態が`reac`のとき（2で入った時刻を含む）: `reac_ctr`$>0$ なら $a_{\mathrm{reac}}$ を求め、`acc_dict`の`reac_ctr`$-1$番目に $a_{\mathrm{pred}}$・$a_{\mathrm{reac}}$ を記録し、$a_{\mathrm{reac}}>a_{\mathrm{pred}}$ なら`model_key`$=$`reac`、そうでなければ`pred`。$\tau$ を`pred`と`reac`の時刻の列へ足し、`train_keys`$=$[`pred`,`reac`]、`reac_ctr`を1増やす。`reac_ctr`$=r$ になったら、状態を`stab`にし、`stab`を作り直し、`acc_dict`の`pred`の平均 $<$ `reac`の平均なら、`pred`のモデルと時刻の列を`reac`のものに置き換え、`acc_best`$=\max$(`acc_dict`の`reac`)、`model_key`$=$`pred`。
4. 時刻の列へ足すとき、長さが $w=10$ を超えたら先頭を捨てる（`_append_train_data`）。
- 作り直した役割のモデルは、次に学習するときに、全手法に共通の事前学習した初期モデルの写しから始める（公式の、その時刻に組み立てた新しいモデルに当たる。FedDriftの論文の「1つの手法の中で作るモデルは同じ初期値」）。
- **公式からの1点の変更（文書に書く）**: 反応の状態を抜けて昇格しなかったときに`model_key`が`reac`のまま残ると、公式の集約器は次の時刻の`train_keys`（[`pred`,`stab`]）に予測の役割が無く、テストのモデルを決められない（`FedAvgEnsAggregatorDriftSurf.init_ds_state`の`test_model_idx`が設定されない）。手元では、反応の状態を抜けたときは昇格の有無によらず`model_key`$=$`pred`にする（元のDriftSurfの論文の、安定の状態では予測のモデルを使うことに合う）。
- 予測: 時刻 $\tau+1$ の全標本を、時刻 $\tau$ の遷移と学習の後の`model_key`の役割のモデルで予測する（逐次の評価）。最初の時刻は`pred`。

## 正解率が公式と同じになる条件と、初期値の公式との差（仕様レビュー1回目のMajor 1への答え）
公式の実装の写しは`results/_sol_prompts/driftsurf-official-excerpt.py`（git管理外。作業場所の外にあるcloneから写した）。
- 正解率: 公式の`_score`は、役割のモデルを`eval()`にし、`no_grad`で、その時刻の全clientの学習データ（`DriftSurf_data_loader`の`win-1`の`train_data_global`）の全標本を推論し、`torch.max(pred, -1)`の添字が観測ラベルと等しい数を全標本数で割る（標本が0なら0）。手元は、時刻 $\tau$ の終わりに、各clientが、その時点で保有する役割のモデル（前の時刻の配布の後、時刻 $\tau$ の学習の前のパラメータ。役割のモデルは大域モデルなので全clientで同じ値）を評価モードで、そのclientの時刻 $\tau$ の全標本（予測と学習に使うのと同じ入力。追加の前処理なし）で推論し、出力の最大の添字（同値は小さい添字。`torch.max`と同じ）が観測ラベルと等しい数と標本数を報告する。サーバは全clientの正解数の和を標本数の和で割る。全clientの時刻 $\tau$ の標本の和集合が`train_data_global`に当たるので、整数の和で同じ値になる。
- 作り直したモデルの初期値: 公式は時刻ごとにプロセスを起動し直し、固定のseed（`torch.manual_seed(args.dummy_arg)`）でモデルを組み立て、前の時刻のモデルが無い役割（作り直した役割）はその組み立てたモデルのまま使う（`main_fedavg.py`と`init_ds_state`）。FedDriftの論文は「1つの手法の中で作るモデルは同じ初期値」と書く。手元は、runの最初に、全手法に共通の事前学習した初期モデルのパラメータ $\theta_0$ を写しとして保存し、最初の`pred`・`stab`と、作り直した役割のモデルを $\theta_0$ から始める（optimizerは新しく作る）。公式では最初の`pred`と`stab`が同じseedの続けての組み立てで違う初期値になるが、手元は共通の事前学習（D06）に合わせて両方 $\theta_0$ にする（文書に書く）。

## 学習（仕様レビュー1回目のMajor 2への答え）
- 標本の保存: 各clientは、観測した標本を時刻の番号ごとに保存する（時刻 $\tau$ の $I$ 標本を観測の順に）。
- 役割とモデルの識別子: runの最初に`pred`を識別子0、`stab`を識別子1として登録する。役割を作り直すときは、新しい識別子を登録して $\theta_0$ を入れ、その役割の古い識別子があれば登録を外す（サーバとclientから除く）。昇格のときは、`pred`の役割を`reac`の識別子へ結び直し、古い`pred`の識別子の登録を外し、`reac`の役割は識別子なしにする（次に反応の状態へ入るときに作り直す）。反応の状態を抜けて昇格しないときは、`reac`の識別子をそのまま残す（学習も予測もしない。次に反応へ入るとき作り直して外す）。時刻ごとの役割と識別子の対応を記録する。
- 時刻 $\tau$ の処理の順: (1) 時刻 $\tau$ の標本を、時刻 $\tau-1$ の終わりに決めた`model_key`の役割のモデルで予測する（最初の時刻は`pred`）。(2) 時刻の終わりに、各clientが標本を時刻 $\tau$ として保存し、$\tau\ge1$ なら役割のモデルの正解数と件数を報告する（求めるのは`pred`、状態が`stab`で`stab`の時刻の列が空でなければ`stab`、反応の状態で`reac_ctr`$>0$ なら`reac`。公式が`_score`を呼ぶものだけ）。(3) サーバが状態遷移を行い（$\tau=0$ は行わず、`pred`・`stab`の時刻の列を[0]、`train_keys`を[`pred`,`stab`]にする）、`train_keys`、各役割の時刻の窓、作り直し・昇格を決めてclientへ知らせる。(4) 各clientは、`train_keys`の各役割の識別子 $m$ について、学習の標本の集合を「窓に入る時刻の自分の標本を時刻の昇順に連結したもの」にし、ほかの識別子の集合は空にする。(5) 既存の`train_feddrift_held_models_for_round`を、学習の対象を`train_keys`の識別子に限って呼ぶ（件数が`batch_sample_count`以上で参加。窓は1時刻以上で $I\ge50>32$ なので全clientが参加する。更新の回数はFedDriftと同じく`updates_per_sample`×`k_steps`で、窓の長さによらない）。(6) サーバは学習した識別子をclientごとの学習の標本の件数（窓の標本数。全clientで同じ）で重みづけて集約し、全clientへ配布し、次の時刻の`model_key`を知らせる。
- round数・optimizerとその作り直し・学習率は、FedDriftの構成`feddrift`の規則（D04・D05・D11）。
- **窓と既存のstoreの接続（仕様レビュー2回目のMajorへの答え。実装の契約）**: 既存の`ModelTrainingSampleStore`はモデルの識別子ごとに標本を追記するだけで、窓で切り出さない。そこで、(i) clientに、時刻の番号→その時刻の標本の列を持つ「時刻の標本の保管」を新しく足す（窓は直近 $w$ 時刻の部分集合なので、直近 $w$ 時刻だけ保つ。`pred`の窓は毎時刻その時刻を足すので常に直近の連続した時刻、`stab`・`reac`の窓はその部分集合）。(ii) DriftSurfの構成では、検出バッチの処理で`ModelTrainingSampleStore`へ標本を追記しない（FedDriftの割り当ての経路を通さない）。(iii) 各時刻の学習の直前に、`ModelTrainingSampleStore.clear_model_training_samples`（IFCAで足したもの）で全識別子の集合を空にし、`train_keys`の各役割の識別子へ、時刻の標本の保管から窓に入る時刻の標本を時刻の昇順に追加する。これで、学習関数が使う「保存済みの標本の全体」は、その役割の窓の標本だけになる。(iv) 集約の重みは、その時刻の学習の直前の、clientごと・識別子ごとの`ModelTrainingSampleStore`の件数（窓の標本数）にする。FedDriftの累積の重み（D08の「保存済み重み」や累積の学習標本数）は使わない（IFCA・FedAvg+Windowの窓1時刻の重みの扱いと同じ接続）。全clientが同じ時刻の数を持つので、重みは全clientで等しい。(v) 予測の記録や`ModelTrainingAndAssignmentCountsStore`の割当の数え方は、学習した識別子の件数として既存の規則で数え、DriftSurfの判断には使わない。
- testの確認: 小さい全体runの(d)で、学習の直前の各識別子の集合の標本の時刻の番号が、記録した窓の時刻の集合と一致し（窓の外の標本が無い）、集約の重みが窓の標本数と一致することを確かめる。部品のtestで、時刻の標本の保管が直近 $w$ 時刻だけを保ち、窓に入る時刻の標本を昇順に返すことを確かめる。

## 論文・公式の実装との違い（文書と論文に書く）
- 準じる点: 3つの役割のモデル、状態遷移の規則（δ、δ/2、$r=3$、`acc_best`、昇格）、窓 $w=10$ 時刻、全clientのデータを合わせた正解率（手元はclientの正解数の合計で、データをサーバへ送らずに同じ値を得る）、作り直したモデルの初期値が手法の中で共通、δを0.02〜0.20から選ぶこと。
- 違う点: (a) 逐次の評価（D12）。(b) 共通の事前学習の初期モデル（D06）。(c) 局所の学習の規則（D04・D05・D11）。(d) 1時刻＝集約の間隔の標本数なので、窓10時刻は間隔50で500標本、間隔500で5000標本（全部）。(e) 上の`model_key`の1点の変更。(f) δはデータ×間隔ごとに選ぶ（FedDriftのδと同じ単位）。
- 通信: 各時刻、学習した役割のモデルの上り・下りはFedDriftの規則で数える。正解数と件数の報告は、割当のIDの報告と同じくclient 1つにつき1通（パラメータのバイトに入れない）。

## Architecture（実装担当が既存の構造を調べて具体化する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_DRIFTSURF_CONFIGURATION_NAME`（値`feddrift-driftsurf`）を`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ。構成`feddrift`を元に、割り当ての方式を新しい値（例: `driftsurf_global_role_assignment`）、学習の窓を新しい値（例: `role_timestep_window`）、DriftSurfの設定（新しい項目。例: `driftsurf_settings`にδ・$r$・$w$）にする。設定の項目は`experiment_spec`の規約（`OptionSpec`・`ActivationRule`・`ChoiceConstraint`、数値は`parameters.py`）。既定値は保存設定から省き、既存の識別子を変えない。
- 掃引: `--driftsurf-accuracy-drop-thresholds`を`--feddrift-loss-increase-thresholds`・`--adaptive-fedavg-initial-learning-rates`と同じ作りで足す（構成を含まない掃引では例外、値は有限で正）。
- 状態遷移の部品は、runtimeから独立した小さいクラスにし（正解率を引数に受け、`train_keys`・`model_key`・各役割の時刻の列・作り直し・昇格の指示を返す）、部品のtestで参照の実装と照合できるようにする。
- client: 各時刻の終わり、学習の前に、保有する役割のモデルのその時刻の標本での正解数と件数を数えてサーバへ報告する（FedDriftの検出の評価の作りを使ってよい。予測の記録とは別に数える）。
- server: 正解数と件数を合計して正解率を求め、状態遷移を行い、学習する役割と窓を決めて配布し、学習・集約の後、予測の役割を全clientへ知らせる。FedDriftのクラスタリング・統合はしない。
- 記録: 時刻ごとのDriftSurfの記録の表を足す（列の例: `timestep_index`、`state_before`・`state_after`、`accuracy_pred`・`accuracy_stab`・`accuracy_reac`（求めないときはNaN）、`accuracy_best`、`training_roles`、`prediction_role`、`reactive_counter`、`promoted`、役割ごとのモデルの識別子と時刻の窓の先頭・末尾）。表の名前と列は実装担当が既存の規約に合わせて決め、naming.mdへ追記する。保存形式の版を上げる必要があれば上げ、既存の構成の表と列は変えない。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_driftsurf_configuration.py`。
- 部品のtest（要求2.2）: 公式の`DriftSurfState`の`__init__`・`_append_train_data`・`_reset`・`run_ds_algo`をtestの中に写し、`_score`だけを、与えた正解率の列を返す関数に差し替え、上の`model_key`の1点の変更を同じく入れた参照の実装と、移植の部品を照合する。入力の列: (i) 安定のまま、(ii) `acc_best`$-\delta$ で反応へ入り昇格する、(iii) `stab`$-\delta/2$ で反応へ入り昇格しない、(iv) 反応の途中で`model_key`が入れ替わる、(v) 窓の長さが10を超える、(vi) 反応を2回繰り返す、(vii) 乱数の正解率の列（固定seed、長さ100、δ 0.02と0.10）。各時刻の状態・`train_keys`・`model_key`・各役割の時刻の列・`acc_best`・`reac_ctr`・作り直しと昇格が一致すること。
- 構成のtest（要求2.1）: 構成の一覧、設定が`feddrift`から割り当ての方式・学習の窓・DriftSurfの設定だけ違うこと、組立て、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、依存境界。掃引のオプションの検査（構成を含まないときの例外、非正・非有限の拒否、runの識別子がδで変わること、既定のδが保存設定から省かれること）。
- 小さい全体run（要求2.3）: 次の3 runに固定する（事後に条件を変えない）。SINE2・δ0.10、MNIST4・δ0.10、SINE2・δ0.02。どれも間隔50、seed 0、10 client、各client 2500標本。3 runを合わせて、少なくとも1回反応の状態へ入り、少なくとも1回昇格することを受け入れの条件にする（満たさなければ失敗として報告し、条件を変えて流し直さない。状態遷移の網羅は部品のtestで行う）。各runで、(a) 少なくとも1つのclientでコンセプトが変わる、(b) 記録した正解率が、その時刻の全clientの標本を連結し、学習の前の役割のモデルのパラメータで1回推論し直して数えた正解数÷標本数（公式の`_score`と同じ計算）と一致し、各clientが保有する役割のモデルのパラメータが全clientで同じ、(b') 作り直した役割のモデルの、最初の学習の直前のパラメータが $\theta_0$ と同じ、(c) 記録した状態遷移が、記録した正解率を参照の実装に入れた結果と一致する、(d) 学習の直前の各役割のモデルの学習の標本が、その役割の時刻の窓に入る時刻の標本だけ、(e) 予測に使ったモデルが`model_key`の役割のモデル、(f) 大域モデルの数が毎時刻3以下で、統合の記録が0件、を確かめる。反応へ入った回数と昇格の回数を報告する。
- 構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・FedAvg+Window・Adaptive-FedAvgのtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは主担当が流す。

## 調整（要求3.2）
- FedDriftの論文の付録に合わせ、δ $\in\{0.02,0.04,\dots,0.20\}$ を、評価に使わないseed 100〜102で、データ×間隔ごとに選ぶ（10データ×4間隔×10値×3 seed、1,200 run）。選び方はFedDriftのδと同じ（最良の平均精度から0.1ppの幅で通信が最小。seedがそろわない値は外す）。上端0.20は論文の範囲として広げない（FedDriftのδと同じ扱い）。ただし、0.20を選んだセルで0.18→0.20の平均精度の上がり幅が1pp以上なら、本評価の前にそのセルの一覧をユーザーに示し、広げるかを決めてもらう（Adaptive-FedAvgの扱い）。下端の0.02は論文の範囲の下端なので広げない。

## 本規模の実行と集計（要求3.2）
- 選んだδで、10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本（400 run、`--study method-comparison-driftsurf-v16`）。δごとにコマンドを分ける。
- 失敗したrunがあれば集計を止めて報告する。データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度と安定期の精度、FedHASP（基準版16）とほかの比較手法との差（勝ち/負け）、通信、MAC、反応へ入った回数と昇格の回数。比べる相手の結果は論文の表1と同じもの（主担当のscratchpadの`all_methods_compare.py`の入力）。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna、effort medium）FAIL。レビューの環境から作業場所の外の公式の実装を読めなかった。Major 2: (1) 作り直したモデルの初期値と、clientの正解数の合計が公式と同じになる条件と、それを確かめるtestが無い——「公式と同じになる条件」の節と、小さいrunの(b)・(b')を足した。公式の写しを作業場所の中（`results/_sol_prompts/driftsurf-official-excerpt.py`）に置いた。(2) 役割の時刻の窓を学習の標本の集合・識別子・時刻の処理の順へ結ぶ規則が未確定——「学習」の節に、標本の保存、役割と識別子の対応（作り直し・昇格）、時刻の処理の順、参加の条件・予算・重みを書いた。ほか: 小さいrunの条件を事前に3 runに固定し、事後に変えないことにした。δの上端の扱いを、1pp以上の上がり幅でユーザーに確かめる条件として書いた。
- 2回目FAIL。1回目のMajor 1（正解率と初期値の条件）は解消と判定。`model_key`の変更は妥当と判定。Major 1件: 既存の`ModelTrainingSampleStore`で窓10時刻をどう実現し、窓の外の標本を学習と集約の重みから除くかが未定義——「学習」の節に「窓と既存のstoreの接続」（時刻の標本の保管、storeへの追記をしない、学習の直前に空にして窓の標本だけを入れる、重みは窓の標本数）と、そのtestを書き、tasks.mdのTask 1にも書いた。
- 3回目FAIL。1回目・2回目のMajor（正解率と初期値の条件、窓と既存のstoreの接続）は解消と判定。Major 1件「δの調整が200 run（SINE2・MNIST4、10 seed）で、データ×間隔ごとの選択と合わない」は、レビューの読み違いとして退けた（「調整」の節は10データ×4間隔×10値×3 seedの1,200 run）。Minor「『公式と同じになる条件』の見出しが初期値まで一致すると読める」——見出しを「正解率が公式と同じになる条件と、初期値の公式との差」に直した。処理・契約を変えない文面の修正なので、再レビューは省いて仕様を確定する（2026-10-10の決定）。
