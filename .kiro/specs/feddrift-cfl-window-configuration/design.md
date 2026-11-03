# Design Document

## Overview
手元のFedDriftの実装の上に、CFL+Windowの構成`feddrift-cfl-window`を足す。公式の実装は`results/_sol_prompts/cfl-official-excerpt.py`（gitの管理外。FedDriftの公式のコードからの写し）にある。IFCAの構成で足した学習の窓（`training_sample_window=current_timestep`）と、1時刻1roundの学習・集約への接続を再利用する。変える点は、(1) 割り当てを時刻をまたいで保ち、分割のときだけ変える、(2) 集約のときに公式の`cluster_cfl`の規則で分割を判定し、分割したroundは集約を省く、(3) 分割で使う2つのモデルを共通の事前学習の初期モデルにする、の3点。

## 公式の規則（`cluster_cfl`。この順で、使われているクラスタをモデルIDの昇順に処理する）
- 状態（run全体で1つ、サーバが持つ）: `cfl_norm`（観測した平均の更新のノルムの最大。初期値0）、`eps1`（初期値0）、`eps2`（初期値10000）、次に使うモデルID（初期値1）。
- 各クラスタ $m$ について: そのクラスタのclientのうち、そのroundの学習の標本が1件以上のclientの更新 $\Delta_c=\theta_c-\theta_m$（全パラメータをstate_dictの順に平らに連結。bufferを含むかは公式の`state_dict`に合わせる）を集める。`max_norm`$=\max_c\|\Delta_c\|$、`mean_norm`$=\|\frac{1}{n}\sum_c\Delta_c\|$。
  - `mean_norm > cfl_norm` なら、`cfl_norm=mean_norm`、`eps1=cfl_norm/10`、`eps2=6*eps1` とし、このクラスタは分割しない。
  - そうでなく、`mean_norm < eps1` かつ `max_norm > eps2` なら、余弦類似度の行列 $S_{ij}=\Delta_i\cdot\Delta_j/(\|\Delta_i\|\|\Delta_j\|+10^{-12})$ を作り、$-S$ を距離とする完全連結の階層的クラスタリングで2つ（`cl1`＝ラベル0、`cl2`＝ラベル1）に分け、$\alpha_{\mathrm{cross}}=\max_{i\in cl1,j\in cl2}S_{ij}$ とする。$\sqrt{(1-\alpha_{\mathrm{cross}})/2}>\gamma$ で、次に使うモデルIDが上限 $K$ 未満なら分割する。`cl1`のclientを $m$ に、`cl2`のclientを新しいID（次に使うモデルID。使ったら1増やす）に割り当てる。
- 分割が1つでも起きたら、そのroundは全モデルの集約を行わず、配布したパラメータのままにする（公式の`aggregate`の`did_split`）。公式は、分割したクラスタのモデル $m$ を乱数で再初期化し、新しいIDのモデルは未使用の枠の構築時の乱数の初期値のまま使う。**手元はここだけ公式から変え**、どちらも共通の事前学習の初期モデル $\theta_0$ の写しにする（「元の実装との違い」の(c)）。
- 窓（公式の`cluster_cfl_init`と`set_weights_win1`）: 各時刻の始めに、前の時刻の割り当て（直前の集約で分割があれば分割の後の割り当て）をそのまま写し、その後で全モデルの学習の標本の集合を空にする（前の時刻までの標本を学習に使わない）。その時刻の標本は、時刻の終わりに割り当てのモデルへ帰属させ、その時刻の学習に使う。IFCAの構成の窓（`training_sample_window=current_timestep`）と同じ空にし方を使い、割り当ての写し→空にする→その時刻の標本の帰属→学習の順にする。
- 2分割のラベル: runtimeではscikit-learnを使わない（実行環境に無い）。公式の`AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="complete").fit(-S)`と同じ併合の順・同じラベル（scikit-learnの木の切り方`_hc_cut`の規則で0と1を付ける）になる実装を、実装担当が書く。testでは、scikit-learnを開発用の依存（`requirements-dev.txt`）に加え、乱数の行列と同点を含む行列で、ラベルとclientの対応が`AgglomerativeClustering`と一致することを確かめる。ラベルは、どちらのclientが元のIDに残るかと、その後のクラスタの処理の順（IDの昇順）を決めるので、結果に影響し得る。
- 更新を出したclientが1つもないクラスタ（窓1時刻では起きない）は、判定を飛ばす（公式は空の列で例外になる）。clientが1つのクラスタは、平均と最大のノルムが等しいので分割の条件を満たさない。
- モデルの上限: 公式の`model_num`は`concept_num`で、手元は $K$（`concept_count`）。IDは0〜$K-1$を順に払い出し、上限に達したら分割しない（公式の`find_unused_model_capped`が $-1$ を返す場合）。

## 元の実装との違い（文書と論文に書く）
- 準じる点: クラスタの割り当ての保持、窓1時刻、分割の判定（閾値の更新、ノルムの条件、余弦類似度、完全連結の2分割、$\gamma=0.1$）、分割したroundの集約の省略、最初は全clientが1つのクラスタ、モデルの上限を $K$ とすること。
- 違う点: (a) 逐次の評価（D12）。(b) 1時刻1round（公式の例は1時刻200 round・5エポック）。分割の判定は集約のたびなので、同じ時刻のデータで何度も判定する公式と違い、手元では時刻ごとに新しいデータで1回判定する（判定の回数は1 runで $5000/I$ 回、公式は10時刻×200回）。全手法が同じround数・更新予算を使う条件（D11）を保つため。機能するかは本規模の前の予備の実行で確かめる（「本規模の実行」）。(c) 共通の事前学習の初期モデル（D06）。分割で使う2つのモデルを、どちらも $\theta_0$ の写しにする。公式は、分割したクラスタのモデルを`reinitialize`（乱数の再初期化）し、新しいIDのモデルは、まだ使っていない枠の構築時の乱数の初期値のまま使う。手元は全手法が同じ事前学習の初期モデルから始まる条件に合わせる。(d) 学習の規則（D04・D05・D11）。(e) 予測のモデルは割り当てたクラスタのモデル（公式の`get_test_model_idx`と同じ）。

## Architecture（実装担当が既存の構造を調べて具体化する。IFCAの実装で足した項目と分岐を再利用する）
- `sweep/experiment_configurations.py`: 定数`FEDDRIFT_CFL_WINDOW_CONFIGURATION_NAME`（値`feddrift-cfl-window`）を`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。構成`feddrift`の設定を元に、割り当ての方式の項目を新しい値（例: `cfl_bipartition_assignment`）、学習の窓を`current_timestep`にする。$\gamma$ は設定の項目（既定0.1、新構成だけで有効）として`experiment_spec`の規約で登録するが、掃引軸は作らない。
- 割り当て: clientは時刻の終わりに、今の割り当てのモデルへその時刻のバッチを帰属させる（全モデルの学習の標本を空にしてから追加。IFCAと同じ窓）。最初の時刻から全clientがID 0。
- サーバ: 集約の前に上の規則で分割を判定し、分割があればそのroundの集約を省き、新しいモデルを登録し（IDは次に使うモデルID）、clientの割り当てを更新して配布する。クラスタリング・統合（FedDriftの`prepare_timestep`の処理）は行わない。
- 学習の対象・件数の条件・集約の重み・通信: IFCAの構成と同じ接続（学習の対象は各clientの割り当てたモデルIDの和集合、各clientでは割り当てたモデルだけが件数の条件を満たせば学習、集約の重みはその時刻の件数、割当IDの報告はclientごとに1通）。分割のroundも、clientの学習と上りの通信は起きる（公式どおり）。
- 記録: 集約ごと・クラスタごとの判定を記録する表（例: `cfl_split_check`。列の例: round（時刻）、モデルID、clientのID、`mean_norm`、`max_norm`、判定の前後の`cfl_norm`・`eps1`・`eps2`、分割を調べたか、$\alpha_{\mathrm{cross}}$、分割したか、新しいモデルID、`cl1`・`cl2`のclient）と、時刻ごとのclientの割り当ての表。表の名前と列は実装担当が既存の記録の規約に合わせて決め、naming.mdへ追記する。保存形式の版を上げる（今は6）。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_cfl_window_configuration.py`。
- 部品のtest（要求2.2）: 公式の`cluster_cfl`の判定（閾値の更新、ノルムの条件、類似度、2分割、$\alpha_{\mathrm{cross}}$、上限、`did_split`）をtestの中に写した参照の実装と、移植の部品を照合する。入力の列: (i) 平均のノルムが増え続ける（閾値の更新だけ）、(ii) 平均が小さく最大が大きいが $\alpha_{\mathrm{cross}}$ が大きい（分割しない）、(iii) 2つの向きに分かれた更新（分割する）、(iv) 上限 $K$ に達している（分割しない）、(v) 2つのクラスタを同じroundに調べ、IDの昇順で処理したときと逆順で処理したときで判定が変わる入力（前のクラスタで更新した`cfl_norm`が後のクラスタの判定を変える）で、昇順の結果と各段階の`cfl_norm`・`eps1`・`eps2`が参照と一致する、(vi) 学習の標本が0のclientを含む、(vii) 乱数の更新（固定seed、client 10、次元20、20 round）。各roundの状態・判定・ラベル・割り当てが一致すること。2分割のラベルとclientの対応は、scikit-learnの`AgglomerativeClustering`と、乱数の行列（30個）と同点を含む行列で照合する。
- 小さい全体run（要求2.3）: SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本の2 runに固定し、(a) 最初の時刻は全clientがID 0、(b) 各時刻の各clientの割り当ては、直前の集約で分割が無ければ前の時刻と同じで、分割があれば分割の記録のとおり（`cl1`は元のID、`cl2`は新しいID）、(c) 各時刻の各clientの予測に使ったモデルIDが、その時刻の割り当てと同じ、(d) 学習の直前のモデルごとの学習の標本が、その時刻の範囲の標本だけ（前の時刻の標本を含まず、その時刻の標本の欠けも無い）で、割り当てていないモデルの集合は空、(e) 記録した判定が、記録したノルム・閾値を参照の規則に入れた結果と一致する、(f) 分割したroundの後に配布したパラメータが、分割したモデルと新しいモデルは $\theta_0$ と同じで、ほかの全モデルはそのroundの前に配布したパラメータと同じ（そのroundのclientの更新が1つも反映されない）、(g) モデルの数が $K$ 以下、統合の記録が0件、を確かめる。分割が起きたかどうかは要求にしない（判定の網羅は部品のtest）。分割の回数を報告する。- サーバの集約の経路のtest（必須。分割が決定的に起きる入力で）: 実際のサーバの集約の関数に、2つの向きに分かれた更新を出すclientの局所のパラメータ（例: 1つのクラスタのclientのうち半分が $+v$、半分が $-v$ の更新で、`cfl_norm`を前のroundで大きくしておく）を与え、同じ1つの検証の中で、(1) 分割が起き、判定の記録が参照の規則と一致する、(2) 分割の後の割り当てが`cl1`は元のID、`cl2`は新しいID、(3) 分割したモデルと新しいモデルの配布のパラメータが $\theta_0$ と同じ、(4) ほかの全モデルの配布のパラメータがそのroundの前と同じ（clientの更新が反映されない）、(5) 次の時刻の始めに、各clientの割り当て・予測のモデル・学習の標本の集合（空）が分割の後の割り当てに従う、を確かめる。
- 構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・DriftSurfのtest、基準版1〜16のgoldenの照合（Windows）、Ruff・Pyright。全pytestは主担当が流す。

## 本規模の実行（要求3.2）
- 予備の実行（1時刻1roundで分割が機能するかの確認）: `--configurations feddrift-cfl-window --datasets sine2 circle2 mnist4 rotmnist4 --seeds 100-102 --aggregation-intervals 50 500 --client-count 10 --per-client-sample-count 5000 --workers <数> --study cfl-window-pilot-v16 --label split-check-seeds100-102 --purpose "1時刻1roundのCFL+Windowで分割が起きるかと精度を、評価に使わないseedで確かめる"`（24 run）。主担当が、runごとの分割の回数・時刻・最終のモデルの数・精度（FedAvgとFedDriftとの差）を集計してユーザーに示し、本規模へ進むか、round数を増やす派生の構成を作るかをユーザーが決める（分割が1回も起きない、または全runで最初の数回の集約で上限まで分割するなら、機能していないと判断する材料にする）。
- 本規模: `--configurations feddrift-cfl-window --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers <数> --study method-comparison-cfl-v16 --label cfl-window-seeds0-9 --purpose "FedDriftの学習・集約の上で、CFL+Window（γ=0.1、モデルの上限は真のコンセプト数）を10データで評価する"`（400 run）。
- どちらも、コマンドを渡す前に主担当が内部で計画（run数と重複先）を確かめ（ユーザーに渡すコマンドには計画の確認の引数を含めない）、開始・結果を`docs/experiments/experiment-log.md`の台帳に記録する。workersはuraniumの負荷に合わせて主担当が決める。

## 本規模の集計
- 失敗したrunがあれば集計を止めて報告する。データ×間隔の40セルごとに、10 seedの平均±標準偏差の累積の精度、論文の表1の各手法との差、通信、分割の回数とモデルの数、変化の直後の $I$ 標本とそれ以外の精度。

## 仕様レビューの経緯
- 1回目（GPT-6 Luna、effort medium）FAIL。Major 7: 分割で使うモデルの初期値の違いの明記、割り当ての保持・窓のtest、集約の省略のtest、クラスタの処理の順で判定が変わる入力、2分割のラベルの再現、上限K、実行の規約。全部を反映した。
- 2回目FAIL。Major 3: (1) 窓の空にし方と割り当ての写しの順序——「窓」の項を足した。(2) 分割が決定的に起きる入力でのサーバの集約の経路のtest——必須のtestとして足した。(3) 初期値の違いが違いの一覧に無い——一覧の(c)には書いてあったので読み違いと判断したが、「公式の規則」の節にも明記した。
- 3回目PASS。Minor 1（更新のないクラスタの扱いは公式では例外になる。窓1時刻では起きないので結果に影響しない）。変えない。
- 2026-11-03のユーザーの質問（1時刻1roundで分割が機能するか）を受け、本規模の前に予備の実行（24 run）を置き、その集計を見て進むかをユーザーが決めることにした（「本規模の実行」）。
