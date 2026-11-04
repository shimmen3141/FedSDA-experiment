# Design Document

## Overview
手元のFedDriftの実装の上に、FedCCFA（Chen et al., NeurIPS 2024）の4構成を足す。FedEM（`.kiro/specs/feddrift-fedem-configuration/`）と同じく、公式の手順に準じ、全手法に共通の条件（逐次の評価、事前学習の初期モデル、学習の予算・optimizer）とストリームへ移すための違いだけを足す。FedCCFAは、モデルの集合をコンセプトごとに分ける既存の比較手法と違い、(1) 表現部を全clientで1つ共有し、(2) clientごとの個人の分類層を、クラスごとのクラスタで部分的に平均し、(3) 表現部をクラスのアンカーへの対照の整合で学習する。FedDriftの検出・新しいモデルの作成・統合は行わない。

## 公式のFedCCFAの手順（`C:\Users\yshin\Research\FedSDA-experiment\FedCCFA`、`647d8b0`）
行は公式のファイルの行。主担当が確認した。
- 設定（`configs/FedCCFA.yaml`）: `rounds: 200`、`rep_epochs: 5`、`clf_epochs: 1`、`balanced_epochs: 5`、SGD（`momentum: 0.9`、`weight_decay: 1e-5`）、`batch_size: 64`、`rep_lr: 0.01`、`clf_lr: 0.1`、`balanced_clf_lr: 0.1`、`gamma: 20.0`、`penalize: CL`、`temperature: 0.1`、`eps: 0.1`、`weights: uniform`、`clustered_protos: true`。論文の探索は $\epsilon\in\{0.05,0.10,0.15,0.20\}$、$\gamma\in\{5,10,20,50,100\}$（`FedCCFA_paper/sections/appendix.tex:134`）。
- roundの流れ（`methods/FedCCFA.py:46-100`）: サーバが選んだclientへ全パラメータ（表現部と、サーバのモデルの分類層＝更新されない初期の分類層 $\phi^{(0)}$）を送る。各clientで`update_label_distribution`→`balance_train`（診断用の分類層を保存）→`train_with_protos`。サーバで`aggregate_rep`→`aggregate_protos`→`send_rep_params`→`merge_classifiers`→クラスごと・組ごとに`aggregate_label_params`と`aggregate_label_protos`を戻す→各clientの`p_clf_params`（個人の分類層）を保存。
- client（`entities/FedCCFA.py`）:
  - `update_label_distribution`（31-42行）: 学習のデータのラベルの割合 $p$ のエントロピー $H=-\sum p\log p$ から、整合の重み $H/\gamma$。
  - `balance_train`（159-186行）・`class_balance_sample`（188-201行）: 各クラスから $\min(\text{最少クラスの件数},5)$ 件を非復元で抜いた均衡の部分集合で、分類層だけ（表現部は固定）を`balanced_epochs`回学習する。分類層は送られてきた $\phi^{(0)}$ から始まる。
  - `train_with_protos`（81-158行）: 分類層を`p_clf_params`へ戻し（初回は空なので、`balance_train`の後の分類層のまま）、表現部を固定して分類層を`clf_epochs`、分類層を固定して表現部を`rep_epochs`学習する。表現部の損失は、教師ありの損失＋整合の重み×`CrossEntropy(cos(特徴, 各クラスのアンカー)/temperature, ラベル)`（`penalize: CL`）。整合は`len(global_protos)>0 and _round>=20`のときだけ（131行）。学習後、学習のデータで各クラスの平均の特徴（アンカー）を求める（`get_local_protos`、203-229行）。
- server（`entities/FedCCFA.py`）:
  - `aggregate_rep`（定義277行、処理277-294行）: 表現部をclientのデータ件数の重みで平均。
  - `madd`（定義335行、処理334-349行）・`merge_classifiers`（353-388行。DBSCANは370-371行、2 client以上の組だけを返すのは379-385行）: クラス $c$ ごとに、各clientの診断用の分類層の $c$ 行（重みとバイアスを連結）の`madd`距離（334-349行、$D(i,j)=\frac{1}{n-2}\sum_{z\ne i,j}|\cos(v_i,v_z)-\cos(v_j,v_z)|$）で`DBSCAN(eps, min_samples=1, metric="precomputed")`。2 client以上の組だけを返す。
  - `aggregate_label_params`（420-445行）・`aggregate_label_protos`（定義448行、処理448-476行）: 組の中で、個人の分類層の $c$ 行とアンカーを、`weights: uniform`ならclientの等重みで平均し、組の全clientへ戻す。単独のclientは自分の行とアンカーのまま（`global_protos`は`train_with_protos`の最後で自分のアンカーになっている、157行）。
- 予測（評価）: clientの表現部＋個人の分類層（`FedCCFA_paper/sections/experiments.tex:18`）。

## 手元への接続（実装担当が既存の構造を調べて具体化する）
- 構成: `sweep/experiment_configurations.py`に4構成の定数を足し、`EXPERIMENT_CONFIGURATION_NAMES`の末尾へ加える。FedEMと同じく、割り当ての方式の設定の項目を新しい値（例: `fedccfa_classifier_clustering`）にし、窓の項目（`current_timestep`／新しい値`recent_observed_samples`と件数200）と整合の開始の項目（`fixed_round_20`／`scaled_to_run_rounds`）を構成ごとに決める。$\epsilon$・$\gamma$・温度は`experiment_spec/parameters.py`へ、$\epsilon$ の掃引の軸は`experiment_spec/sweep.py`の`SweepAxis`へ登録する（AGENTS.mdの規約）。
- モデル: 既存のFedDrift系の構成と同じ分類器の構造（独立のモデルの構造）を使い、表現部＝`classification_layer`より前の全パラメータ、分類層＝`classification_layer`。グローバルなモデルは1つ（ID 0）で、全clientがその表現部を共有する。個人の分類層はclientごとに持つ（サーバのグローバルなモデルの分類層は $\phi^{(0)}$ のまま更新しない。公式と同じ）。2クラスのデータでは、FedCCFAの構成だけ分類層を2出力・softmaxにし、要求1.2の変換で事前学習の初期モデルから作る。
- 時刻の中の順序と予測: 時刻 $t$ の各標本は、(1) ラベルを見る前に、今のclientのモデル（直前の配布の表現部＋直前のroundの後の個人の分類層）で予測し、(2) ラベルを見て窓の標本へ加える。$I$ 件がそろったら、round $r=0,\dots,R-1$ のそれぞれで、client側の(i)〜(v)とサーバ側の(vi)(vii)（要求1.4）を行い、表現部・個人の分類層の行・アンカーを配る。最初の時刻の予測は事前学習の初期モデル（2クラスは変換後）。集約の間は学習しない。
- 窓: `-window`は時刻の初めに空にする。`-window200`は観測済みの直近 $\max(I,200)$ 標本（古い順に捨てる）。境界の例（$I=50$）: 時刻0の標本0〜49は事前学習の初期モデルで予測し、各標本のラベルはその予測の後に窓へ入る。最初の集約（時刻0の終わり）の窓は、`-window`も`-window200`も標本0〜49（観測済みが50件しか無い）。2回目の集約の窓は、`-window`は標本50〜99、`-window200`は標本0〜99。5回目以降の`-window200`は直近200標本（例: 時刻4の終わりは標本50〜249）。ラベルの分布・均衡の部分集合・学習・アンカー・表現部の集約の重み（窓の標本数）は、どれもこの窓で求める。
- 学習の予算: 表現部の学習は、`train_feddrift_held_models_for_round`と同じ件数の条件・更新の回数（`updates_per_sample`×`detection_batch_sample_count`）・ミニバッチの作り方（窓から32件を非復元に抜く）。個人の分類層はその $1/5$（切り上げ）回、同じミニバッチの作り方。診断用の分類層は均衡の部分集合の全部を1ミニバッチにして5回。optimizerは共通のAdam（AMSGrad、weight decay 0.001）を段ごと・roundごとに作り直す（D05と同じ）。表現部の学習率は共通の値、2つの分類層の学習率はその10倍。
- 欠けたクラス（移植のための拡張。公式の実装との照合の対象外）: (a) 均衡の部分集合は、窓に1件以上あるクラスだけから、それらの最少の件数と5の小さい方の件数を抜く。(b) 整合の重みのエントロピーは窓にあるクラスだけで求める（$0\log0=0$）。(c) アンカーは窓にあるクラスだけ求め、窓に無いクラスはそのclientの直前のアンカーを使う。直前のアンカーも無いクラスは、対照の整合のロジットから除く（そのクラスのラベルの標本は窓に無いので、教師ありの損失には影響しない）。(d) クラス $c$ のクラスタリングとその行・アンカーの平均には、窓に $c$ が1件以上あるclientだけを入れ、ほかのclientは自分の行とアンカーのまま。(e) 表現部の集約は全clientで行う。公式は「全クラス各5件以上」を前提にしており（`experiments.tex:12`）、(a)は公式の`min(最少クラスの件数,5)`（`FedCCFA.py:192-193`）を延ばしたもの。
- 整合の開始: runの最初のroundを $r=0$ とする通算のround番号 $r$（時刻 $t$ のround $k$ は $r=tR+k$）で判定し、さらに $r=0$（アンカーが無い）では行わない。`-window`・`-window200`は $r\ge 20$、`-scaled-alignment`は $r\ge\lceil 0.1\,(N/I)\,R\rceil$（$N$ はclientごとの標本数）。保つのは「runの学習roundのうち、整合を始めるまでの割合」（論文: 200 roundのうち20 round＝10%）。$R$ が一定なので、これは経過した標本数の割合10%とも同じで、時刻の途中のroundから始まってよい（時刻の中のroundも通算で数える）。$N=5000$ の例: $R=1$・$I=50$ なら全100 roundで $r\ge10$、$R=1$・$I=500$ なら全10 roundで $r\ge1$、$R=5$・$I=500$ なら全50 roundで $r\ge5$、$R=1$・$I=500$ の`-window`（20）は一度も整合しない。
- 通信（1 roundごと。時刻の境目で特別な送受信はしない。各値はパラメータの個数とbyteを既存の通信の記録で数える）:
  | 部品 | 向き | 回数 | 備考 |
  |---|---|---|---|
  | 表現部 | 上り（全client→サーバ）・下り（サーバ→全client） | 毎round 1回ずつ | 公式の`aggregate_rep`・`send_rep_params` |
  | 初期の分類層 $\phi^{(0)}$ | 下り | runの最初の配布で1回だけ | 公式は毎round全パラメータを送るが、$\phi^{(0)}$ は変わらないので再送を数えない。最初の配布の表現部と合わせて1モデル分 |
  | 診断用の分類層 | 上り | 毎round 1回 | クラスタリングの入力 |
  | 個人の分類層 | 上り | 毎round 1回（分類層の全体） | 組の平均の入力 |
  | 個人の分類層の平均した行 | 下り | 毎round、2 client以上の組に入ったclientへ、そのクラスの行だけ | 単独のclientには送らない |
  | アンカー | 上り（窓にあるクラス）・下り（組の平均。2 client以上の組に入ったclientへ、そのクラスだけ） | 毎round | 単独のclientには送らない |
  $R>1$ では各roundで同じ数え方を繰り返す。
- 計算量: 表現部・分類層の学習とアンカーの順伝播、診断用の分類層の学習を、既存の計算量の記録に計上する。

## 公式のFedCCFAとの違い（文書と論文に書く）
- 準じる点: roundの手順、MADDとDBSCAN（`min_samples=1`）、等重みのクラスごとの平均、整合の重み $H/\gamma$ と対照の整合（温度0.1）、学習の段の比率（分類層:表現部=1:5、診断5回）、分類層の学習率の比（10倍）、予測に個人のモデルを使うこと。
- 全手法に共通の条件から来る違い: (a) 逐次の評価（ラベルを見る前に予測し、変化の直後も評価する）。公式は200 round後の全テストデータでの精度。(b) 局所学習の予算・optimizer（Adam、D04・D05・D11）。公式はSGDとエポック数。(c) 事前学習の初期モデルと、共通のネットワーク（公式はCNN）。
- ストリームへ移すための違い: (d) 学習のデータは観測済みの窓（$I$ 件か直近 $\max(I,200)$ 件）。公式は固定のデータ。(e) 1時刻のround数 $R$（既定1）。(f) 整合の開始（`-scaled-alignment`は全roundの10%）。(g) 欠けたクラスの扱い（上の(a)〜(e)）。(h) 2クラスのデータの2出力の分類層。(i) 全client参加（公式は全参加と一部参加の両方を扱う）。(j) 公式のコードのアンカーの平均の参照共有の副作用は再現しない（2026-11-05、主担当の判断。下の「実装で決めたこと」）。

## Validation
- 新しいtest`tests/refactoring/test_feddrift_fedccfa_configuration.py`。
- 公式の式との照合（要求3.1）: 公式の`madd`・`merge_classifiers`の組の作り方・`aggregate_label_params`・`aggregate_label_protos`・`aggregate_rep`・`update_label_distribution`の重み・`train_with_protos`の対照の整合の損失・`class_balance_sample`の件数の規則を、MIT Licenseの表記つきの抜粋として`tests/refactoring/fedccfa_official_excerpt.py`に置き（関数全体の写しではなく式の部分。各関数に公式のファイル・行を書く）、固定の入力で手元の値・組と一致すること（許容誤差1e-6、組は集合として一致）を確かめる。round内の順序を、呼び出しの記録で確かめる。
- 公式の実装との照合（要求3.1の前半）: 全クラスが窓にある条件（公式の前提の範囲）で行う。一度きりの検証script（`.validation-fedccfa/official_round_equivalence.py`、git管理外。公式のclone `C:\Users\yshin\Research\FedSDA-experiment\FedCCFA`をimportする）で行い、結果をtasks.mdのImplementation Notesに記録する（pytestは公式のcloneに依存させない）。条件: 3 client、10クラス、入力の次元8、隠れ層16の小さい全結合（公式の`clf_keys`＝最後の2キーと手元の`classification_layer`が同じ切り分けになるもの）、client 2だけ数字1と2のラベルを入れ替え。学習データは2通り: (A) 各クラスちょうど5件（均衡の部分集合が全データになる）、(B) クラスごとの件数が違う（例: client $i$ のクラス $c$ は $2+((c+i)\bmod 5)$ 件。最少2件なので均衡の部分集合は各クラス2件）。(B)では、均衡の部分集合として抜く標本の添字を、公式（`np.random.choice`の差し替え）と手元の両方へ同じ値で差し込み、抜いた件数とクラスの構成も照合する。全段を1ミニバッチ（batch_sizeをデータ件数以上）にして並びの乱数を除く。手元の移植は、照合用にだけ、段ごとの更新回数（`rep_epochs`・`clf_epochs`・`balanced_epochs`に当たる回数）とoptimizer（SGD、momentum 0.9、weight decay 1e-5、表現部0.01、分類層0.1）を公式に合わせる入口を持つ（評価のrunの構成からは使えない）。初期の重みを両方で同じにし、round 0〜2を、整合の開始を0にした場合と20にした場合の両方で動かす。比べるもの: 各clientの均衡の部分集合（(B)）、診断用の分類層、個人の分類層、表現部（集約前と集約後）、アンカー、クラスごとの組（集合として一致）、組の平均の後の分類層の行とアンカー、単独のclient（例: クラス1・2のclient 2）に戻る行とアンカーが自分の値のままであること。許容差は絶対1e-5。
- 元の条件での挙動の確認（要求3.2、公式の論文の設定に近づけたもの）: 数値の正しさは上の公式の実装との照合（許容差つき）で確かめ、この確認は、移植を流れの中で動かしたときに公式の論文が主張する機能（入れ替わったクラスだけが分かれる）が現れることを、判定の境界を固定して確かめる役割に限る（クラスタの細かい形はデータと乱数に依存するので、期待値を一意に固定しない）。結果の組（時刻9〜11の各クラスの組）はImplementation Notesに記録する。 MNIST4、10 client、間隔50、$R=3$、`-window200`、$\epsilon=0.1$、整合はscaled、seed 0。テストの中だけ、コンセプトの予定を次に差し替える（runの`generate_random_client_concept_traces`の段の差し込み）: client 0〜4（群A）はコンセプト0のまま、client 5〜9（群B）は時刻4の初め（標本200）に一斉にコンセプト1（数字1と2の入れ替え）へ移り、そのまま。各client 600標本（12時刻）。合格条件: (1) 時刻9〜11の最後のroundで、クラス1と2のそれぞれについて、2 client以上の組のどれも群AとBの両方を含まず、群Aだけの組と群Bだけの組がそれぞれ1つ以上ある（窓にそのクラスが無いclientは除く）。(2) 同じroundで、クラス0・3〜9の8クラスのうち5クラス以上で、群AとBの両方を含む組がある。(3) 群Bの、ラベルが1か2の標本の予測の精度が、時刻9〜11で、時刻4〜5より30ポイント以上高い。(4) 移る前（時刻1〜3の最後のround）に、クラス1・2で群AとBの両方を含む組がある。所要時間を記録し、重ければ`slow`にする。満たさないときは条件を緩めず、主担当へ報告する。
- 小さい全体run（要求3.3）: 4構成のうち`-window`と`-window200-scaled-alignment`で、SINE2とMNIST4、間隔50、seed 0。規模はFedEMの小さい全体runと同じく、検査が全部働く最小の規模（client数・標本数は実装担当が決め、理由を書く）。(a) 窓の標本番号。(b) round内の順序と、予測がそのラベルを学習へ加える前に行われていること。(c) 表現部・個人の分類層・診断用の分類層の更新の回数が要求1.6のとおり。(d) 欠けたクラスの規則（窓に無いクラスのclientがそのクラスの組に入らない、均衡の部分集合が窓にあるクラスだけから最少件数ずつ抜かれる、エントロピーが窓にあるクラスだけ、アンカーの保持と、直前も無いクラスの対照のロジットからの除外）を、欠けたクラスを含む固定の窓の単体のtestでも確かめる。(i) 最初と2回目の集約の窓の標本番号が、上の「窓」の境界の例のとおり（`-window`と`-window200`の両方）。(e) SINE2の2出力の分類層の初期の予測確率が1出力のsigmoidと一致（許容誤差1e-6）。(f) 通信の計数が部品ごとに上の規則のとおり。(g) 整合の開始のround（上の式と例。$R=1$ と $R=3$）。(h) 記録の行数と列。$R=3$ でも(b)(c)を確かめる。
- 既定の不変: 既存の構成の保存設定・識別子が変わらない。構成・掃引・識別子・依存境界・スキーマのtest、FedEM・IFCA・Window・CFL・DriftSurf・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`、Ruff・Pyright。全pytestは主担当が流す（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）。

## 記録
- 表`fedccfa_label_cluster`（列: `timestep_index`、`round_index`、`class_label`、`client_id`、`cluster_index`（単独・窓に無いクラスは $-1$）、`window_class_sample_count`）。
- 表`fedccfa_round`（列: `timestep_index`、`round_index`、`client_id`、`alignment_active`、`alignment_weight`、`window_sample_count`、`window_missing_class_count`）。
- 名前と列は既存の記録の規約に合わせて実装担当が決めてよく、naming.mdへ追記する。保存形式の版を上げる。

## 予備の実行（要求4.2）
- 構成と $R$: 4構成×$R\in\{1,5\}$、$\epsilon=0.1$、$\gamma=20$。SINE2・CIRCLE2・MNIST4・ROTMNIST4×間隔50・500×seed 100〜102（192 run、uranium）。各clientのコンセプトの予定は論文の5.1節の設定（300標本の後から標本ごとに確率0.0015）。
- 集計: 構成×$R$×データ×間隔ごとの累積の精度・通信・局所の計算量、変化の直後の $I$ 標本とそれ以外の精度（`docs/research/feddrift-post-change-accuracy-20261101.md`の定義）、クラスごとの組と真のコンセプトの一致（変わるクラスについて、同じ組の対のうち同じコンセプトの割合と、同じコンセプトの対のうち同じ組の割合）、欠けたクラスの割合、整合を行ったroundの割合。参考として、FedEM（予備の実行）・FedDrift・FedAvg+Windowの値を並べる。
- 選ぶ手順（集計の単位は構成×$R$。指標は累積の精度（%）で、4データ×2間隔×3 seedの24 runの平均を1つの値にする。差は最大値からの差（最大値−各値）で、0.1ポイント以内は差が0.1以下。最大の構成は必ず候補に入るので、候補が空になることはない。各段で絞った後に1つも残らない段は飛ばす。最後に2つ以上残れば、4構成の並び（`-window`、`-window-scaled-alignment`、`-window200`、`-window200-scaled-alignment`）の先のものを選ぶ）: (1) 構成×$R$ ごとの、4データ×2間隔×3 seedの累積の精度の平均の最大値を求める。(2) 最大値との差が0.1ポイント以内のものを候補にする。(3) 候補に $R=1$ があれば $R=1$ だけに絞る（通信が $R$ 倍のため。表1のほかの手法と通信をそろえる）。(4) 残りに`-window`（窓が短い）があればそれに絞る。(5) 残りに整合の開始が公式どおり（20）のものがあればそれを選ぶ。$R=5$ が $R=1$ より0.1ポイントを超えて高いときは、その差と通信の差をユーザーに示し、$R$ はユーザーが決める。

## $\epsilon$ の調整の実行
- seedの分け方: 予備の実行と $\epsilon$ の調整は評価に使わないseed 100〜102だけで行い、論文の表に載せる値は本規模の実行（seed 0〜9）だけから出す（FedDriftのδ・DriftSurfの閾値・Adaptive-FedAvgの学習率と同じ）。
- 予備の実行で選んだ構成と $R$ で、$\epsilon\in\{0.05,0.10,0.15,0.20\}$（論文の探索の範囲）×10データ×間隔50・100・200・500×seed 100〜102（480 run）。
- データ×間隔の40個の組のそれぞれで独立に、その組の3 seedの累積の精度の平均を $\epsilon$ ごとに求め、最良の値から0.1ポイント以内の $\epsilon$ のうち、論文の採用値0.10に最も近いもの（同じ距離なら小さい方）を選ぶ（FedDriftのδ・DriftSurfの閾値と同じく組ごとに選ぶ）。選んだ値の一覧をCSVに残す。$\gamma$ は論文の採用値20に固定する（調整の範囲と計算の予算のため。差として書く）。

## 本規模の実行
- 選んだ構成・$R$、データ×間隔ごとに選んだ $\epsilon$ で、10データ×間隔50・100・200・500×seed 0〜9（400 run）。結果は`paper_results.py`へFedEM・DriftSurfと同じ形で足す（論文のリポジトリ）。

## 実装で決めたこと（2026-11-05、主担当の判断）
- 公式の`aggregate_protos`（`entities/FedCCFA.py:300-309`）は、最初のclientの`local_protos`のTensorへ他のclientのアンカーをin-placeで足し込み、`aggregate_label_protos`（453-462行）も組の先頭のclientのTensorをin-placeで変える。`train_with_protos`の最後（157行）で`global_protos`が同じTensorを参照するので、先頭のclientの整合の目標と、組の平均の入力が変わる（実装担当の直接の確認では、アンカー[1,2,3]の組の平均が期待2に対して3.6666667）。論文（`methodology.tex`）の記述は組の中の等重みの平均なので、手元は副作用の無い平均にする。公式の実装との照合は、この参照共有だけを写しで隔離した対照で、(A)(B)×整合の開始0・20の4条件すべてで最大の差9.54e-7、組・均衡の部分集合・単独のclientの値が一致した（主担当も再実行して確認）。隔離しない照合は4条件とも不一致で、その差はこの副作用から来る。
- 仕様に無かった境界: (1) あるクラスを窓に持つclientが2以下のときは、MADDの比較相手が無いので距離0（公式は $n-2=0$ で割る）。(2) ゼロのベクトルのcosの類似度は0。(3) 初めて現れたクラスでアンカーがまだ無い標本は、アンカーが得られるまで対照の損失から除く。(4) 整合の開始の後でも、$H=0$ か、窓のラベルに対応するアンカーが1つも無いroundは整合しない。

- 1回目（GPT-6 Luna medium）FAIL（Major 2: γの既定値と`lambda`の分岐、公式の実装との照合と元の条件での挙動の確認の合格条件。Minor 2: 公式の行の参照、選択と評価のseed）。γの既定値はもともと20だったが、正の値だけを受け付けることと`lambda`の分岐を持たないことを明記した。公式のコードと移植を共通の固定の入力で1 roundずつ照合する検証scriptと、挙動の確認の対象の時刻・区間・最低の改善幅を定めた。行の参照を関数の定義に合わせ、選択はseed 100〜102、評価はseed 0〜9であることを明記した。
- 2回目FAIL（Major 4: 公式の実装との照合で均衡の部分集合が働かない入力だけだった、欠けたクラスの扱いが公式の前提と両立する形で定義されていない、整合の開始の境界、最初の集約の窓とラベルの時点の確かめ方。Minor 3: 1回目の対応の実効性、既存の構成の不変の確かめ方、εの選び方の単位）。不均衡な入力と添字の差し込み・単独のclientの照合を足し、欠けたクラスを照合の対象外の拡張と明記して単体のtestを足し、整合の開始を $r$ の式と例で、窓を境界の例で定め、既存の構成の不変とεの選び方の単位を書いた。
- 3回目FAIL（Major 3: 元の条件での挙動の確認の合格条件の水準、整合の開始の式が何を保つか、通信の数え方。Minor 3: 行の参照、欠けたクラスの拡張の明示、選ぶ手順の再現性。1・2回目の指摘は解消と判定）。挙動の確認と数値の照合の役割を分けて書き、整合の開始が保つ量（学習roundのうち整合までの割合）を明記し、通信を部品・向き・回数の表にし、行の参照と選ぶ手順の細部を書いた。欠けたクラスは2回目の対応で拡張・照合の対象外と明記済み。
- 4回目PASS（Minor 1: 公式の行の参照で、関数の定義の行と処理の行を区別する）。定義の行と処理の範囲を併記した。Minorの文面の修正なので再レビューは省く（2026-10-27の決定）。
