# FedDrift元論文・公式コードとの対応

この文書は、新実装のFedDriftの比較条件の正本である。基準版14では、検知・選択・新モデルの複写元・検知の基準値・学習対象・統合重みを、下記の範囲で論文と公式コードへそろえる。公式コードの直接利用や、論文の実験全体の再現を意味しない。

参照は元checkoutの`FedDrift_paper/`のTeXと、`FedDrift/`の公式clone（commit `e4ac32fe17edf8de23579678a12fb4d7b5f45746`）。過去の差のIDは[2026-10-21の実装監査](../../results/implementation-audits/feddrift-20261021/feddrift-implementation-audit.md)に対応する。監査は変更前の記録として保持する。実装の仕様は[要求](../../.kiro/specs/feddrift-official-algorithm-alignment/requirements.md)・[設計](../../.kiro/specs/feddrift-official-algorithm-alignment/design.md)、版と検証状況は[基準の文書](refactoring-baseline.md)・[実装記録](../../.kiro/specs/feddrift-official-algorithm-alignment/tasks.md)を参照する。

## そろえた範囲

| 差 | 基準版14の既定 | 論文・公式コードの根拠 |
|---|---|---|
| D01 検知・モデル選択 | 分類の誤り率を最小化。同値は保有順で最初。最小値が保存した基準値＋δを超えたとき検知 | `algo-creation.tex`の式2・Algorithm 2・accuracyの説明。公式`FedAvgEnsDataLoader.py`の`cluster_hierarchical`・`train_acc_matrix` |
| D01 統合の距離 | `max(E_ij − E_ii, E_ji − E_jj, 0)`。complete linkage、停止閾値はδ | 論文Algorithm 2、公式の距離A・linkage F |
| D02 新モデルの初期値 | その塊で誤り率が最小の既存モデルからパラメータを複写 | 公式`find_unused_model_lru(..., best_model)`。論文はInitializeとのみ記載 |
| D03 次の検知の基準値 | 検知の有無によらず、その塊の既存モデルの最小の誤り率 | 論文の前時刻の最小値、公式の`set_acc(c, newest_acc)`。学習後の再評価は行わない |
| D04 学習・集約の対象 | 今の時刻にどれかのclientへ割り当てられたモデルだけ。そのモデルの過去の学習標本を持つ全clientが、件数条件を満たせば学習 | `appendix-expt-setup.tex`の「現在クラスタされているモデルだけを学習」。公式Trainer・Aggregatorの現在の割当による除外 |
| D07 評価clientの上限 | 実験のclient数。今の塊を含む、対象モデルに割り当てた標本で評価 | 論文Algorithm 2の全client・全時刻のデータからの標本、公式の`cluster_data` |
| D08 統合の重み | 今の塊を含む、全clientのモデル別の学習標本数。件数0なら保存済み重みを使用 | 論文Mergeの`Σ w N`、公式`merge`の現在の時刻までの割当。塊の件数が同じなら比例する |
| D10 隔離 | W=1。作成した時刻は統合せず、次の時刻から統合・他clientの選択の候補 | 公式`update_marking`の時刻τ＋Wでの解除に対応。W>1の公式の固定・選択制限は実装しない |

2値1出力の分類器は、既存の予測関数と同じく確率が0.5を超えたとき1、それ以外は0とする。多クラスはargmax（同点は最小クラス番号）。確信度だけが変わり予測クラスが変わらなければ、誤り率は変わらない。

δのデータ別の既定はSEA2・SEA4が0.04、SINE2が0.20、CIRCLE2・MNIST2・MNIST4が0.10（公式`default_deltas`）。掃引の`--feddrift-loss-increase-thresholds`による明示値を優先し、次にbuilder引数、そのどちらもなければデータ別の既定を使う。検知と統合に同じ値を渡す。論文付録の0.02〜0.20の掃引は別の実験として行う。

統合前に標本数を数え、統合時のIDの付替えで標本を連結する。その後の集約は、その件数で重みを置き換えるので、統合の重みをさらに加算する二重計数はしない。未使用モデルは保有を続け、次の時刻の選択に備える。

## 開示する差・共通の実験条件

| 差 | 本実装の条件と限界 |
|---|---|
| D05 optimizer | 配布でoptimizerを作り直す。公式も時刻ごとにプロセスを起動し直し、R=1では実質同じだが、複数roundで状態を保持する公式の学習とは異なる。未使用モデルは再構築せず保持する |
| D06 初期化 | 全手法共通の単一モデルの事前学習（既定500件・10 epoch）。論文の合成データの「1概念・1モデルから始める」に沿う。公式READMEのclient別初期化Fとは異なる。付録のF型はFMoWで使われる |
| D07 評価の場所・標本・不足対 | client側で評価し、各clientの標本上限は50。評価の保存標本が6件未満なら、現在の割当の学習標本が11件以上のときだけ代用し、評価実施には5件を要する。距離に必要な4評価が最低件数に足りない対は統合しない。公式の中央で全データから最大20バッチを取る方式と同じ標本ではない |
| D09 モデル容量 | 公式の容量上限・LRUによる枠の再利用を実装しない。モデルを再来に備えて保持する |
| D11 ネットワーク・更新予算 | 全手法で共通の構造・オンラインの更新予算を使う。集約間隔の塊で検知し、既定のlocal roundは1。論文の固定500件・R=100の条件とは異なる。roundだけを増やして論文値を再現できるとは扱わない |
| D12 評価 | 学習前の予測で評価するprequential方式。公式の次の時刻のテストとは評価時点が異なる |

クロス評価のclient上限の撤廃で増える計算・通信も計数する。D04の対象集合は、既存の小さな通知と同じくメッセージ件数へ入れ、パラメータのbyteへは入れない。対象外モデルのround学習・集約・パラメータ通信は数えない。比較結果には精度だけでなく、上り・下り通信量、計算量の計数/MAC、経過時間を並べる。今回の実装変更による精度や費用の改善・悪化は、対照実験前には結論を出さない。

## 真のコンセプトで割り当てるOracle構成

`feddrift-oracle-concept-assignment`は、`FeddriftRunClientSettings.model_assignment_policy="true_concept_assignment"`で選ぶ。既定の`detection_batch_assignment`は通常のFedDriftで、既存構成の設定・識別子・goldenは維持する。

最初にコンセプト数Kのモデルを、共通の事前学習モデルの写しから登録する。ID kをコンセプトkに固定し、検出バッチ内も標本ごとに真のコンセプトへ振り分ける。検出、新モデルの作成、統合は行わない。その時刻にどこかのclientで1件以上帰属したモデルを学習対象とし、対象モデルの過去の標本を持つclientが、モデル別の件数条件（既定32件以上）を満たせば参加する。R回のround、更新予算、モデル別の累積標本数による集約の重み、通信の数え方はFedDriftと同じ。割当IDの集合の報告はclientごとに1通と数える。標本ごとの学習先は`sample_prediction.training_model_id`へ保存する（FedDriftの保存形式の版3で追加、Oracle構成だけの列）。

論文のOracleは真のクラスタリングを使って学習し、次の時刻の標本で評価する（変化した時刻は平均から除く）。本実装は逐次の評価なので、現在の標本の概念はラベルと一緒に明かされ、予測には直前にラベルを見た標本の概念を使う。最初は同じ初期値のモデルのうちID 0で予測する。変化の直後の1標本は古い概念のモデルで予測し、その標本も精度に含める。FedHASPの標本ごとの局所学習とは学習時期も異なり、この構成はFedDriftの割当が正しい場合の比較手法として扱う。

## IFCA+Windowの2構成

`feddrift-ifca-window`は、真のコンセプト数Kを与え、K個のモデルをrunのseedとIDから決めた独立の乱数状態で初期化する。`feddrift-ifca-window-pretrained-cluster0`はID 0だけを共通の事前学習モデルの写しにし、ID 1以降は前者と同じ初期値にする。初期化でrunの乱数状態を変えない。共通の事前学習はデータ生成・乱数系列を揃えるためどちらも実行するが、全乱数版のクラスタにはそのパラメータを使わない。

最初の時刻は、FedDrift論文付録のIFCAの初期化に合わせ、比較せず全clientをID 0に割り当てる（付録のFMoWだけの例外は使わない）。2時刻目からclientごとに、その時刻のバッチでの平均有界損失が最小のモデルを選ぶ。同値は小さいIDを選ぶ。通常のFedDriftの比較基準（誤り率）とは異なり、IFCAの選択は損失で行う。選択したモデルで次の時刻を予測する。検出・モデルの追加・クラスタリング・統合は行わず、Kモデルを保持する。

学習前に全モデルの学習標本を空にし、選択モデルへその時刻のバッチだけを追加する（窓1時刻）。全clientの選択IDの和集合が学習対象になり、各clientでは選んだモデルだけが件数条件を満たせば学習に参加する。FedDriftのround数・更新予算・学習率・集約の重み・通信の数え方を使う（D04・D05・D08・D11）。集約の重みはclientごと・モデルごとの標本数で、この構成ではその時刻の件数となる。割当IDの報告はclientごとに1通。論文のIFCA-Wはroundごとに選び直すが、本構成は既定の1時刻1roundに固定するので時刻ごとの選択と一致する。

論文のIFCA+Windowに準じるのは、全乱数初期化、最小損失でのクラスタ選択、窓1時刻、および最初の割当。移植による違いは、逐次評価（ラベルを見る前に予測し、変化した時刻も含む。論文は時刻の学習後に次時刻で評価し変化した時刻を除く）、上記のFedDriftの学習・集約の規則、真のコンセプト数Kを与える有利な条件である。ID 0だけ事前学習する派生版は事前学習の利点がID 0へ偏る。論文の表には全乱数版をIFCA+Windowとして載せ、これらの違いを本文か脚注で開示する。事前学習版は初期化の影響を示す参考または付録の候補とし、掲載場所はユーザーが決める。

`model_assignment_policy="minimum_loss_cluster_assignment"`、`training_sample_window="current_timestep"`、`cluster_model_initialization="all_random"`または`"pretrained_cluster0"`で指定する。既定は順に`detection_batch_assignment`、`all_timesteps`、`all_pretrained`。既定値は保存設定から省き、既存のrun識別子を維持する。新API専用で、旧CLIの動作は変更しない。

選択は`cluster_selection`表にclient ID、時刻（0始まり）、先頭・末尾の標本番号、選択ID、モデルIDの列と平均有界損失の列（後二者は行×K）を保存する。最初の時刻は比較しないため損失をNaNで表す。標本範囲は両端を含み、モデルIDは昇順。既存の`detection_batch`表は空になる。

## データと基準版

新実装の対象はSINE2・SEA2/4・CIRCLE2・MNIST2/4。SEA4は論文付録の閾値（9、8、7、9.5）と10%のラベルノイズを用いる。公式配布CSVの概念4との差は、論文の定義を選んだ条件として残る。FMoWは対象外。既定の概念系列は、最低間隔を満たした後に確率的に変化する系列で、論文の固定staggered系列とは異なる。

基準版1〜13は、有界損失・判定前の現在モデルからの複写・複写したモデルの値を基準値にする扱い・学習標本を持つ全保有モデルの学習・前の集約時点の統合重みを明示する。評価clientの上限は3、通常の閾値は0.1（経路確認の`feddrift-drift-tail`だけ0.03という既存条件を維持）。旧goldenは変更しない。新旧の項目を混ぜたrun設定と、新方式のW≠1は拒否する。

基準版14のgoldenの生成・配置、FedSDA・FedAvgが版13と同じ値になることの最終確認は主担当が行う。生成前のコードと、生成・確認済みのgoldenは区別する。

## WindowとAdaptive-FedAvgの2構成

`feddrift-window`と`feddrift-adaptive-fedavg-window`は、全clientで共通の事前学習モデルの写し1個（ID 0）を使い、
全標本をそのモデルへ帰属させる。真の概念IDは帰属には使わない。検出・モデルの作成・クラスタリング・統合はしない。
学習前に標本の集合を空にし、その時刻のバッチだけで学習する（IFCAと同じ窓1時刻）。
標本数で重みづけた平均で集約する。学習対象・件数の条件・バッチ32・更新予算・1時刻1round・
配布時のoptimizerの作り直し・割当IDの報告1通/clientはFedDriftの規則（D04・D05・D08・D11）へ接続する。

WindowはFedDriftと同じAdam（AMSGrad、weight decay 0.001）とデータ別の学習率を使う。
合成データでは0.01、MNIST/Fashion-MNISTとそれらの回転データでは0.001。
Adaptive-FedAvgは局所学習だけをSGD（momentum 0、weight decay 0）にする。事前学習の初期値と乱数系列は共通。
サーバは各時刻の集約後、state_dictの順に全値（bufferを含む）を平らに連結したfloat64ベクトルから、
公式`FedAvgEnsDataLoader.AdaState`の式で次の時刻の学習率を決める。初期学習率は0.01、
3つの移動平均の係数は全て0.5。式の集約回数は1始まりで、時刻番号（0始まり）に1を足した値。
非有限の推定値は例外で止める。学習率はモデルと一緒に送るスカラー1個で、通信量へ追加しない。

公式の`win-1_iter`は、時刻ごとに複数roundあるうち`round_idx == comm_round - 5`の集約後の値で、
その時刻のiteration番号を渡して1回だけ学習率を更新する。手元も更新頻度と回数の数え方は同じだが、
1時刻1roundの唯一の集約後の値に置き換える。最後から5つ目を選ぶ意図は公式コードと論文には記載されていない。
2構成とも単一モデルと窓1時刻は論文に準じるが、逐次評価（D12）、共通の事前学習モデル（D06）、
局所学習のバッチ・更新予算・round数（D04・D05・D11）が異なる。
Adaptive-FedAvgの初期学習率はデータ×間隔ごとに選ぶ（論文はデータごと）。
候補は論文付録の0.1・0.01・0.001・0.0001に固定し、端が選ばれても広げない。

版5で、Adaptive-FedAvgだけが持つ`adaptive_learning_rate`表を追加した。列は`timestep_index`、
`aggregation_count`、`variance_ratio`（分散の比）、`smoothed_ratio`（補正済みの比の移動平均）、
`learning_rate_for_next_timestep`。Windowと既存構成の表と列は変えない。

## DriftSurfの構成

`feddrift-driftsurf`はサーバに`pred`・`stab`・`reac`の役割を置き、全clientが予測の役割の大域モデルを使う。
`model_assignment_policy="driftsurf_global_role_assignment"`、`training_sample_window="role_timestep_window"`、
`driftsurf_settings=DriftSurfSettings()`で指定する。既存構成の既定は順に`detection_batch_assignment`、`all_timesteps`、`None`のまま。
δ（`accuracy_drop_threshold`）の既定は0.10、反応の長さは3時刻、窓は10時刻に固定する。
全clientは各時刻の予測後・学習前に、公式が評価する役割だけをeval/no_gradで推論し、最大出力の添字が観測ラベルと一致する数と件数を報告する。
手元の二値モデルの1列の確率pは、クラスごとのスコア(1-p,p)へ対応付ける。同値は小さい添字0とする。
サーバで整数の和を割るため、同じクラス別出力を使う公式の全clientのデータを集めた正解率と同じになる。
報告はclientごとに1通で、parameterのバイトには数えない。

最初の時刻は遷移せず`pred`・`stab`を学習する。以後は公式`DriftSurfState.run_ds_algo`と同じδ・δ/2・最高正解率・反応の比較・昇格の規則を使う。
各clientは直近10時刻の標本を時刻別に保存する。検出バッチから既存の学習storeへ追記せず、学習直前に全モデルの標本を空にし、役割の窓に含む時刻だけを昇順に連結する。
集約の重みはこの窓の標本数であり、累積の学習件数や保存済みの重みではない。更新予算は窓の長さによらずFedDriftの規則（D04・D05・D11）を使う。
昇格した時刻の公式の学習役割`pred,reac`は、同じ昇格モデルの識別子へ対応させ、一度だけ学習・集約する。
FedDriftのclient別の検出・モデル作成・クラスタリング・統合は行わない。役割の作り直しで新IDを登録し、古いIDを除くため大域モデルは常に3個以下。

公式からの変更は、反応終了時に昇格しなくても予測の役割を必ず`pred`へ戻す点である。
公式では`reac`が残ると次時刻の学習役割`pred,stab`からテストモデルを決められない。
また公式は固定seedから連続して組み立てる最初の2モデルの初期値が異なるが、手元は両方と作り直す役割を共通の事前学習モデルθ0の写しにする（D06、optimizerは新規）。
そのほかの違いは逐次評価（D12）、局所学習の規則（D04・D05・D11）、1時刻が集約間隔の標本数である点（窓10時刻は間隔50で500標本、間隔500で5000標本）、δをデータ×間隔ごとに選ぶ点である。

保存形式は版6。DriftSurfだけ`driftsurf`表を追加し、既存構成の表・列は変えない。
列は`timestep_index`、`state_before`・`state_after`、`accuracy_pred`・`accuracy_stab`・`accuracy_reac`（未計測はNaN）、
`accuracy_best`、`training_roles`（カンマ区切り）、`prediction_role`、`reactive_counter`（安定状態は−1）、`entered_reactive`、`promoted`、
役割ごとの`*_model_id`（未登録は−1）・`*_window_timesteps`（カンマ区切り、空窓は空文字）。
IDと窓は遷移後・学習時の対応で、`prediction_role`は次時刻の予測に使う役割を表す。

## CFL+Windowの構成

`feddrift-cfl-window`は`model_assignment_policy="cfl_bipartition_assignment"`、
`training_sample_window="current_timestep"`、`cfl_split_gamma=0.1`を使う。γの掃引軸は持たない。
最初は共通の事前学習モデルID 0に全clientを割り当て、分割したときだけ割り当てを変える。
各時刻の始めに割り当てを写してから全モデルの学習標本を空にし、時刻末に今の標本を割り当て先へ追加する。
予測はその時刻の割り当てモデルで行う。モデルの上限は真のコンセプト数K。
FedDriftの検出・検出による作成・モデル統合は行わない。

公式の`cluster_cfl`と同じく、クラスタIDの昇順で平均更新のノルムの最大値を観測し、
`eps1=cfl_norm/10`、`eps2=6*eps1`を更新する。平均が`eps1`未満かつ最大が`eps2`超なら、
更新の余弦類似度の完全連結2分割を調べ、`sqrt((1-alpha_cross)/2)>gamma`と上限を満たせば分割する。
runtimeはNumPyで近隣鎖・併合の安定ソートと、標準のheapで木の切り方を実装し、
開発用scikit-learn 1.9.1の`metric="precomputed", linkage="complete"`とラベルまで照合する。
更新のないクラスタは飛ばす（公式は空列で例外になるが、通常の窓では起きない）。
分割したroundでは全モデルの集約を省き、clientの学習・上り通信は計上する。

公式・論文との違いは、(a) D12の逐次評価、(b) D11の全手法共通の更新予算で1時刻1round
（公式の例は200 round・5エポック）、(c) D06の共通の事前学習初期値θ0、
(d) D04・D05・D11の学習規則である。とくに分割で使う元のモデルと新モデルは両方θ0の写しにする。
公式は元のモデルを乱数で再初期化し、新モデルは未使用枠の構築時の乱数初期値を使う。
割り当てを保持する規則、窓1時刻、分割判定、γ=0.1、上限K、分割roundの集約省略、
割り当てモデルを使う予測は公式に準じる。

保存形式は版7。CFLだけ`cfl_assignment`（時刻・client・最初の標本位置・モデルID）と
`cfl_split_check`（時刻・元モデルID・更新のあるclient ID・平均/最大ノルム・判定前後の
`cfl_norm`/`eps1`/`eps2`・2分割を調べたか・`alpha_cross`・分割したか・新モデルID・両群のclient ID）を持つ。
client IDの列はカンマ区切り文字列、未判定の`alpha_cross`はNaN、新モデルがないときのIDは−1。
既存構成の設定・識別子・表は変えない。

1時刻1roundで分割が機能するかは、本規模の前に主担当が24 runの予備実行で判断する。
手順と本規模400 runの条件は[承認済み設計](../../.kiro/specs/feddrift-cfl-window-configuration/design.md)に従う。
