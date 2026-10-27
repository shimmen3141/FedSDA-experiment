# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAを比較手法と比べて評価する研究者（ユーザー）。
- 今の状況: 比較手法のFedDriftは、公式コードの直接利用ではなく、内部の旧実装を引き継いだ再実装である。2026-10-21の監査
  （`results/implementation-audits/feddrift-20261021/feddrift-implementation-audit.md`、独立レビューgpt-6-luna high）で、
  検知・モデル選択・初期化・学習の対象・統合の重み・隔離に、公式コード（`FedDrift/`、commit `e4ac32fe`）との差（D01〜D12）が確認された。
  `docs/experiments/differences-from-feddrift.md`の「忠実」「論文準拠」の記述は、この監査と合わない。
- 何を変えるか: アルゴリズムの差（D01〜D04、D08、D10）と、クロス評価の対象のclientの上限（D07の一部）を、元論文（`FedDrift_paper/`）と公式コードにそろえる。
  共通のオンラインの予算・評価に合わせた差（D05、D06、D07の残り、D09、D11、D12）は、そろえずに実験条件として開示する。新しい形を既定にし、基準版14を足す。

## Introduction
2026-10-27、ユーザーは、FedDriftを公式の実装にそろえる作業を、IMPROVE-023より先に進めると決めた（「その順で進めてください」）。
元論文はTeXで`FedDrift_paper/`にある（`algo-creation.tex`のAlgorithm 1・2、`algo-training.tex`のAlgorithm 3、`appendix-expt-setup.tex`）。

論文と公式コードで決まること（主担当の照合。根拠の行は設計書）:

| 監査のID | 論文 | 公式コード | 今の実装 | この仕様 |
|---|---|---|---|---|
| D01 | 検知・統合は「モデルの正解率の差」に基づく（`algo-creation.tex` 103行）。δは0.02〜0.20で掃引（付録） | 検出の塊の正解率、距離Aも正解率 | 正解の確率に基づく有界損失 | 誤り率（1−正解率）にそろえる |
| D02 | 新しいモデルの初期値は明記なし（「Initialize」） | 最良の既存モデル（最小の誤り）を複写 | 判定前の今のモデルを複写 | 公式にそろえる |
| D03 | 前の時刻の「全モデルの最小の損失」と比べる（式2） | 判定の時点の最良の既存モデルの正解率を保存 | 複写したモデルの損失を保存 | 公式にそろえる |
| D04 | 「今クラスタされているモデルだけを学習する」（付録39行） | 今の時刻に割り当てがどのclientにも無いモデルは、学習も集約もしない | 学習標本を持つ保有モデルを全部学習 | 論文・公式にそろえる |
| D07（一部） | モデルjのクラスタの全データの標本で評価（Algorithm 2） | 全データからの部分標本 | 評価させるclientを最大3つ | 上限をclientの数にする |
| D08 | 統合の重みは、今の時刻τまでの割当の標本数（Algorithm 2のMerge） | 同じ | 前の集約の時点の重み（今の時刻の割当を含まない） | 論文・公式にそろえる |
| D10 | 検知したclientを個別のクラスタへ隔離（Algorithm 2） | 隔離中のclientはそのモデルに留まり、再選択・検知をしない。隔離中のモデルは他のclientの選択の候補から外す。既定W=1 | 統合の対象から外すだけ | W=1では同じ振る舞い。W=1だけを受け付ける（要求5） |

開示する条件として残すもの（この仕様では変えない）: D05（optimizerの作り直し。公式も時刻ごとにプロセスを起動し直し、R=1では実質同じ）、
D06（全手法で共通の事前学習。論文の合成データの「1つの概念・1つのモデルから始める」（`algo-creation.tex` 8行）と同じ向き。公式のREADMEのFedDriftの指定`H_A_F_1_06_0`は
clientごとの初期モデル（F）だが、論文の付録ではF型の初期化はFMoWだけに使った（`appendix-expt-setup.tex` 37行）ので、論文に従う）、D07の残り（client側の評価、各clientの評価の標本の上限、評価が足りない対の扱い）、
D09（モデルの数の上限とLRU。論文に記述なし）、D11（ネットワークと更新の予算。全手法で共通）、D12（学習前の予測による評価）。

testは、重要なもの・後で効くもの・複雑なものに絞る（基準版12のspecと同じユーザー指示）。

## Requirements

### Requirement 1: 検知と選択の基準を誤り率にする（D01）
**Objective:** 研究者として、FedDriftの検知とモデルの選択を、論文と公式コードと同じ正解率の基準で行いたい。

#### Acceptance Criteria
1.1 When 検出の塊を評価するとき, the FedDrift client shall 各保有モデルの、その塊での誤り率（予測のargmaxが観測のラベルと違う標本の割合）を求める。
1.2 The FedDrift client shall 誤り率が最小のモデル（同じ値なら保有順で最初）を選び、`最小の誤り率 > 保存した基準値 + δ`のとき検知とする。
1.3 The FedDrift server shall クロス評価の距離を、誤り率で`D(i,j) = max(E_ij − E_ii, E_ji − E_jj, 0)`とする（E_ijはモデルiをモデルjの評価の標本で評価した誤り率）。
1.4 The FedDrift settings shall 基準の種類（今の有界損失と、新しい誤り率）を設定の値で選べ、基準版1〜13の設定は今の有界損失を明示する。
1.5 The FedDrift default settings shall δを、公式コードのデータ別の既定値（`default_deltas`: SEA 0.04、SINE 0.20、CIRCLE 0.10、MNIST 0.10）にする。
  統合の停止の閾値δ'は、論文と公式の既定のとおりδと同じ値にする。δの掃引（論文の付録の0.02〜0.20）は、この仕様の外の実験で行う。
1.6 The FedDrift settings shall δを、掃引の軸で明示した値、組立ての引数で明示した値、データ別の既定の順に優先し、明示した値があるときデータ別の既定を使わない。
  基準版1〜13の設定は0.1を明示する。

### Requirement 2: 新しいモデルの複写元と、次の検知の基準値（D02・D03）
#### Acceptance Criteria
2.1 When 検知したとき, the FedDrift client shall 新しいモデルを、その塊で誤り率が最小の既存モデルから複写する。
2.2 The FedDrift client shall 検知の有無によらず、次の検知の基準値として、その塊での既存モデルの最小の誤り率を保存する。
2.3 The FedDrift settings shall 複写元と基準値の扱いを設定の値で選べ、基準版1〜13の設定は今の扱い（判定前の今のモデル、複写したモデルの値）を明示する。

### Requirement 3: 今の時刻に使われていないモデルを学習・集約しない（D04）
#### Acceptance Criteria
3.1 The FedDrift run shall 各学習のroundで、今の時刻の検出の塊でどのclientにも割り当てられなかったモデルを、どのclientでも学習せず、サーバで集約しない。
3.2 The FedDrift run shall 今の時刻にどれかのclientに割り当てられたモデルは、今と同じく、そのモデルの学習標本を十分に持つ全clientで学習し、集約する。
3.3 The FedDrift run shall 学習しないモデルも保有し続け、次の時刻の選択の候補にする（論文の付録の「再来に備えて保持する」）。
3.4 The FedDrift settings shall この扱いを設定の値で選べ、基準版1〜13の設定は今の扱いを明示する。
3.5 The FedDrift run shall 学習しないモデルの計算・通信を数えない（今の計数の規則のまま、実際に行った処理だけを数える）。

### Requirement 4: 統合の重みに今の時刻の割当を含める（D08）
#### Acceptance Criteria
4.1 When サーバがモデルを統合するとき, the FedDrift server shall 各モデルの重みを、統合の時点で全clientがそのモデルへ割り当てた学習標本の件数の和（今の時刻の塊を含む）にする。
  全clientで件数が0のモデルは、今の重み（初期のモデルの事前学習の重みなど）を使う（式と時点は設計書）。
4.2 The FedDrift settings shall この扱いを設定の値で選べ、基準版1〜13の設定は今の扱いを明示する。

### Requirement 5: 隔離の期間（D10）
公式コードでは、時刻τで検知したclientは`(新しいモデル, τ+W)`で隔離され、時刻τ+Wの最初に解除される（`update_marking`）。公式のFedDriftの既定はW=1で、
このとき隔離は時刻τの中だけで効き、時刻τ+1には解除されて、clientの選び直し・検知と、そのモデルの統合が通常どおり行われる。今の実装の
`new_model_isolation_timestep_count=1`（時刻τ+1から統合の対象）と、W=1では同じ振る舞いになる（設計書で根拠の行とともに確かめる）。

#### Acceptance Criteria
5.1 The FedDrift settings shall 要求1〜4の新しい値を使う設定では、`new_model_isolation_timestep_count`を1だけ受け付け、ほかの値を拒否する（W>1の公式の振る舞い——隔離中のclientの固定と、隔離中のモデルの選択の候補からの除外——は実装しない）。
5.2 The test shall W=1で、新しいモデルを作った次の時刻に、そのモデルが統合の対象になり、ほかのclientの選択の候補にもなることを確かめる。

### Requirement 6: クロス評価の対象のclient（D07の一部）
#### Acceptance Criteria
6.1 The FedDrift default settings shall 1つのモデルを評価させるclientの上限（`maximum_cross_evaluation_client_count`）を、実験のclientの数にする
  （論文のAlgorithm 2の全データの標本での評価に合わせる。増える計算・通信は数えて報告する。理由は設計書）。
6.2 The FedDrift settings for 基準版1〜13 shall 上限3のままとする。

### Requirement 7: 基準版14と既存の値の維持
#### Acceptance Criteria
7.1 The run provenance shall `BEHAVIOR_BASELINE_VERSION`を14にする。基準版14の既定のFedDriftは、要求1〜6の新しい値を全部使う。
7.2 The goldens shall 基準版1〜13の全caseの値を変えない。基準版14のFedSDA・FedAvgのcaseの値は、基準版13と同じ。
7.3 The FedDrift settings shall 新しい値の組合せのうち、意味を持たないもの（要求5.1のW>1、そのほか設計書で列挙するもの）を拒否する。
7.4 The run artifacts shall 新しい設定の値を、runの設定の記録に含める（同じ条件の結果の重複の照合で区別できる）。

### Requirement 8: 文書
#### Acceptance Criteria
8.1 The docs shall `docs/experiments/differences-from-feddrift.md`を、そろえた差と開示する差の現在の状態へ書き直す（今の状態の正本）。「忠実」「論文準拠」は、そろえた範囲と残る差を明示した表現にする。
8.2 The docs shall 監査の資料（`results/implementation-audits/feddrift-20261021/`）と論文の該当箇所を参照する。
8.3 The docs shall 基準版の表（`docs/experiments/refactoring-baseline.md`）に版14を足す。
