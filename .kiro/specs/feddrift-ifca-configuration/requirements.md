# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedDriftとFedAvgだけ（2026-11-01のユーザー判断で、Oracle・IFCA・余裕があればAdaptive-FedAvgを足す）。FedDriftの論文（AISTATS 2023）は、時間とともに変わる設定へ窓を足したIFCA（IFCA+Window）を比較手法にしている。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、IFCA（クラスタの数 $K$ を与え、各clientが時刻ごとに損失が最小のクラスタのモデルを選ぶ）に学習の窓（1時刻）を足した構成を2つ足す。2つは初期化だけが違う。既存の構成と既定は変えない。Oracleの構成（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`）の後に実装する。

## Boundary Context
- 対象: 新しい2構成、それを守るtest、文書、本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、研究ツール、Oracle・Adaptive-FedAvgの構成。

## Requirements
### Requirement 1: IFCA+Windowの2構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-ifca-window`と`feddrift-ifca-window-pretrained-cluster0`を持つ。どちらも、クラスタの数 $K$ をデータのコンセプトの数（`dataset_definitions`の`concept_count`）とし、runの最初に $K$ 個のグローバルモデルを持つ（$K$ を正しく与えることはIFCAに有利な条件で、文書と論文に書く）。
2. The system shall モデルの初期値を構成ごとに次のとおりにする。`feddrift-ifca-window`: $K$ 個とも、事前学習をせず、runのseedとモデルのIDから決まる分けた乱数の状態で初期化する（IFCAの元の論文とFedDriftの論文のクラスタごとの乱数の初期化に当たる）。`feddrift-ifca-window-pretrained-cluster0`: ID 0だけを全手法に共通の事前学習した初期モデルの写しにし、ID 1〜$K-1$は前者と同じ乱数の初期化にする。初期化でrunの乱数を消費しない。
3. When 各時刻の終わりに検出バッチ（$I$ 標本）がそろう, the system shall そのclientの $K$ 個のモデルそれぞれの、そのバッチでの平均損失（FedDriftの検出と同じ損失）を求め、最小のモデルを、そのバッチの学習の行き先と、次の時刻の予測のモデルにする（同値はIDの小さい方）。最初の時刻は、全clientの学習の行き先と予測のモデルをID 0にし（FedDriftの論文の付録のIFCAの初期化）、2つ目の時刻から損失が最小のモデルを選ぶ。FedDriftの変化の検出・新しいモデルの作成・統合は行わない。
4. The system shall 学習の標本を、今の時刻にそのモデルへ帰属した標本だけにする（窓1時刻）。学習のround数、学習の対象の規則、件数の条件、集約の重み、通信の数え方、学習率は、FedDriftの構成`feddrift`と同じ規則にする（具体的な接続はdesign.mdに定める）。
5. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 2構成が構成の一覧に含まれ、組立てが例外を出さず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 2構成の初期値が要求1.2のとおりであること（`feddrift-ifca-window`の $K$ 個が互いに異なり事前学習の初期モデルとも異なる、`feddrift-ifca-window-pretrained-cluster0`のID 0が事前学習の初期モデルと同じ、同じseedで同じになる）を、runの組立ての部品のtestで確かめる。
3. The validation shall 小さい全体run（design.mdに定める固定条件）で、選択・学習の窓・モデルの数・登録・統合を記録から確かめる（design.md）。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と本規模の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、2構成と、FedDriftの論文のIFCA+Windowとの違い（design.mdの「論文のIFCA+Windowとの違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、2構成×10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本（800 run）のコマンドをユーザーに渡す。集計はdesign.mdの「本規模の集計」。
