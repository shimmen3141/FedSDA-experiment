# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: 比較手法はFedDriftとFedAvgだけ（2026-11-01のユーザー判断で、Oracle・IFCA・余裕があればAdaptive-FedAvgを足す）。FedDriftの論文（AISTATS 2023）は、時間とともに変わる設定へ窓を足したIFCA（IFCA+Window）を比較手法にしている。
- 何を変えるか: 手元のFedDriftの実装（`methods/feddrift`、`runtime/feddrift_*`）の上に、IFCA（クラスタの数 $K$ を与え、各clientが時刻ごとに損失が最小のクラスタのモデルを選ぶ）の構成`feddrift-ifca-window`を足す。学習の標本は今の時刻の標本だけにする（窓1時刻）。既存の構成と既定は変えない。Oracleの構成（`.kiro/specs/feddrift-oracle-concept-assignment-configuration/`）の後に実装する。

## Boundary Context
- 対象: 新しい構成、その構成を守るtest、文書、本規模の実行のコマンド。
- 対象外: 既存の構成の設定・値・runの識別子・golden、研究ツール、Oracle・Adaptive-FedAvgの構成。

## Requirements
### Requirement 1: IFCAの構成
#### Acceptance Criteria
1. The system shall 構成`feddrift-ifca-window`を持つ。クラスタの数 $K$ はデータのコンセプトの数（`dataset_definitions`の`concept_count`）とし、runの最初に $K$ 個のグローバルモデルを持つ（$K$ を正しく与えることはIFCAに有利な条件で、文書と論文に書く）。
2. The system shall モデルの初期値を次のとおりにする: ID 0は全手法に共通の事前学習した初期モデルの写し。ID 1〜$K-1$は、事前学習をせずに、runのseedとモデルのIDから決まる別の乱数で初期化したもの（IFCAの元の論文のクラスタごとの乱数の初期化に当たる。全部を同じ初期モデルにするとクラスタが分かれないため）。
3. When 各時刻の終わりに検出バッチ（$I$ 標本）がそろう, the system shall そのclientの $K$ 個のモデルそれぞれの、そのバッチでの平均損失（FedDriftの検出と同じ損失）を求め、最小のモデルを次の時刻の予測と、そのバッチの学習の行き先にする（同値はIDの小さい方）。FedDriftの変化の検出・新しいモデルの作成・統合は行わない。
4. The system shall 学習の標本を、今の時刻にそのモデルへ帰属した標本だけにする（窓1時刻。過去の時刻の標本は学習に使わない）。学習のround数、学習の対象の規則（その時刻にいずれかのclientに選ばれたモデル）、件数の条件、集約の重み、通信の数え方、学習率は、FedDriftの構成`feddrift`と同じにする。
5. The system shall 既存の全部の構成の設定とrunの識別子を変えない。

### Requirement 2: 構成を守るtest
#### Acceptance Criteria
1. The validation shall 構成の一覧に含まれ、組立てが例外を出さず、`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`に専用のtestと一緒に登録されていることを確かめる。
2. The validation shall 小さい全体run（SINE2とMNIST4、間隔50、seed 0、10 client、各client 2500標本）で、(a) モデルの数がすべての時刻で $K$ で、作成・統合の記録が0件、(b) ID 1〜$K-1$の初期値がID 0と異なり、同じseedで同じになる、(c) 各時刻の選択が、そのバッチでの平均損失が最小のモデル、(d) 学習の標本がその時刻の標本だけ、を記録またはowner の状態から確かめる。
3. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子のtestが値を変えずに通ることを確かめる。

### Requirement 3: 文書と本規模の実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧と`docs/experiments/differences-from-feddrift.md`に、新しい構成と、FedDriftの論文のIFCA+Windowとの違い（逐次の評価、事前学習のモデルと乱数の初期化の組合せ）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、10データ×間隔50・100・200・500×seed 0〜9、10 client×5000標本のコマンドをユーザーに渡す。集計はOracleのspecの「本規模の集計」と同じ項目。
