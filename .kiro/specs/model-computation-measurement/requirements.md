# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 新実装は、全体runの結果から、goldenが比べる33指標のうち26項目を導出でき、実旧・Windows用のgoldenと一致する。残る7項目は、計算量（`compute_*`）である。旧は、clientが、処理の各所で計数を足す（`compute_counters`。約40か所）。新実装には、計数がない。

ユーザーの決定（2026-10-10）: 計算量の指標は必要。旧の計数に、漏れ・重複がないかの検査を含めて、1つのspecで扱う。

調査で分かったこと（2026-10-10、主担当の実測。下書きのtest。commitしていない）:

- 旧の計算量は、四則演算のような細かい単位ではなく、「モデルへ入力した標本の数」と「更新の回数」の計数である。モデルの大きさ（層の幅、アダプタのrank）は反映しない。
- **旧の計数は正確だった。** goldenの3ケース（sine2・sea2・mnist2）で、旧の全体runの間、共有部（`SharedFeatureBackbone`）と概念固有部（`ResidualConceptAdapter`）の順伝播、および、optimizerの更新を、PyTorchのhookで外側から数えた。共有部を通った標本数、概念固有部を通った標本数、勾配つき（学習）の標本数、勾配なし（推論）の標本数、共有部と概念固有部のoptimizerの更新回数は、どれも、旧の計数（`backbone_examples`、`head_examples`、`training_examples`、学習以外の用途の標本数の合計、`backbone_optimizer_steps`、`head_optimizer_steps`＝`optimizer_steps`）と、完全に一致した。
- 旧の計数は、clientの計算だけを数える。初期モデルの事前学習の計算は、含まない（定義）。
- **新実装は、旧より多く計算している。** goldenの条件（sine2）の新の全体runを、同じ方法で数えると、学習の標本数（322048）と、optimizerの更新回数（概念固有部10165、共有部4412）は、旧と一致するが、推論が多い: 概念固有部を通った標本数が+2057、共有部を通った標本数が+5776。結果（状態と指標）は、旧と同じである。呼出し元ごとの内訳から、原因は3箇所:
  1. クロス評価（正誤を比べるとき）: 渡されたモデルの順伝播を、損失のためと、予測のために、2回行う（旧は、1回の順伝播から、損失と予測を得る）。+977標本。
  2. 警報時の区間の準備: 変更区間より前の保留標本の損失を、検査のために計算し、その後、取込み（損失統計の更新）で、もう一度計算する。+1080標本。
  3. 集約後の再較正: 保留標本の損失を、保有モデルごとに、共有部から計算し直す（旧は、共有部の特徴を1回だけ計算し、概念固有部だけをモデルごとに計算する）。共有部 +3719標本。
- 検出器の計数（更新回数、その回に評価した「候補の変化点×賭け率」の数）は、新の監視が、標本ごとの観測結果（`LossMonitoringObservation`の`component_update_count`・`evaluated_candidate_bet_count`）として、すでに返している。合計を保持する場所がない。

変えたいこと: (1)旧の計数の検査を、恒久のtestとして残す。(2)新実装の計算量を、モデルの順伝播とoptimizerの更新を、実行中に外側から数える方法で計測する（数え方を、各部品へ書き込まない）。(3)新実装の3箇所の重複をなくし、実際の計算量を、旧と同じにする。(4)goldenの計算量の7項目を導出して、実旧・goldenと照合する。

## Introduction

研究者が、新実装の全体runの計算量を、漏れ・重複が起きにくい方法で得られるようにする。新実装の実際の計算量を、旧と同じにする。

## Boundary Context

- **In scope**: 旧の計数と、外側からの数え直しの照合（test）。モデルの計算の計測（共有部・概念固有部を通った標本数の、学習と推論の別、optimizerの更新回数）。検出器の計算の計数の保持。3箇所の重複の解消。計測つきの全体runの実行と、計算量の指標の導出、goldenの7項目との照合。
- **Out of scope**: 用途別の内訳（予測・検出・統計・クロス評価・初期化・再較正）と、ラウンドごと・clientごとの時系列（旧の`telemetry`）。実行時間。モデルの大きさを反映した計算量（FLOPsほか）。事前学習の計算を含めること。sine2以外のdataset。
- **Adjacent expectations**: 3箇所の重複の解消は、結果（状態、乱数の消費、例外）を変えない。固定旧実装とgoldenは、変えない。

## Requirements

### Requirement 1: 旧の計数の検査

**Objective:** As a 研究者, I want 旧の計算量の計数に、漏れ・重複がないことを確かめたい, so that goldenの計算量を、正しい基準として使える

#### Acceptance Criteria

1. When goldenの3ケース（sine2・sea2・mnist2）で、旧の全体runを実行したとき, the 検査 shall 共有部と概念固有部の順伝播へ入力された標本数（勾配の有無の別）と、optimizerの更新回数（共有部・概念固有部の別）を、外側から数え、旧の計数の対応する合計と一致することを確かめる。
2. The 検査 shall 恒久のtestとして残す（旧実装は変えない）。

### Requirement 2: モデルの計算の計測

**Objective:** As a 研究者, I want モデルの計算を、部品へ数え方を書き込まずに計測したい, so that 部品を足しても、漏れ・重複が起きない

#### Acceptance Criteria

1. While 計測の区間の中, the Model Computation Measurement shall 共有部（`SharedFeatureExtractor`）と概念固有部（`NonlinearResidualAdapter`）の順伝播のたびに、入力された標本数を、勾配が有効なら学習、無効なら推論として、足す。
2. While 計測の区間の中, the Model Computation Measurement shall optimizerの更新のたびに、共有部のパラメータを持つoptimizerなら共有部の更新回数、そうでなければ概念固有部の更新回数を、1足す。
3. When 計測の区間を出たとき（例外で出たときを含む）, the Model Computation Measurement shall 計測のための登録を外し、以後のモデルの計算に影響しない。
4. The Model Computation Measurement shall 途中の時点の計数を読め、2つの時点の計数の差を計算できる。
5. The 計測 shall モデルの計算の結果と、乱数の消費を変えない。

### Requirement 3: 検出器の計算の計数

#### Acceptance Criteria

1. When clientが標本1件を処理したとき, the FedSDA Run Client shall 監視の観測結果の、検出器の更新回数と、評価した候補×賭け率の数を、clientの計数へ足す。
2. The 計数 shall 全体runの後で、合計として読める。実旧の対応する計数（`drift_detector_updates`、`drift_detector_hypotheses`）と一致する。

### Requirement 4: 重複した計算の解消

**Objective:** As a 研究者, I want 新実装が、旧より多く計算しないでほしい, so that 計算量の指標が、旧と同じ基準で比べられる

#### Acceptance Criteria

1. When クロス評価で、正誤を比べるとき, the Client Model Cross Evaluation shall 渡されたモデルの順伝播を1回だけ行い、損失と予測の両方を、その出力から得る。
2. When 警報時に、変更区間より前の保留標本を取り込むとき, the Alarm Training Interval Preparation shall それらの標本の損失を、1回だけ計算する（検査は、順伝播を行わない方法で、状態の更新より前に行う）。
3. When 集約後の再較正で、保留標本の損失を、保有する全モデルで計算するとき, the Post Aggregation Prediction Recalibration shall 共有部の特徴を1回だけ計算し、概念固有部だけを、モデルごとに計算する。
4. The 解消 shall 結果（戻り値、状態、乱数の消費、拒否する入力と、その時点で状態を変えないこと）を変えない（既存の、実旧との対照が、そのまま通る）。
5. The 3箇所 shall 解消の後の順伝播の標本数が、実旧の同じ処理の計数と一致することを、testで確かめる。

### Requirement 5: 計測つきの全体runと、計算量の指標

#### Acceptance Criteria

1. When 計測つきの全体runを実行したとき, the Measured Run Execution shall 参加者の初期準備（事前学習を含む）が終わった時点から、全体runの終わりまでの、モデルの計算の計数を返す（事前学習の計算は含めない。旧と同じ定義）。
2. When 全体runの結果・参加者・モデルの計算の計数を与えたとき, the FedSDA Run Metric Derivation shall モデルの計算の計数と、検出器の計算の計数（全clientの合計）を、指標へ含める。
3. When goldenの条件（sine2）で実行したとき, the 計算量の指標 shall 実旧の実行結果の、計算量の7項目（推論の標本数、学習の標本数、optimizerの更新回数、共有部を通った標本数、概念固有部を通った標本数、検出器の更新回数、検出器の候補の数）と、完全に一致し、Windowsの基準環境では、Windows用のgoldenと一致する。これで、goldenの33指標のすべてが、照合される。
4. If 計測なしで導出するとき, the FedSDA Run Metric Derivation shall モデルの計算の計数を「なし」として、ほかの指標を、これまでどおり返す。

### Requirement 6: 新実装だけで動くこと

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、計測つきの全体runを実行して、計算量の指標を得られる。
2. The 追加 shall 依存の許可の範囲に収める。
