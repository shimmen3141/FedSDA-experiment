# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。

現在の状況: 最終構成のFedSDAの全体runを、新実装だけで実行でき、最終状態（サーバ、全client、診断の記録、候補検証の判定記録、乱数）が、実旧の全体runと一致する。goldenの条件（sine2）でも一致することを、調査で確かめた。しかし、新実装は、全体runの結果から、指標（精度、検出の適合率・再現率、通信量、モデル数ほか）を計算する処理を持たない。旧は、`run_random_drift_experiment`の後半（`compute_metrics`ほか）で計算し、標本・イベントごとの列を`_save_raw_run`で保存する。最終構成の回帰test（tests/test_proposed_regression.py）は、33の指標と、31の離散列を、goldenと比べる。

変えたいこと: 新の全体runの結果（実行の枠の結果と、参加者の記録）から、goldenが比べる指標のうち、計算量を除く26項目を導出する。導出した指標と、参加者の記録から作れる31の離散列が、実旧の結果、および、Windows用のgoldenと一致することを確かめる。

ユーザーの決定（2026-10-10）: 計算量の指標（`compute_*`の7項目）は、必要である。ただし、旧の計数に漏れ・重複がないかの検査を含めて、後の1つのspecでまとめて扱う。このspecでは、照合の対象から外す。順序は、(1)本spec、(2)計算量のspec、(3)Linux用のgolden。

調査で確かめたこと:

- goldenの26指標の出どころ（旧）: `compute_metrics`（federated_drift_experiment/metrics.py）の`accuracy`・`stable_accuracy`・`precision`・`recall`・`f1`・`total_detect`・`provisional_proposal_count`・`provisional_forward_count`、`run_random_drift_experiment`の`final_model_count`・`comm_*`（12項目。上り・下り・合計）・`final_parameter_values`・`final_parameter_bytes`、`_add_model_diagnostic_results`の`routing_soft_prediction_sample_count`・`routing_switching_recalibration_sample_count`・`routing_aggregation_recalibration_sample_count`。
- 検出の指標は、clientごとに、真の概念の変更位置（概念列の、前の標本と概念が違う位置。処理されなかった末尾を含む、概念列の全体から取る）と、学習帰属の切替の位置を、順に1対1で対応づけて数える（真の変更位置から、許容遅延の範囲内にある、最初の未使用の切替）。定常精度は、各変更位置の直後の決まった件数（回復の窓）を除いた精度。
- 新の記録との対応は、既存の新旧照合のhelper（clientの全状態、サーバの全状態、クラスタリングの診断の記録）が、項目ごとに確かめている。31の離散列は、どれも、新の記録（標本ごとの記録、適応記録、判定記録、登録の来歴、クラスタリングの観測、概念列）の値の並べ替えで作れる。

## Introduction

研究者が、新の全体runの結果から、手法の比較に使う指標を得られるようにする。指標と、記録から作る離散列が、旧の結果とgoldenに一致することを確かめる。

## Boundary Context

- **In scope**: 概念列からの真の概念の変更位置の抽出。変更位置とイベントの対応づけ、検出の適合率・再現率・F1、精度、定常精度の計算。全体runの結果と参加者からの、26指標の導出。導出した指標と、記録から作る31の離散列の、実旧の結果（同じprocessの中の`run_random_drift_experiment`と、その保存結果）およびWindows用のgoldenとの照合。
- **Out of scope**: 計算量の指標（次のspec）。goldenが比べない指標（警報の適合率、誤検出の内訳、遅延、変化点の誤差、候補の判定の平均ほか）と、診断の集計。結果の保存（NPZ・CSV）と図。Linux用のgolden（後のspec）。sine2以外のdataset。
- **Adjacent expectations**: 全体runの実行、参加者、記録のownerは、変えない。固定旧実装とgoldenは、変えない。

## Requirements

### Requirement 1: 真の概念の変更位置と、イベントの対応づけ

**Objective:** As a 研究者, I want 概念列から変更位置を取り出し、イベントと対応づけてほしい, so that 検出の指標を、旧と同じ規則で数えられる

#### Acceptance Criteria

1. When 概念列を与えたとき, the Run Metric Calculations shall 前の標本と概念が違う標本位置を、昇順で返す（先頭の標本は、変更位置にしない）。
2. When 変更位置の列と、イベントの位置の列と、許容遅延を与えたとき, the Run Metric Calculations shall 変更位置を与えられた順に見て、それぞれに、変更位置以上・変更位置＋許容遅延以下にある、まだ使っていない最初のイベント（与えられた順）を対応づけ、対応したイベントの数と、対応した変更位置の数を返す。
3. If 入力が、決まった型・範囲でないとき（整数のtupleでない、負の位置、負の許容遅延、boolなど）, the Run Metric Calculations shall 拒否する。

### Requirement 2: 精度と検出の指標

**Objective:** As a 研究者, I want 精度・定常精度・検出の適合率と再現率を得たい, so that 手法を比べられる

#### Acceptance Criteria

1. When clientごとの予測の正誤の列を与えたとき, the Run Metric Calculations shall 全clientの全標本の正解の割合を返す（標本がなければ0）。
2. When 正誤の列・変更位置・回復の窓の件数を与えたとき, the Run Metric Calculations shall 各clientで、その標本位置以下の最も近い変更位置から、窓の件数より手前にある標本を除いた、正解の割合を返す（最初の変更位置より前の標本は含める。残る標本がなければNaN）。
3. When clientごとの変更位置と検出位置を与えたとき, the Run Metric Calculations shall clientごとに、検出位置を昇順にして対応づけ、全clientの合計で、適合率（対応した検出÷検出）、再現率（対応した検出÷変更位置）、F1、検出の数を返す（分母が0のときは0）。

### Requirement 3: 全体runからの導出

**Objective:** As a 研究者, I want 全体runの結果と参加者から、指標をまとめて得たい, so that runごとに、同じ形の結果が得られる

#### Acceptance Criteria

1. When 全体runの結果・参加者・指標の設定（許容遅延、回復の窓）を与えたとき, the FedSDA Run Metric Derivation shall 次を返す: 精度、定常精度、検出の適合率・再現率・F1・検出の数（学習帰属の切替の位置を検出とする）、最終のグローバルモデルの数、通信量（上りと下りの、モデル・軽量メッセージ・パラメータの値の数・バイト数）、最終のパラメータ量（共有部を1回、概念固有部をモデルごとに数えた、値の数とバイト数）、候補検証の判定の数、混合予測を行った標本の数、集約後の再較正で再生した標本の数（予測の重みと、globalの診断証拠のそれぞれ。全clientの合計）。
2. The 導出 shall 参加者と記録の状態、乱数を変えない。
3. If 参加者が、最終構成のFedSDAのclientとサーバでない、clientの数が概念列の数と合わない、または設定が決まった型・範囲でないとき, the FedSDA Run Metric Derivation shall 拒否する。

### Requirement 4: 実旧・goldenとの一致

**Objective:** As a 研究者, I want 導出した指標と、記録から作る列が、旧の結果とgoldenに一致してほしい, so that 新実装の結果を、旧の結果の代わりに使える

#### Acceptance Criteria

1. When goldenの条件（sine2）で、新の全体runと、実旧の`run_random_drift_experiment`を、同じprocessで実行したとき, the 導出した指標 shall 旧の結果の26項目と、値が完全に一致する。
2. When 同じ実行で、新の記録から、旧の保存形式の31の離散列を作ったとき, the 列 shall 旧が保存した配列と、形・型・値が完全に一致する。
3. When Windowsの基準環境で実行したとき, the 導出した指標と列 shall Windows用のgolden（tests/proposed_regression_golden.json）のsine2の、26指標（絶対誤差1e-9）と、31の離散列（形と、正規化したJSONのSHA-256）に一致する。goldenとその回帰testは、変更しない。
4. When 小さい条件（処理されない末尾がある条件を含む）で実行したとき, the 導出した指標 shall 実旧の全体runの後に、旧の指標の計算を呼んだ結果と、一致する。
5. The 計算の部品 shall 実旧の関数（対応づけ、定常精度、指標の計算）と、境界を含む入力で一致する。

### Requirement 5: 新実装だけで動くこと

#### Acceptance Criteria

1. When 共用のfresh processの確認を実行したとき, the 共用script shall 旧実装とtestのmoduleを読み込まずに、全体runの後で、指標を導出できる。
2. The 追加 shall 依存の許可の範囲に収める。
