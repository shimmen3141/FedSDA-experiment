# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: 予測を最後の同期の時点のモデルで行う構成（`fedsda-synchronized-model-prediction`）は、mnist・sea2・sine2の間隔50で上がるが、1つのモデルが離れた概念を受け持つsea4の端の概念と、同期の時点のモデルが古くなる間隔200のcircle2・sine2で、局所更新したモデルの予測に劣る（研究バックログの8）。手元の診断で、同期の時点のモデルの結合予測と、局所のモデルの結合予測を、clientごとの2つの候補の重みで混ぜると、sea4（+0.47・+0.51）、circle2・200（+0.89）、sine2・200（+0.67）で全seedが上がり、mnist2ではほぼ同じだった。固定の半々で混ぜるより、適応的な重みのほうが全条件で上だった。
- 何を変えるか: この混ぜ方を、予測に使うパラメータの設定の3つ目の値として選べるようにし、構成名で掃引できるようにする。既定の動作と値は変えない。

## Introduction
2026-10-15の会話で、ユーザーは、この案を実装すると決め、名前の案を求めた。診断は、ソースを変えずに予測の関数を実行時に包んで行った（`results/fedsda-training-recency/20261015_circle2-diagnosis/local_sync_mixture_probe_2.py`）。実装は、診断と同じ予測になることを確かめる。

## Boundary Context
- 対象: 予測の設定の値、2つの候補の重みのowner、標本の予測（結合して予測する2つの設定）、clientの接続、構成名、文書。
- 対象外: 学習・検出・候補の検証・統計・アップロード（現在の値のまま）、集約後の予測重みの再較正（モデルの重みだけを作り直し、2つの候補の重みは作り直さない）、現在のモデルだけで予測する設定との組合せ（拒否する）、既定。
- 用語: 「同期の結合予測」は、同期の時点のモデル（`parameters_at_last_synchronization`と同じ選び方）の確率を、予測に使うモデルの重みで結合したもの。「局所の結合予測」は、保有モデルの現在の分類器の確率を、同じ重みで結合したもの。

## Requirements
### Requirement 1: 2つの結合予測を混ぜて予測できる
**Objective:** As a 研究者, I want 同期の結合予測と局所の結合予測を、clientごとの重みで混ぜて予測できること, so that 場面に応じて2つの良い所を使える。
#### Acceptance Criteria
1. The FedSDA system shall 予測に使うパラメータの設定（`prediction_model_parameter_source`）に、値`mixture_of_synchronized_and_local_parameters`を持つ。既定（`current_local_parameters`）と`parameters_at_last_synchronization`の動作と値は変えない。
2. When この値のとき, the FedSDA client shall 配布の反映の直後に保有モデルを写し（`parameters_at_last_synchronization`と同じ）、各標本で、同期の結合予測と局所の結合予測を、予測に使うモデルの重み（Fixed-ShareまたはAdaHedge）で作り、clientごとの2つの候補の重み（最初は半々）で混ぜた確率で予測する。予測の記録の`combined_prediction_is_correct`は、混ぜた確率での予測の正否、`any_model_or_combined_prediction_is_correct`は、その正否と同期の側のモデルごとの正否の和（この2項目から作る指標も、混ぜた確率の側）。記録のそれ以外の項目（モデルごとの正否、最大重みのモデル、重みごとの結合の正否ほか）と、モデルの重みの更新は、同期の結合予測の側（`parameters_at_last_synchronization`と同じ）で行う。
3. When 予測の正否が決まった後, the FedSDA client shall 2つの候補の重みに、それぞれの損失（1−真のクラスの確率。確率をPythonのfloatへ直してから倍精度で計算する。診断と同じ式）の`exp(−2·損失)`を掛けて正規化し、Fixed-Shareの時間尺度Tの逆数`1/T`を互いに分け合う。
4. If この値と、現在のモデルだけで予測する設定（`current_training_model_prediction`）を組み合わせたとき, the FedSDA system shall 設定の検査で拒否する。
5. The FedSDA system shall 上の2・3の予測が、診断の差し替え（同期の時点の予測の構成で、予測の関数を包んで混ぜた確率の予測の正否）と、標本ごとに同じになる。

### Requirement 2: 構成名で選べる
**Objective:** As a 研究者, I want この予測を構成名1つで実行できること, so that 研究室サーバで、ほかの構成と同じ条件で比べられる。
#### Acceptance Criteria
1. The sweep command shall 構成`fedsda-synchronized-local-mixture-prediction`を受け、そのrun設定は、提案構成と、予測に使うパラメータの設定だけが違う。帰属の保留の容量の軸と組み合わせられる。
2. The configuration shall goldenの事例にせず、専用のtestで守る構成の一覧（`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）に登録する。

### Requirement 3: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存のgoldenと照合が値を変えずに通ること, so that 既定の動作が変わっていないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenと、既存の構成のtest（同期の時点の予測の構成の、診断との一致を含む）を、値を変えずに通す。設定の項目は増えないので、runの識別子は変わらない。

### Requirement 4: 文書
**Objective:** As a 研究者, I want 文書が新しい構成を説明すること, so that サーバのコマンドを正しく書ける。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書の構成の表に、新しい構成と仕組み（混ぜる重み、学習率2、分け合い1/T）を書く。研究バックログの8に、実装したことを書く。
