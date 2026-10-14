# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: FedSDAのclientは、標本を1つ予測するたびに保有モデルを局所更新し、round内の次の標本は局所更新したモデルで予測する。round内の位置ごとの精度を見ると、集約・配布の直後の予測はFedDriftと同程度だが、局所更新が入った直後から下がる（研究バックログの8）。手元の診断で、予測と予測重みの更新だけを最後の同期の時点のモデルで行い、学習・検出・候補の検証は従来どおり局所更新したモデルで進めると、circle2・sine2・sea2の3データ×seed 5つの全部で、精度が上がった（+0.17〜+0.91ポイント）。
- 何を変えるか: この予測の仕方を設定で選べるようにし、構成名で掃引できるようにする。既定の動作と値は変えない（本規模の比較の後に、既定にするかを決める）。

## Introduction
2026-10-15の会話で、ユーザーは、この方式をFedSDAへ実装すると決めた。手元の診断は、ソースを変えずに実行時に関数を差し替えて行った（`results/fedsda-training-recency/20261015_circle2-diagnosis/fedsda_distributed_prediction_probe.py`）。実装は、この診断と同じ予測になることを確かめる。

## Boundary Context
- 対象: FedSDAのclientの、標本1件の予測（全モデルの結合と、現在のモデルだけの予測の両方）が使うパラメータ。同期の時点の保有モデルの写しを持つowner。予測の設定の項目。構成名。goldenの照合での新しい設定の項目の扱い。文書。
- 対象外: 学習・損失の監視（検出）・候補の検証・統計・アップロード（局所更新したモデルのまま）。集約後の予測重みの再較正（配布の直後に行うので、その時点の局所の値は同期の時点の値と同じ）。FedDrift・FedAvg。既定の変更。
- 用語: 「同期の時点のモデル」は、サーバからの配布をclientが反映し終えた時点の、clientの保有モデル全部（配布されたモデルと、まだ登録していない局所のモデルを含む）の値。

## Requirements
### Requirement 1: 予測に使うパラメータを選べる
**Objective:** As a 研究者, I want 予測を、局所更新したモデルで行うか、最後の同期の時点のモデルで行うかを選べること, so that 予測の仕方の違いが精度に効くかを、本規模で比べられる。
#### Acceptance Criteria
1. The FedSDA system shall 予測の設定に「予測に使うパラメータ」の項目を持ち、値は`current_local_parameters`（今の動作）と`parameters_at_last_synchronization`の2つ。既定の構成（`fedsda`ほか既存の構成）は`current_local_parameters`で、乱数の使い方と結果の値は今と同じ。
2. When `parameters_at_last_synchronization`のとき, the FedSDA client shall 配布を反映し終えた時点で、保有モデル全部の値を写して持ち、次の同期まで変えない。
3. When `parameters_at_last_synchronization`で標本を予測するとき, the FedSDA client shall 写しを持つ保有モデル（同期の時点の分類器のまま——同じオブジェクト——のもの）は写しの値で、写しを持たない保有モデル（同期の後に作ったもの）は、共有部を写しの値、モデル固有部を現在の値で評価し、予測・予測の記録・予測重みと証拠の更新を行う。写しを持つモデルが無い（最初の同期の前）ときは、現在の値で行う。結合して予測する設定と、現在のモデルだけで予測する設定の両方に効く。
4. The FedSDA client shall 損失の監視・候補の検証・学習・統計・アップロードには、どちらの設定でも、現在の値（局所更新したモデル）を使う。
5. The FedSDA system shall 上の3の予測が、手元の診断の差し替え（配布の後に保有モデル全部のstate_dictを控え、予測の間だけ読み込み、終わったら戻す）と、同じ予測の記録と指標になる。

### Requirement 2: 構成名で選べる
**Objective:** As a 研究者, I want この予測の仕方を構成名1つで実行できること, so that 研究室サーバで、提案構成と同じ条件で比べられる。
#### Acceptance Criteria
1. The sweep command shall 構成`fedsda-synchronized-model-prediction`を受け、そのrun設定は、提案構成と「予測に使うパラメータ」の項目だけが違う。
2. The configuration shall goldenの事例にせず、専用のtestで守る構成の一覧（`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）に登録する。旧実装に対応する構成は無いので、旧とは照合しない。

### Requirement 3: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存のgoldenと照合が値を変えずに通ること, so that 既定の動作が変わっていないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenと、実旧との照合を、値を変えずに通す（新しい設定の項目は、どの版でも既定の値であることを確かめてから、比較から除く）。
2. FedSDAのrunの識別子は、run設定の保存の形が変わるので、変わる（前のspecと同じ扱い）。

### Requirement 4: 文書
**Objective:** As a 研究者, I want 文書が新しい構成を説明すること, so that サーバのコマンドを正しく書ける。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書に、新しい構成と、何を同期の時点の値で行い、何を現在の値で行うかを書く。研究バックログの8に、実装したことを書く。
