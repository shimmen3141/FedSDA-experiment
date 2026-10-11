# Requirements Document

## Project Description (Input)

ablationの構成の、移植の5つ目。提案構成の、予測に使う重みを、Fixed-Shareの重みから、AdaHedgeの重みへ替えた構成を、新実装で実行できるようにする。合わせて、重みの出どころの名前を、役割（予測・診断）から、方式（Fixed-Share・AdaHedge）へ改める（ユーザー決定の案A）。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`global-routing`）: 提案構成と同じ構成名で、`--soft-routing-context global`（集約後の再較正は、提案構成と同じ`fifo_replay`）。旧のclient（`clients/fedsda.py`の`_record_prediction`）は、どちらの指定でも、AdaHedgeの重み（`expert_router`）と、Fixed-Shareの重み（`switching_expert_router`）の両方を、毎標本で更新する。予測に使う重み（`probabilities`）だけが違う。使わない側の、結合の正否・最大重みのモデル・実効モデル数も、記録する。

## Requirements

### Requirement 1: 重みの出どころの名前

#### Acceptance Criteria

1. The 新実装 shall AdaHedgeの重みと、その証拠・記録・指標・表の列を、方式の名前（`adahedge_…`、`AdaHedgeWeightEvidence`）で持つ。真の概念別の重み（真の概念を使うので、常に診断）は、名前を変えない。
2. The 新実装 shall Fixed-Shareの重みの保持・計数・指標・表の列を、方式の名前（`fixed_share_…`、`FixedShareWeightController`）で持つ。
3. The 「結合した予測」「最大重みのモデル」「最大の予測重み」「実効モデル数」 shall 予測に使う重みでの値を表す。
4. The 改名 shall 値を変えない（改名だけの時点で、全pytestが、改名の前と同じ結果）。

### Requirement 2: AdaHedgeの重みで予測する構成

#### Acceptance Criteria

1. Where 予測の結合の方式が「AdaHedgeの重みでの結合」のとき, the 標本1件の予測 shall AdaHedgeの重みで、全保有モデルの確率を結合して予測する。Fixed-Shareの重みと、AdaHedgeの証拠の更新は、提案構成と同じに行う。
2. The 標本の予測の記録 shall どちらの方式でも、Fixed-Shareの重みでの、結合の正否・最大重みのモデル・実効モデル数を持つ（Fixed-Shareの重みで予測する方式では、予測に使う重みでの項目と同じ値）。
3. The 予測の診断の指標と、記録の表 shall Fixed-Shareの重みでの、正解の数・定常と回復の区間の正解の割合・実効モデル数の平均と、標本ごとの3列を持つ。
4. The 構成 shall 掃引のコマンドで、`fedsda-adahedge-prediction-weights`という名前で選べる。この構成のrun設定は、提案構成と、予測の結合の方式だけが違う。
5. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
6. The 照合 shall この構成に固有の経路を通る: 実際の予測の正否が、全標本で、AdaHedgeの重みでの結合の正否と同じで、Fixed-Share側の項目と違う標本がある。提案構成では、逆。結果が、提案構成と違う条件がある。
7. The 提案構成 shall 結果が変わらない。
