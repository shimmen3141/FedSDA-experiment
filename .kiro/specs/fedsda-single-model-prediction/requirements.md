# Requirements Document

## Project Description (Input)

ablationの構成の、移植の6つ目。提案構成から、モデルの予測の結合を外した構成（現在の学習帰属のモデルだけで予測する）を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`hard-routing`）: 構成名`FedSDA_NoCached_ResidualAdapter_ClassESR`（結合の引数は無い。集約後の再較正は、既定の「しない」）。旧のclient（`ResidualAdapterClassConditionalESRFedSDAClient`）は、重み（AdaHedge、Fixed-Share）を持たない。予測は、基底の`_record_prediction`（`clients/base.py`）: 現在のモデルだけを評価し、予測の正否・真の概念・予測したモデルだけを記録する。予測の内訳（どれかのモデルが正解したか、重みでの結合の正否ほか）は、数えない。

## Requirements

### Requirement 1: 現在のモデルだけで予測する構成

#### Acceptance Criteria

1. Where 予測の結合の方式が「現在の学習帰属のモデルでの予測」のとき, the 標本1件の予測 shall 現在の学習帰属のモデルだけを評価して予測する。Fixed-Shareの重みと、AdaHedgeの証拠は、作らない・更新しない。
2. The 標本の予測の記録 shall この方式では、重みに関する項目を、「現在のモデルに重み1」の値で持つ（最大重みのモデルは現在のモデル、最大の重みと実効モデル数は1、どの重みでの結合の正否も、現在のモデルの予測の正否）。
3. The run設定 shall この方式を、集約後の重みの再較正をしない設定との組でだけ、受け付ける。
4. While この方式のとき, the client shall 学習帰属の変更でも、AdaHedgeの証拠を始め直さない（使われない証拠の計数を、動かさない）。
5. The 指標 shall この方式では、結合して予測した標本の数を0とする。
6. The 構成 shall 掃引のコマンドで、`fedsda-single-model-prediction`という名前で選べる。この構成のrun設定は、提案構成と、予測の結合の方式と、再較正の方針だけが違う。
7. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、実旧が持つ指標と離散列の全部が一致する。
8. The 照合 shall この構成に固有の経路を通る: 複数のモデルを保有している標本があり、重みと証拠が1回も使われず、予測したモデルが、実旧と同じ。結果が、提案構成と違う条件がある。
9. The 提案構成 shall 結果が変わらない。
