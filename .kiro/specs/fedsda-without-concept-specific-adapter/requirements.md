# Requirements Document

## Project Description (Input)

ablationの構成の、移植の9つ目。提案構成のモデルから、概念固有の残差adapterを外した構成を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`shared-backbone`）: 構成名`FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting`（ほかの引数は、提案構成と同じ。adapterのrankは渡さない）。モデルは`SharedBackboneMLP`（共有の特徴抽出部と、概念別の出力層）。提案構成のモデル（`ResidualAdapterMLP`）は、これを継承して、共有特徴と出力層の間に、概念別の残差adapterを足したもの。clientとサーバは、2つの構成で同じ（提案構成のclientのクラスは、この構成のclientのクラスを、上書きなしで継承する）。

## Requirements

### Requirement 1: adapterを持たないモデル

#### Acceptance Criteria

1. Where モデル構造が「adapterを持たない」のとき, the 分類器 shall 共有特徴を、そのまま、概念別の分類層へ渡す。概念固有部のパラメータは、分類層だけ。
2. The 分類器 shall この構造では、構築のとき、adapterのための乱数を消費しない（同じ乱数の状態から作ると、パラメータと、構築の後の乱数の状態が、旧の`SharedBackboneMLP`と一致する）。
3. The 計算量の計測 shall この構造でも、概念固有部へ入力された標本数を数える。

### Requirement 2: 構成

#### Acceptance Criteria

1. The 構成 shall 掃引のコマンドで、`fedsda-without-concept-specific-adapter`という名前で選べる。run設定は、提案構成と、モデル構造の名前だけが違う。
2. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
3. The 照合 shall 固有の経路を通る: 保有モデルが、adapterのパラメータを持たず、概念固有部のパラメータの数が、実旧のモデルと同じ。複数のモデルを保有するclientがある。結果と、最終のパラメータの数が、提案構成と違う。
4. The 提案構成 shall 結果が変わらない。
