# Requirements Document

## Project Description (Input)

ablationの構成の、移植の10個目（最後）。提案構成のモデルを、共有部を持たないモデルへ替えた構成を、新実装で実行できるようにする。旧実装での違いの全部（14項目）は、[調査の文書](../../../docs/research/independent-models-migration-scope.md)にある。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`independent`）: 構成名`FedSDA_NoCached_ClassESR_RestartingSoftRouting`。モデルは`SimpleMLP`（隠れ層と出力層が、1つのまとまり。optimizerは1つ）。clientは、共有表現のmixinを持たず、サーバは`FedSDANoCachedServer`。

## Requirements

### Requirement 1: 共有部を持たないモデル

#### Acceptance Criteria

1. Where モデル構造が「共有部を持たない」のとき, the 分類器 shall 隠れ層と分類層の全部を、モデルごとに独立に持つ（保有モデルどうしが、パラメータを共有しない）。同じ乱数の状態から作ると、パラメータ・出力・構築の後の乱数の状態が、旧の`SimpleMLP`と一致する。
2. The client shall この構造では、採用した候補と、配布を受け取った後の一時モデルの、optimizerの状態を、作り直さない（旧の、共有部を持たないモデルは、共有部のつなぎ直しをしない）。配布で作り直すモデルのoptimizerは、どのモデルも、作り直しの設定で作る。
3. The 計算量の計測 shall この構造では、共有部の計数を0とし、概念固有部の計数に、モデルの全部の層を数える。

### Requirement 2: モデルごとに順に学習する方式

#### Acceptance Criteria

1. Where ローカル学習の方式が「モデルごとに順に」のとき, the client shall 標本を持つ保有モデルを、標本列の順に1つずつ取り、そのモデルだけのbatchで、更新の回数ぶん、続けて更新する（乱数の消費の順は、モデル→更新）。
2. The run設定 shall 共有部を持たないモデル構造を、この学習の方式と、集約後の重みの再較正をしない方針との組でだけ、受け付ける。この学習の方式は、この構造でだけ、受け付ける。

### Requirement 3: サーバ

#### Acceptance Criteria

1. The サーバ shall 共有部が空のモデルを、集約・配布・統合できる（全パラメータを、モデルのIDごとに扱う）。
2. Where 共有部が空のとき, the クロス評価 shall 下りの通信量を、候補と評価先の組ごとに、評価するclientの数だけ、候補の全パラメータとして数える（旧と同じ。同じclientへの重複を省かない）。

### Requirement 4: 構成

#### Acceptance Criteria

1. The 構成 shall 掃引のコマンドで、`fedsda-independent-models`という名前で選べる。run設定は、提案構成と、モデル構造・学習の方式・再較正の方針だけが違う。
2. The 構成の全体run shall 小さい条件（sine2の7条件）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。mnist2の2条件で、全体runの全状態（モデルごとのパラメータ・勾配・optimizerの状態を含む）が一致する。
3. The 照合 shall 固有の経路を通る: 保有モデルがパラメータを共有せず、共有部の計算と、共有部のoptimizerが無い。複数のモデルを保有するclientがある。結果が、提案構成と違う。
4. The 提案構成と、これまでの9構成 shall 結果が変わらない。
