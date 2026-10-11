# Research & Design Decisions

- 旧の`SharedBackboneMLP`: 共有部（`SharedFeatureBackbone`）、出力層（`head`）。概念固有のパラメータ（`personalized_parameters`）は、出力層だけ。`ResidualAdapterMLP`は、構築の順が、共有部→adapter→出力層で、adapterの圧縮の層と展開の層の初期化が、乱数を消費する（展開の層は、その後で0にする）。
- 旧のclient（`SharedBackboneRestartingSoftRoutingFedSDAClient`）と、サーバ（`SharedBackboneFedSDANoCachedServer`）は、提案構成と同じ。違いは、モデルのクラスだけ。
- 新の計算量の計測は、概念固有部へ入力された標本数を、残差adapterの順伝播で数えていた。adapterを持たない構造では、数える場所がなくなるので、置き換えた部品でも数える。積和演算の数は、全結合層ごとに数えるので、この構造では、分類層の分だけになる。
- 最初の実装で、小さい条件の7つ全部が、実旧と一致した。
