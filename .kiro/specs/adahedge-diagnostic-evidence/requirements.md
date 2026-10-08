# AdaHedge診断の証拠状態 — 要求 revision2

研究者が最終構成の保存診断を維持できるよう、診断用AdaHedgeの証拠状態と操作をモデル推論・学習・通知から独立して検証できるようにする。予測を決めるFixed-Shareとは別の状態である。

## 境界

対象は整数モデルIDを使う単一の診断証拠状態。入力済みのモデル別有界損失から診断用重みと証拠を更新し、明示された概念操作で再始動する。

学習帰属変更通知の発火と複数状態への配布、集約後の再較正と損失再生、context/meta方式、診断値の集計・保存、モデル推論、真の概念による状態選択、検出episode、client進行、新全体runは対象外。モデルID集合と損失は呼出側から受け取る。後続の通知側が再始動の時機を決める。

## 1. 状態と診断用重み

- 1.1 The AdaHedge Diagnostic Evidence shall 累積損失、mixability gap、集合変更回数、概念操作後の再始動回数を独立して保持し、空の証拠とゼロの計数で開始する。
- 1.2 When 非空のモデルID集合から診断用重みを取得するとき, the AdaHedge Diagnostic Evidence shall ID昇順で旧AdaHedgeと同じ重みを返す。初回またはID集合の変更では損失とgapをゼロから開始し、変更前にモデルIDを保持していた場合だけ集合変更回数を増やす。損失がすべてゼロで、観測後更新を一度も行っていない場合も保持済みとして数える。
- 1.3 When 同じID集合を異なる列挙順で受け取るとき, the AdaHedge Diagnostic Evidence shall 証拠を消去せず、同じ重みを返す。単一モデルでは重み1、証拠がゼロなら一様、無限学習率では累積損失最小の同率モデルへ均等に重みを配る。

## 2. 観測後更新と再始動

- 2.1 When モデル別観測損失と診断用重みを受け取るとき, the AdaHedge Diagnostic Evidence shall 有限損失を0から1へ制限し、期待損失、mix loss、非負のgap増分、累積損失を旧AdaHedgeと同じ演算順で更新する。ID集合の変更規則は重み取得時と同じとする。
- 2.2 When 概念操作後の明示再始動を受け取るとき, the AdaHedge Diagnostic Evidence shall 損失とgapだけを消去し、再始動回数を1増やし、集合変更回数を保持する。空の状態での再始動も1回として数える。
- 2.3 The AdaHedge Diagnostic Evidence shall 通知を自動発火せず、Fixed-Share重み、学習帰属、モデル値、候補検証、監視、乱数を変更しない。

## 3. 入力拒否と観測値

- 3.1 If IDが整数でない、真偽値である、集合が空、または列挙したIDが重複しているとき, the AdaHedge Diagnostic Evidence shall 状態の変更前に拒否する。負の整数IDは一時モデルとして受理する。
- 3.2 If 損失と重みのID集合が異なる、損失・重みがbuiltin int/float以外（boolを含む）または有限floatへ変換できない、重みが0から1の範囲外、または重みの総和と1の絶対差が1e-12を超えるとき, the AdaHedge Diagnostic Evidence shall 集合同期を含む状態変更の前に拒否する。有限損失の範囲外は拒否せず制限する。受理した重みを再正規化しない。
- 3.3 The AdaHedge Diagnostic Evidence shall 呼出元の入力を変更せず、返した重みや累積損失の観測値を変更されても所有状態へ影響を受けない。

## 4. 基準と検証範囲

- 4.1 When 同じ正常入力列で実行するとき, the AdaHedge Diagnostic Evidence shall 初回、有限・無限学習率、同率、単一モデル、負ID、ID集合変更、損失制限、繰返し再始動について実旧AdaHedgeの重み・損失・gap・対応計数と一致する。
- 4.2 The change shall 固定旧実装とgoldenを変更せず、旧実装へ依存しない新実装を用意する。部品の一致を保存診断全体や新全体runの一致として扱わず、Windows基準の全回帰が済むまでTask 3と最終GOを完了にしない。
