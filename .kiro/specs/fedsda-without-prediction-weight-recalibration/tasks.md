# Implementation Plan

- [x] 1. 構成の追加
- [x] 1.1 再較正をしない方針と、構成の名前
  - 実旧との照合のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. 検証
- [x] 2.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 1.1_

## Implementation Notes

### 実装

- 手順: testの土台（実旧の全体runが、旧の構成名と設定の上書きを受け取る。小さい条件の照合を、構成を受け取る関数にする）を先に整え、提案構成の照合が変わらず通ることを確かめた。その後、構成のtestを先に書き、古いsourceで読込みに失敗することを確かめてから、sourceを直した。
- 設定の値を足したので、設定の検証のtest（許す値の一覧の期待）を1か所、直した。
- 照合の結果: 小さい条件の7つ全部で、構成の全体runの、指標の全部と離散列が、実旧（`SHARED_BACKBONE_ROUTING_RECALIBRATION = "none"`）と一致した。再較正の回数と、再生した標本数は、新旧とも0。同じ条件の提案構成では、再生した標本数が0でない条件があり、予測の結果が違う条件がある。

### 独立レビューと検証

- 独立レビュー（2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `874e2c67-fba0-4bb1-a880-08ef46ac3bfc`、対象`0c27442..f0eb8e1`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし）。レビュー担当は、構成の照合（9 passed）、掃引のtest（51 passed）、Ruffを独立に実行し、旧の該当箇所（サーバの`run_round`、clientの`recalibrate_routing_after_aggregation`）を読んで、省く範囲が新旧で同じことを確かめた。
- 指摘の採否: (1)任意: 共用の`tests/refactoring/fresh_process_smoke.py`は、再較正の結果を、Noneの確認なしで読む（提案構成だけを実行するので、今は動く）→変えていない。この構成をそのscriptで実行するときに直す。(2)Minor: 実旧の「再較正が0回」の検査は、予測重みとglobalの診断重みの計数だけを見ていて、真の概念別などの始め直しは数えていない→変えていない（旧のサーバは、`none`のとき、clientの再較正の関数そのものを呼ばないので、その中の始め直しも起きない。全体runの指標と離散列の一致でも確かめられている）。(3)任意: 設定の値が、sourceとtestに別々に書かれている→変えていない（run設定の等値の検査が、ずれを検出する）。
- 検証（Windows基準環境）: `f0eb8e1`で、全pytest（並列、`-n 8 --dist loadfile`）12478 passed / 4 skipped / 3 warnings、exit 0（8分6秒。独立レビューと同時に実行した）。goldenの3ケースと、提案構成の、小さい条件の照合を含む。JUnitはfailure 0、error 0。Ruff（check・format）、Pyright 0 errors。固定旧`748c3aa`からの差分は空。この後は、specの文書・steering・掃引の使い方の文書だけを変えた。
- WSLでは、実行していない（ユーザーの決定: specごとのWSLは省く）。変異テストは実行していない（既定）。

### 未検証・残る制約

- 照合は、sine2の小さい条件だけ（ユーザーの決定した基準）。ほかのdataset、本番の規模での、この構成の新旧の一致は、確かめていない。
- この構成のgoldenは、まだ無い（全構成のgoldenを、旧実装を外す前に、まとめて作る）。
