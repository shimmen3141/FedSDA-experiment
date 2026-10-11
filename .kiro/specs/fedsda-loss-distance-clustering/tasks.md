# Implementation Plan

- [x] 1. 構成の追加
- [x] 1.1 損失の増加量での判定と、構成の名前
  - 実旧との照合のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 2. 検証
- [x] 2.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 1.1_

## Implementation Notes

### 実装

- 手順: 構成の照合のtestと、判定の基準の新しいfieldのtestを先に足し、古いsourceで読込みに失敗することを確かめてから、sourceを直した。specの文書は、実装の後に書いた（手順からの逸脱。前の構成と同じ理由）。
- 照合の結果: 小さい条件の7つ全部で、構成の全体runの、指標の全部と離散列が、実旧（`FEDSDA_CLUSTERING_DECISION = "distance"`、linkageはaverage）と、最初の実装で一致した。対の判定の値は、全部、損失の距離と同じで、同じクラスタになった対と、ならなかった対の、両方がある。同じ条件の提案構成では、判定の値が違う対があり、モデル別の診断の指標が違う条件がある。
- 設計の文書の、クロス評価の記述を、1回、直した（最初は、確かめていない前提——実旧の全体runが、診断を集める設定——を書いていた。コードを読んで、旧のFedSDAのサーバは常に診断を集めること、クラス別の集計は提案構成の判定のときだけ作ることを確かめ、新だけが表`client_cross_evaluation_class`を持つ違いを書いた）。

### 独立レビューと検証

- 独立レビュー（2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `25a2f9e2-86f1-40da-a1c5-3c056bd33597`、対象`08721c9..44f2c37`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし）。レビュー担当は、構成の照合（25 passed）、クラスタリングのtest（202 passed）、Ruffを独立に実行し、旧の該当箇所（`perform_hierarchical_clustering`、`_clustering_cutoff`、`cluster_models`）を読んで、判定の値・閾値・linkage・標本数が足りない対の扱いが、新旧で同じことを確かめた。
- 指摘の採否: (1)Minor: 「損失の距離」と「損失の増加量」の語が混ざっている→変えていない（既存の名前`loss_increase_distance`が、両方の語を含む。記録の列の名前も同じ）。(2)Minor: 設定の不一致の拒否を、片方の向きしか試していない→採用。両方の向きを試す（`8ab3691`）。(3)任意: 新しい分岐の単体test→足していない（ユーザーの決定した基準は、全体runの照合）。(4)任意: 整形の確認→主担当が`ruff format --check`を実行した（成功）。
- 検証（Windows基準環境）: `44f2c37`で、全pytest（並列、`-n 8 --dist loadfile`）12495 passed / 4 skipped / 3 warnings、exit 0（独立レビューと同時に実行した）。goldenの3ケースと、提案構成の小さい条件の照合を含む。JUnitはfailure 0、error 0。Ruff（check・format）、Pyright 0 errors。固定旧`748c3aa`からの差分は空。この後は、testを1件直し（`8ab3691`。そのtestを実行して成功）、specの文書・steering・掃引の使い方の文書を変えた。`8ab3691`での全pytestは、実行していない。
- WSLでは、実行していない（ユーザーの決定: specごとのWSLは省く）。変異テストは実行していない（既定）。

### 未検証・残る制約

- 照合は、sine2の小さい条件だけ（ユーザーの決定した基準）。ほかのdataset、本番の規模での、この構成の新旧の一致は、確かめていない。
- この構成では、記録の表`client_cross_evaluation_class`を、新だけが持つ（設計の「クロス評価は変えない」）。
- この構成のgoldenは、まだ無い（全構成のgoldenを、旧実装を外す前に、まとめて作る）。
