# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 計数
- [x] 1.1 集約と統合が、パラメータの積和演算の数を、結果に含める
  - 既存の、実旧との対照へ、計数の照合（集約＝通信量の上りの値の数、統合・距離＝記録と層の形からの計算）を先に足す。
  - 完了: 集約と統合のtestが成功する。
  - _Boundary: ClientModelAggregation、ModelConsolidation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2_

- [x] 1.2 clientが、標本ごとの保有モデル数を記録する
  - ownerのtestと、clientの全状態の新旧照合への照合を先に書く。
  - 完了: ownerのtest、clientのtest、依存境界のsuiteが成功する。
  - _Boundary: HeldModelCountRecordStore、FedsdaRunClient_
  - _Requirements: 2.1, 2.2, 2.3, 5.1_

- [x] 1.3 計測つきの全体runが、ラウンドごとのモデルの計算を返す
  - 合計の一致、同期の値が学習を含まないことのtestを先に書く。
  - 完了: 計測つきの全体runのtestが成功する。
  - _Depends: なし_
  - _Boundary: fedsda_measured_run_execution_
  - _Requirements: 3.1, 3.2, 3.4_

- [x] 2. まとめと指標
- [x] 2.1 計算量のまとめを計算する
  - 手計算との照合、NaN、拒否のtestを先に書く。
  - 完了: まとめのtestと、依存境界のsuiteが成功する。
  - _Boundary: computation_cost_summary_
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 2.2 指標の導出へ足し、goldenの条件で、実旧と照合する
  - 指標の導出のtestへ、サーバの計数・保有モデル数・ラウンドごとの値・まとめの照合を先に足す。共用scriptへ足す。
  - 完了: 指標の導出のtest（goldenの33指標の照合を含む）、共用script、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2, 1.3, 2.1_
  - _Boundary: fedsda_run_metric_derivation_
  - _Requirements: 3.3, 4.4, 5.1, 5.3, 5.4_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 1.4, 3.4, 5.4_

## Implementation Notes

- ユーザーの決定（2026-10-10）: 論文で言いたいことは、(a)clientの計算が少ない、(b)clientが保有する1モデルあたりの計算量（保有モデル数は常に変わる）、(c)モデル数が増えても、計算が増えにくい。サーバの計算にも触れる。単位は、積和演算。
- 手順からの逸脱: 保有モデル数（1.2）は、clientの全状態の新旧照合へ照合を足し、失敗を確かめてからsourceを変えた。集約・統合の計数（1.1）、ラウンドごとの計算（1.3）、計算量のまとめ（2.1）、指標の導出（2.2）は、testとsourceを同時に適用した（testを先に書いて失敗を確かめる手順からの逸脱）。
- 実装中の発見: (1)診断のパラメータ距離は、診断の観測に残る対だけでなく、クラスタリングの対象のモデルの、全部の対で計算されている（観測に残らない対の距離は、使われない。挙動は変えていない。計数は、全部の対を数える）。(2)goldenの条件（sine2）で、ラウンドごとのモデルの計算（30ラウンド。ローカルの処理と、同期の別）が、実旧のラウンドごとの計数（用途別）の、全clientの合計と一致した。終端の処理の分（旧の累計−ラウンドの合計）、標本ごとの保有モデル数の、ラウンドごと・clientごとの合計（旧の予測の計数）も、一致した。(3)goldenの条件の値（参考。sine2、client 3、標本1500件）: clientの順伝播の積和演算は約5.6×10^8（共有部 約3.7×10^8、概念固有部 約1.9×10^8）、逆伝播の見積りは約1.0×10^9、サーバは約2.1×10^5（集約 202545、統合 5211）で、clientの順伝播の約0.04%。平均の保有モデル数は約2.52。標本あたりの順伝播は約1.25×10^5、共有部の標本あたりは約8.3×10^4、概念固有部の保有モデル×標本あたりは約1.7×10^4。
- 設計の判断: サーバの計数は、集約と統合の結果へfieldとして足した（ラウンドごとの値が、そのまま残る）。保有モデル数は、標本ごとの列として、clientが記録する。ラウンドごとのモデルの計算は、計測つきの全体runが、サーバの操作を包んで、2つの時点の計数を控える。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。GPT-6 Lunaは利用上限で使えない期間。session `2a038001-740e-49c3-a256-4fbb81f5c6a8`、対象`8b9cd77..5e5e113`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Major・Minorなし。任意 3件）。レビュー担当は、対象test（owner・まとめ・計測つきの全体run・client・依存境界 3120 passed、全体runの対照と指標の導出 36 passed・skipなし、集約と統合 48 passed）、Ruff（check・format）、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない（基準どおり）。レビューの後、作業ツリーは空だった。
- 指摘の採否（すべて任意）: (1)「保有モデル×標本あたり」の値に、保有モデル数に依らない共有部の計算が含まれ、保有モデルが増えると機械的に小さくなる→採用。共有部の標本あたりの値と、概念固有部だけの保有モデル×標本あたりの値を、まとめへ足した（主張(b)には、概念固有部だけの値を使う）。(2)設計の集約の説明が、実装（上りの一覧から数える）と違う→設計の文面を直した。(3)サーバの計数のhelperの戻り値を、全体runの対照が使っておらず、統合と距離を通る条件が分からない→採用。全体runの対照の経路の網羅へ、統合の加重平均と、パラメータ距離を足した（`2daeb58`）。Blocker・Majorがないので、確認のレビューは行っていない（反映は、まとめのfieldの追加と、testと文書）。
- 検証（Windows基準環境、`2daeb58`。この後は、specの文書とsteeringだけを変えた）: 全pytest（並列、`-n 8 --dist loadfile`）11564 passed / 3 skipped / 3 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。レビューの前の`5e5e113`でも、全pytestは 11564 passed / 3 skipped。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、変更・新規のsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: サーバの計数には、旧に対応する計数がない（集約は通信量で、統合と距離は、記録と層の形からの計算で確かめた）。割り算、写し、損失統計の平均は、数えていない。検出器の計算（評価した候補×賭け率の数）は、積和演算へ換算していない（別枠）。回帰（モデル数と計算量の傾き）、図、保存は、ない（ラウンドごとの計算量と、標本ごとの保有モデル数は、得られる）。FedDriftの計測は、手法の移植の後。照合したdatasetはsine2だけ。Linuxでは、このspecのtestを実行していない。
