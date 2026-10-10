# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 計算とowner
- [x] 1.1 判定の計算と、判定の基準の型を、実旧の関数との対照つきで実装する
  - 実旧の、Wilsonの下限・クラス別の同時信頼下限・average linkageへ、同じ入力を与えて照合するtestを先に書く。
  - 完了: 3つの関数の対照（境界の入力を含む）と、拒否、判定の基準の型のtest、依存境界のsuiteが成功する。
  - _Boundary: model_clustering_calculations_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.3, 8.2_
- [x] 1.2 診断の記録のowner、グローバルモデルを外す操作、clientの割当概念の計数を足す
  - 完了: 記録のownerの単独のtest（順、検査、不正で不変）、外す操作のtest（パラメータと統計が外れ、次の正式IDと来歴は変わらない。持たないIDは拒否）、clientの操作のtest、依存境界のsuiteが成功する。
  - _Boundary: ModelClusteringRecordStore、GlobalModelRepository、FedsdaRunClient_
  - _Requirements: 3.1, 3.2, 4.4, 8.2_

- [x] 2. クラスタリングと統合、同期
- [x] 2.1 クラスタリングと統合、サーバの1ラウンドの同期を、実旧のサーバのラウンドとの対照つきで実装する
  - ラウンドの対照（旧は`run_round(round_index, clustering_enabled=…)`、新は同期の関数）を先に書く。クラスタリングの診断の記録を、実旧の来歴の観測と照合するhelperを書く。
  - クラスタリングと統合の関数、同期の関数、結果の型を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラスで、ラウンドごとに、全状態・診断の記録・乱数の一致を示し、要求6.2の経路を通ったことを確かめるtestが成功する。拒否と途中の失敗のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2_
  - _Boundary: cluster_and_consolidate_global_models、ModelConsolidation、synchronize_models_in_server_round、ServerRoundSynchronization_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1, 7.2, 7.3, 8.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptのサーバの代役を、同期の関数の呼出しにする
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに同期の関数を呼び、クラスタリングが1回以上行われ、記録が足されたことを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.1_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 8.1_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 6.1, 8.1, 8.2_

## Implementation Notes

- 手順からの逸脱: 判定の計算（1.1）は、対照testを先に書いた。クラスタリングと統合・同期（2.1）は、sourceを先に書き、その直後に対照testを書いた（testを先に書く手順からの逸脱）。最初の実行で全条件が通った（REDの実行は記録していない）。
- 実装中の発見: (1)対照の条件は、実旧だけで32条件（2値・多クラス、8つの標本列、評価標本の追加の件数2種類）を22ラウンド進めて選んだ（下書きのtest。commitしていない）。2モデルの統合、3モデルが1つへ集まる統合、統合しないクラスタリング、同じrunでの2回めのクラスタリングが、既定の条件で自然に起きた。(2)新の同期の関数は、旧のサーバの`run_round(round_index, clustering_enabled=<登録の前に、送信できるモデルを持つclientがいるか>)`と、12条件×16ラウンドで、全状態・診断の記録・乱数が一致した（統合の後のラウンドを含む）。(3)共用scriptでは、採番だけが行われてモデルが登録されないこと（LEGACY-016）があるので、グローバルモデルの数は「初期＋登録−吸収」以下、とした。共用scriptの2つの流れでは、クラスタリングは1回起きるが、統合（吸収）は起きていない。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。判定の数値と、グローバルモデルの書換えの順なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `3d9e7668-2ce9-40a2-ad55-0ffeb4170c19`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`98c9815..0f8ffd5`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし、Minor 3件、任意 3件）。対象test（234 passed）、依存境界・fresh process・グローバルモデルのownerのtest（3081 passed）、Ruff、共用script、`spec_checks.py names`を独立に実行した。全pytest・Pyright・`pip check`・固定旧実装の差分は確かめていない。
- 指摘の採否（Minorと任意だけなので、再レビューは受けていない）: (1)Minor: クロス評価したモデルが1つのとき、旧は何もしないが、新のクラスタリングの関数は観測を記録する→モデルが2つ未満の結果を拒否するようにした（同期の関数が、2つ以上のときだけ呼ぶ）。拒否のtestへ条件を足し、設計へ書いた。(2)Minor: 同期の関数が、判定の基準を、クロス評価の前に確かめていない→最初に、型と範囲を確かめるようにした（クラスタリングが無効のラウンドでも）。testを足した。(3)Minor: 対を判定する件数の下限（5件）の境界が、対照に含まれるか分からない→クロス評価の結果の1つの組の件数を、4と5へ書き換えて、実旧のクラスタリングの距離の有無と照合するtestを足した（5件ちょうどは距離を持ち、4件は持たない）。(4)任意: testの読みにくい式を直した。(5)任意: 共用scriptで統合が起きていない→記録だけ（統合は、実旧との対照が確かめている）。(6)任意: 真の概念の診断は、clientへ真の概念を渡したときだけ入る→既知（再開案内の「真の概念IDの配線」）。
- レビューの後の変更は、上の(1)〜(4)（sourceは、検査の追加だけ）。
- 検証（Windows基準環境、`07f7fea`。この後は、sourceとtestを変えていない）: 全pytest 11280 passed / 3 skipped / 2 warnings、exit 0（前spec 11043＋判定の計算・診断の記録・クラスタリングと統合の3ファイル 236＋グローバルモデルを外す操作 1）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtestはskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、変更した全sourceで「未登録: なし」。レビュー担当は全pytest・Pyright・`pip check`を実行していない（基準どおり）。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 実行の枠のサーバの操作の実体（`RunServerOperations`を満たすclass）は、まだない（同期の関数を呼ぶだけになる。共用scriptの代役が、その形）。判定の基準（閾値・評価の件数の下限・信頼水準）と、clientの上限の、設定の置き場所は仮（同期の関数の引数）。真の概念の一致の診断は、clientへ真の概念を渡したときだけ入る。診断の記録の保存と集計（旧の`clustering_oracle_diagnostic_summary`・`pair_diagnostic_summary`ほか）は未移植。最終構成でない判定・linkage・後処理は移植していない。WSLでは実行していない。
