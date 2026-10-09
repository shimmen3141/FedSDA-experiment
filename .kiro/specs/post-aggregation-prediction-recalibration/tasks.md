# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 診断証拠の操作
- [x] 1.1 単一の診断証拠へ、集約後の再生・再始動と、3つの計数を足す
  - 実旧のAdaHedgeと、通常の観測・概念操作による再始動・集約後の再生・集約後の再始動を混ぜた列で照合するtestを先に書く。照合のhelperへ、3つの計数を足す。
  - 完了: 足した操作の対照（空の列、途中でモデル集合が変わる列、モデルが1つの列を含む）と、不正な列で証拠と計数が変わらないことのtest、診断証拠の既存のtestが成功する。
  - _Boundary: AdaHedgeDiagnosticEvidence_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.3, 5.2_

- [x] 2. 再較正
- [x] 2.1 clientの再較正を、実旧のサーバのラウンドとの対照つきで実装する
  - 旧の設定の差し替えへ、再較正の方式を足す（既定は`none`のまま）。配布の対照のoracleで、旧は`run_round(round_index, clustering_enabled=False)`、新は登録→集約→配布→全clientの再較正、を行うラウンドの対照を先に書く。
  - 再較正の関数（検査、損失の列、globalの診断証拠の再生、真の概念別の診断証拠の再始動、Fixed-Shareの再生）と、clientの操作を実装する。依存の許可集合を登録する。
  - 完了: ラウンドの対照が、2値・多クラス、学習率の2つの設定が違う条件で、ラウンドごとに全状態と乱数の一致を示し、要求4.2の経路を通ったことを確かめるtestが成功する。clientの再較正だけの対照、拒否のtest、再較正が変えない状態のtest、依存境界のsuiteが成功する。
  - _Depends: 1.1_
  - _Boundary: recalibrate_prediction_state_after_aggregation、PostAggregationPredictionRecalibration、FedsdaRunClient.recalibrate_prediction_state_after_aggregation_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 5.1, 5.3, 5.4, 6.2_

- [x] 3. 新実装だけの確認
- [x] 3.1 共用のfresh process scriptで、配布の後に全clientの再較正を行う
  - 完了: 共用scriptが、旧実装とtestのmoduleを読み込まずに、ラウンドごとに配布の後で全clientの再較正を行い、空でない列での再較正が1回以上あることを確かめて成功する。別processで実行するtest、Ruff、Pyrightが成功する。
  - _Depends: 2.1_
  - _Boundary: 共用のfresh process script_
  - _Requirements: 6.1_

- [x] 4. 検証
- [x] 4.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 3.1_
  - _Requirements: 4.1, 6.1, 6.2_

## Implementation Notes

- 手順からの逸脱: task 2.1は、対照testとsourceを続けて書き、最初の実行で全条件が通った（REDの実行は記録していない）。
- 実装中の発見: 再較正を足した新のラウンド（登録→集約→配布→全clientの再較正）は、旧のサーバの`run_round(round_index, clustering_enabled=False)`（方式`fifo_replay`）と、9条件×13ラウンドで、全状態と乱数が一致した。共有部の特徴を、旧は1回、新はモデルごとに計算するが、値は同じだった。既存の対照（配布のspecまで）は、旧の設定の既定（`none`）のままで、変えていない。
- 既存のtestの変更: 診断証拠の照合のhelper（`get_adahedge_evidence_snapshot`・`assert_adahedge_matches_legacy`）へ3つの計数を足したので、固定の4要素の期待値を書いていた3箇所を7要素へ直した。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。複数の状態の更新順と、旧のルータとの数値の対応なので、難度からHaikuを選んだ（GPT-6 Lunaは利用上限で使えない期間でもある）。session `60214683-cb97-4750-8036-58c60b2e1b7f`、`is_error=false`、`modelUsage`は`claude-haiku-5-5`だけ。対象は`1c2966c..9b96a00`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし、Minor 1件、任意 2件）。対象test（109 passed）、依存境界とfresh process（3044 passed）、Ruff、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない。`get_true_concept_diagnostic_evidence`ほか2つの既存の関数の中身は読んでいないと申告している。レビュー担当は、旧`clients/fedsda.py`の写しを、worktreeの外（自分のsessionの一時フォルダ）に1つ作った（依頼文の「写しも作らない」に反する）。元と同一であることを確かめて、主担当が削除した。worktreeの`git status`は、前後とも空。
- 指摘の採否（文書だけの修正なので、再レビューは受けていない）: (1)Minor: 診断証拠の集約後の再生は、行ごとにモデル集合が違う列を受け入れ、途中でモデル集合の変化の回数が増える。要求2.4（「変えない」）・5.2（「対応でない」）の文面と食い違う→要求の文面を直した（sourceは変えない）。旧の`AdaHedgeRouter.replay`も、移植済みのFixed-Shareの再生も、行ごとのモデル集合の違いを受け入れて、通常の観測と同じ規則で同期する。診断証拠の対照testが、途中でモデル集合が変わる列を、実旧と照合している。再較正が作る列は、全行が同じモデル集合である。要求5.2の「対応でない」は、Mappingでない、の意味だったので、そう書いた。(2)任意: 設計の「旧と違う点」へ、移植しない旧の方式の名前を書いた。(3)任意: 損失の計算の失敗のtestが、torchの乱数の状態の先頭の要素だけを比べている→不採用（読取りの`torch_random_state`は、要素が1つのtupleである）。
- 検証（Windows基準環境、`9b96a00`。レビューの後は、sourceとtestを変えていない）: 全pytest 10991 passed / 3 skipped / 2 warnings、exit 0（前spec 10948＋診断証拠 30＋再較正 13）。JUnitは10994 testcase、failure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`は成功。経路の網羅のtestはskipされていない（skip 3件は以前からのもの）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、変更した全sourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 全clientへ順に呼ぶ関数は置いていない（呼出し側が、clientの操作を順に呼ぶ）。clientの操作は、設定を読まずに、常にこの方式で再較正する（設定の選択肢が1つのため）。診断証拠の3つの計数の保存は、診断の保存を決めるspecで扱う。計算量の記録は未移植。WSLでは実行していない。
