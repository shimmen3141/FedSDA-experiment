# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 判定記録の保持
- [x] 1.1 判定記録を起きた順に保持するownerを作る
  - ownerのtest（順、読取りの不変、2つの型、拒否）を先に書く。
  - 完了: ownerのtestと、依存境界のsuiteが成功する。
  - _Boundary: CandidateValidationDecisionRecordStore_
  - _Requirements: 1.4, 1.5, 3.4_

- [x] 1.2 clientが、候補検証の確定と終端の回収で、判定記録を保持へ足す
  - clientの全状態の新旧照合へ、実旧の候補の判定の一覧との照合を足す（先に書いて、失敗を確かめる）。組立てのtest。
  - ownerの束、組立て、2つの操作を変更する。全体runの対照へ、判定の種類の経路と、不干渉の一致を足す。共用scriptへ確認を足す。
  - 完了: clientのtest、全体runの対照の全条件、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunClient、assemble_fedsda_run_client_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [x] 2. 検証
- [x] 2.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 1.2_
  - _Requirements: 2.2, 3.1, 3.3, 3.4_

## Implementation Notes

- 調査の結果（requirements.mdの冒頭に詳細）: (1)Linuxの再現性——旧実装の最終構成の3ケースを、WSL Ubuntu（Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu、1 thread）で、別のprocessで2回実行し、結果（33指標、31の離散列のhash、経路の件数）が完全に一致した。Windows用のgoldenとは、sine2（指標10、離散列15）とsea2（指標1、離散列2）で違い、mnist2は一致した。調査用のscriptと2回の出力は、Git管理外（`venv/refactoring-tests/linux_repro_dump.py`、`linux-repro-run1.json`・`linux-repro-run2.json`）。(2)goldenの条件（sine2）で、新の全体runは、実旧の全体runと、全状態で一致した（下書きのtest。commitしていない）。(3)(4b)を、判定記録の保持（本spec）、結果の導出と照合、Linux用のgolden、の3つに分けた。
- 手順からの逸脱: ownerのtest（1.1）は、sourceより先に書いたが、失敗の実行は記録していない（moduleがないだけの失敗）。clientの側（1.2）は、新旧照合のhelperへ照合を足して、失敗（ownerの束にfieldがない）を確かめてから、sourceを変えた。
- 実装中の発見: 全体runの対照の9条件が通った判定の種類は、採用、区間の判定での棄却（前半、後半、両方）、現行モデルの維持、終端の回収の6つ（経路の網羅のtestの必須集合へ足した）。別の保有モデルの再利用（実旧の理由`alternative_reference_refit`）は、全体runの対照でも、clientの軌跡の対照でも通っていない。
- 設計の判断: clientの操作が、下位の処理が成功して返った後に、戻り値の判定記録を、保持へ足す（下位の関数の引数と、そのtestを変えない）。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。GPT-6 Lunaは利用上限で使えない期間。session `0c9968c9-50b3-4b3d-a58a-90dfee662671`、対象`2bfb686..36f5c5f`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Major・Minorなし。任意 4件）。レビュー担当は、対象test（owner・client・依存境界 3104 passed、全体runの対照 14 passed）、Ruff（check・format）、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない（基準どおり）。レビューの後、作業ツリーは空だった。
- 指摘の採否（すべて任意。sourceとtestは変えていない）: (1)判定の評価は、採否（Pythonのfloatで比較）と、理由（float32の差で比較）で丸めが違い、境界の近くで、採否が真で理由が「不合格」になりうる。旧と同じ挙動で、[LEGACY-001](../../../docs/research/implementation-findings/legacy-001-candidate-decision-reason-rounding.md)に記録済み。保持した判定記録を使う導出（次のspec）は、採否と理由を、それぞれ記録のとおりに写す（理由から採否を逆算しない）→ここへ記録。(2)失敗時の既知の制約（適応記録だけが残る）は、標本の処理の例外が握りつぶされないことに依存する。レビュー担当が、実行の枠で確認した→記録だけ。(3)別の保有モデルの再利用の判定記録は、保持した一覧としては照合されていない→下の「未検証」へ記録。(4)clientの状態の読取り（testの読取りの関数）は、判定記録の保持を含まない。拒否のtestは、保持が変わらないことを直接には確かめていない（追加は、成功の後にしか起きない）→記録だけ。
- 検証（Windows基準環境、`36f5c5f`。この後は、specの文書とsteeringだけを変えた）: 全pytest 11313 passed / 3 skipped / 2 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分（federated_drift_experiment、tools、2つの旧回帰testとgolden）は空。`spec_checks.py names`は、変更した2つのsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 別の保有モデルの再利用（`alternative_reference_refit`）の判定記録は、保持した一覧としては、実旧と照合していない（中身は、判定の評価の部品のtestが照合している。旧との対応は、レビュー担当がコードで確認した）。次のspecで、goldenの条件の全体runが、この種類を通るかを確かめる。標本1件の処理が、確定の後の段で失敗したときは、判定記録が足されない（設計の「失敗時の扱い」）。Linuxでは、このspecのtestを実行していない。
