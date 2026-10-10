# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 計算の部品
- [x] 1.1 変更位置の抽出、対応づけ、精度、定常精度、検出の指標を計算する
  - 実旧の関数をoracleにしたtest（多数の入力と境界、拒否）を先に書く。
  - 完了: 部品のtestと、依存境界のsuiteが成功する。
  - _Boundary: run_metric_calculations_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 4.5, 5.2_

- [x] 2. 全体runからの導出と照合
- [x] 2.1 全体runの結果と参加者から指標を導出し、実旧・goldenと照合する
  - goldenの条件の照合（実旧の実行と保存結果、Windows用のgolden）、小さい条件の照合、状態を変えないこと、拒否のtestを先に書く。31の離散列は、testが、新の記録から作る。
  - 導出を実装する。共用scriptへ、導出を足す。
  - 完了: 導出のtest（Windowsでは、goldenの照合を含む）、依存境界のsuite、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_metric_derivation_
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_
  - _Requirements: 4.1, 4.3, 5.1, 5.2_

## Implementation Notes

- ユーザーの決定（2026-10-10）: 計算量の指標（`compute_*`の7項目）は必要。旧の計数の検査（漏れ・重複がないか）を含めて、後の1つのspecで扱う。本specでは照合しない。順序は、本spec→計算量→Linux用のgolden。
- 手順からの逸脱: 2つのtest fileは、sourceより先に書いた。失敗の実行は記録していない（moduleがないだけの失敗）。最初の実行で、指標と列の照合は通った。testの側で直したのは、goldenの条件が通る判定の理由の集合（実行して確かめて書いた）と、小さい条件の選び方（実旧だけの下書きで、条件ごとの判定の理由を調べた。下書きは削除）。
- 実装中の発見: (1)goldenの条件（sine2）で、導出した26指標は、同じprocessの中の実旧の`run_random_drift_experiment`の結果と完全に一致し、新の記録から作った31の離散列は、実旧が保存した配列と、形・型・値で一致した。Windows用のgolden（sine2）とも一致した（26指標は1e-9、31列は形とSHA-256）。(2)goldenの条件の全体runは、別の保有モデルの再利用の判定（`alternative_reference_refit`）を通る。前のspec（candidate-validation-decision-record-retention）で未検証だったこの種類について、候補の判定の列（提案位置、採否、理由、確定位置）は、実旧の保存配列と一致した。(3)小さい7条件でも、31列は、実旧の保存処理（`_save_raw_run`を、実旧の全体runの結果へ直接呼ぶ）の配列と一致した（未完了の候補検証の判定、区間の判定での棄却、統合、サーバの付け替えを含む）。
- 設計の判断: 指標だけをsourceに置き、旧の保存形式の31列は、testが新の記録から作る。指標の名前は新実装の語で付け、旧の名前との対応は、testの`derive_legacy_metric_values`が持つ。goldenの条件の新の設定は、testの`make_golden_condition_settings`が、旧の設定の値から写す（写していない値は、research.mdの「新の設定へ写していない値」）。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。GPT-6 Lunaは利用上限で使えない期間。session `dede8d03-1e0b-4f98-af34-f3e66d746502`、対象`179aa19..9aec02c`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし。Minor 3件、任意 3件）。レビュー担当は、対象test（計算の部品と依存境界 3190 passed、導出 17 passed・skipなし——Windows用のgoldenとの照合を含む）、Ruff（check・format）、共用scriptを独立に実行した。全pytest・Pyright・`pip check`・`spec_checks.py`は実行していない（基準どおり）。レビューの後、作業ツリーは空だった。
- 指摘の採否: (1)Minor: testが31列を作る処理のうち、未完了の候補検証の判定の分岐は、goldenの条件では通らず、小さい条件は列を比べていなかった→採用。小さい条件でも、31列を、実旧の保存処理の配列と照合するようにし、条件の網羅のtestへ、未完了の判定・区間の判定での棄却・統合・サーバの付け替えを足した（`69c6762`）。(2)Minor: 「31列がすべて作れる」の根拠が、goldenの条件だけ→(1)で解消。設計のtestの節へ追記した。(3)Minor: goldenの条件のうち、新の設定へ写していない値（新実装では固定）の一覧がない→採用。research.mdへ書いた。(4)任意: 混合予測を行った標本の数は、常時有効の方針のときだけ、予測した標本の総数と同じ→sourceに注釈がある。方針を変えるときに見直す（ここへ記録）。(5)任意: 状態を変えないことのtestは、NumPyの乱数を比べていない（導出は使わない）。goldenの`definition`と`_env`は、このtestでは確かめない（旧の回帰testが確かめる）→記録だけ。(6)任意: 命名は妥当→対応なし。Blocker・Majorがないので、確認のレビューは行っていない（sourceは、レビューの後に変えていない。変えたのは、testと文書）。
- 検証（Windows基準環境、`69c6762`。この後は、specの文書とsteeringだけを変えた）: 全pytest 11481 passed / 3 skipped / 2 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。導出のtestは 21 passed、skipなし（Windows用のgoldenとの照合を含む）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分（federated_drift_experiment、tools、2つの旧回帰testとgolden）は空。`spec_checks.py names`は、新しい2つのsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 計算量の7項目は、導出も照合もしていない（次のspec）。goldenが比べない指標（警報の適合率、誤検出の内訳、遅延、変化点の誤差、候補の判定の平均、診断の集計ほか）と、結果の保存は、範囲外。照合したdatasetはsine2だけ（sea2・mnist2は、datasetの移植が要る）。Linuxでは、このspecのtestを実行していない（Windows用のgoldenとの照合は、Windows以外ではskipする。Linux用のgoldenは、後のspec）。別の保有モデルの再利用の判定記録の、平均損失ほかの項目は、保持した一覧としては、実旧と照合していない（列の4項目だけ）。
