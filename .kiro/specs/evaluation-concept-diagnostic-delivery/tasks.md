# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 契約と受渡し
- [x] 1.1 実行の枠が、真の概念を、診断専用の引数としてclientへ渡す
  - 実行の枠のtestを先に直す: 観測用のclientが真の概念を記録する、受渡しのtest（clientと位置の対応）、契約の引数のtest、概念列の拒否のtest。
  - 契約、区間の進行（検査と受渡し）、全体runの実行を変更する。区間の進行を直接呼ぶ、ほかのtestと共用scriptを、概念列を渡すように直す。
  - 完了: 実行の枠のtest、clientのtest、依存境界のsuite、共用scriptが成功する。
  - _Boundary: RunClientOperations、run_stream_protocol_intervals、execute_stream_protocol_run_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2, 4.1, 4.3_

- [x] 2. 全体runの照合
- [x] 2.1 全体runの対照を、test専用の中継なしにし、判断への不干渉のtestを書く
  - 全体runの対照から、真の概念を渡す中継を外す。真の概念を渡さない中継で実行した全体runとの比較へ、不干渉のtestを書き換える。共用scriptの全体runの確認を、真の概念が渡っていることの確認にする。
  - 完了: 全体runの対照の全条件が、中継なしで、診断まで実旧と一致する。不干渉のtest、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: 全体runの対照test、共用のfresh process script_
  - _Requirements: 2.1, 3.1, 3.2, 4.2_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_
  - _Requirements: 3.1, 4.2, 4.3_

## Implementation Notes

- ユーザーの決定（2026-10-10）: 真の概念に依存する診断を、全体runで出す。実行の枠が、標本ごとの真の概念を、診断専用の引数として、clientへ渡す。
- 手順からの逸脱: task 1.1（契約と受渡し）と task 2.1（全体runの照合）を、1つのcommit（`1caa491`）にまとめた（1.1だけの状態では、古い形の不干渉のtestが失敗するため）。実行の枠のtestは、sourceより先に書き、受渡しのtestが失敗することを確かめてから、sourceを変えた。概念列の検査の関数`validate_evaluation_concept_traces_match_observed_streams`は、実装のときに足したので、命名表と設計へ、後から書いた（`7f3c878`。レビューの前）。
- 実装中の発見: (1)全体runは、test専用の中継なしで（実行の枠の本来の経路で）、9条件すべて、診断（真の概念別の診断証拠、割当概念の計数、標本ごとの記録、クラスタリングの真の概念の一致）まで、実旧と一致する。(2)概念列の型`ClientConceptTrace`は、0/1だけを受理する（SINE-2用。変えていない）。clientのtestの流れは、4つの概念の標本列を使うので、概念列は2値へ畳んだ値にしてある（受渡しの確認にだけ使う）。3つ以上の概念を持つdatasetを足すときに、この型の制約を見直す。(3)標本ごとの記録のうち、真の概念に依存する欄は、`observed_concept_id`と`true_concept_diagnostic_prediction_is_correct`の2つ（不干渉のtestを、欄ごとに比べる形へ強めたときに分かった）。
- 設計の判断: 概念列の検査の例外は、区間の進行が`TypeError`／`ValueError`をそのまま出す（`RunExecutionError`で包まない。どの段も始まっていないため）。契約（Protocol）の`evaluation_concept_id`は必須。`FedsdaRunClient.process_observed_sample`の同じ引数は、任意のまま（真の概念なしの単体の利用と、不干渉のtestのため）。
- 独立レビュー（1回、2026-10-10）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。実効値は出力されない）。GPT-6 Lunaは利用上限で使えない期間。session `e8d66e8a-f025-4e7d-9f13-6dd5a3346caf`、対象`80a10cd..7f3c878`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし。Minor 2件、任意 2件）。レビュー担当は、対象test（実行の枠と依存境界 3210 passed、client 53 passed、全体runの対照 14 passed）、Ruff（check・format）、共用scriptを独立に実行したと報告した。`spec_checks.py names`も実行したと報告したが、これは依頼文の許可コマンドの外である（結果は主担当の実測と同じ「未登録: なし」）。全pytest・Pyright・`pip check`は実行していない（基準どおり）。レビューの後、作業ツリーは空だった。
- 指摘の採否: (1)Minor: 本specの3文書が、single-run-executionの「要求1.4」を根拠にしていたが、正しくは要求2.4。また、single-run-executionの設計・命名表の、本specが置き換える記述を挙げていなかった→採用。3文書の番号を直し、設計の「過去の仕様との対応」へ、置き換える4つの記述を列挙した（single-run-executionの文書は書き換えていない）。(2)Minor: 不干渉のtestは、違ってよい項目を項目全体で許していたので、その項目の中の、概念以外の欄の違いを検出できなかった→採用。標本ごとの記録（概念の2つの欄を除く）、計数（割当概念の計数を除く）、診断証拠（globalの診断証拠）、保留中の標本（概念の欄を除く）が、一致することを確かめるassertを足した（`7245bb8`）。(3)任意: 候補の採用の前の検査が、割当概念の計数のkeyの有無を読む（真の概念の有無に依存する。正常な経路では成立しない検査）→設計の「『判断へ影響しない』の範囲」へ、例外として書いた。挙動は変えていない。(4)任意: spec.jsonの状態が古い→この記録で更新した。Blocker・Majorがないので、確認のレビューは行っていない（sourceは、レビューの後に変えていない）。
- 検証（Windows基準環境、`7245bb8`。この後は、specの文書とsteeringだけを変えた）: 全pytest 11305 passed / 3 skipped / 2 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。レビューの前の`7f3c878`でも、全pytestは 11305 passed / 3 skipped（その後の変更は、不干渉のtestのassertの追加だけ）。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分（federated_drift_experiment、tools、2つの旧回帰testとgolden）は空。`spec_checks.py names`は、変更した3つのsourceで「未登録: なし」。
- 変異テストは実行していない（既定）。
- 未検証・残る制約: 診断の保存と集計、真のドリフト位置を使う指標は、次のspec（指標の導出とgoldenとの照合）。Linuxでは実行していない。
