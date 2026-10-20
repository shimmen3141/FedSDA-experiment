# Implementation Plan

- [x] 1. 変更の前の所要時間を測る
  - 研究の比較などを同時に動かさずに、全pytestを今の配り方で実行し、全体の時間と、goldenのtestのファイルの時間（JUnitの合計）を記録する。
  - 完了条件: 時間と合否の件数がImplementation Notesにある。
  - _Requirements: 4.2_

- [x] 2. 実行の使い回しを足す
  - 使い回しのtestを先に書き、REDを確かめる。`capture_case_and_count_divergence_events`を分け、testの側だけがcacheを使う。
  - 完了条件: 使い回しのtest（同じ設定では1回、違う設定では2回実行する）と、goldenのtestの全部が通る。
  - _Requirements: 3.1, 3.2_

- [x] 3. 版の差の説明のtestを分け直し、配り方を足す
  - 指定するケースが経路を通ることを確かめてから、3つのtestをケースごとに分け直す。配り方のtestを先に書き、`tests/conftest.py`へgoldenのtestのファイルの組を足す。
  - 完了条件: 指定した3つのケースが経路を通ることの確認の結果（ケースごとの、`sample_prediction`の差の有無、クロス評価の有無と下りの増減、統合の数）と、配り方のtestの結果（収集したケースの名前を持つtestの数と、組ごとの数、ファイルの組へ戻ったtestが0であること）がImplementation Notesにあり、配り方のtestと、goldenのtestの全部が、並列の全pytestで通る。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_

- [x] 4. 変更の後の所要時間を測り、文書を合わせる
  - 1と同じ条件で全pytestを実行し、時間と合否を比べる。共通引継ぎ手順の全pytestの所要時間の記述を更新する。
  - 完了条件: 前後の時間と合否がImplementation Notesにあり、文書が新しい時間を書く。
  - _Requirements: 4.1, 4.2_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruffの結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 下調べ（2026-10-20）: 基準版9の全pytestのJUnit（`../../venv/refactoring-tests/consolidation-full.xml`）のファイル別の合計は、構成の照合50.8分（3組に分けて配っている）、goldenのtest 45.4分（306件、1組）、次は13.5分。「ケース×版」は173組で、run設定が違うのは134組（FedDrift・FedAvgの4ケースは各9版→1、FedSDAの大きい条件のケースは、版6と版7で同じ設定のものなど）。script: `C:\\Users\\yshin\\AppData\\Local\\Temp\\claude\\c--Users-yshin-Research-FedSDA-experiment\\799b882d-c886-45dd-a18e-a06f3e20f4b4\\scratchpad\\golden_reuse_count.py`。
- 仕様レビュー1回目（Luna high、session `01a1590b-2a10-7822-8d12-6228f60df311`）: Major 1件（使い回しのtestが、違う設定で別に実行することを確かめない）、Minor 2件（配り方のtestが、ケースの名前を持つ全部のtestがケースの組に入ることを確かめない。特定のケースの確認の結果の記録先）。全部反映。2回目（session `01a1590d-095d-70f2-b626-332e51cd2ce0`、再確認）: Majorは解消、Minor 1件（配り方のtestの結果も記録する）を反映した。Minorのみのため再レビューなし。
- Task 1（2026-10-20 20:21〜20:35、研究の比較などを同時に動かさない、`-n 12 --dist loadfile`、commit `ae32e8b`）: 14580 passed / 8 skipped、exit 0、874.7秒（14.6分）。JUnit（`../../venv/refactoring-tests/golden-parallel-before.xml`）のファイル別の合計は、構成の照合14.9分（3組）、goldenのtest 14.0分（1組）、次は4.1分。下調べの45.4分・2761秒は、研究の比較と同時の実行による値だった。同時の実行が無くても、goldenのtestのファイル（1組）が全体の下限であることは変わらない。
- Task 2（2026-10-20）: 手順の逸脱として、使い回しの実装（`_execute_case_run`・`_assemble_case_capture`・`capture_case_reusing_runs`）を、testより先に書いた（REDは確かめていない）。使い回しのtest（`test_runs_are_reused_only_for_identical_full_run_settings`）は、実行の関数を包んで数え、`feddrift-drift-tail`の版1と版2（完全なrun設定が同じ）で1回だけ実行すること、経路の記録が版ごと（版2だけが`unregistered_adopted_model_count`を持つ）で、ほかの結果が同じこと、毎回実行する関数と同じ結果であること、版3だけseedを1つ変えた設定では、版2と版3で別に実行することを確かめる。初回の実行では、seedを変えた設定が`feddrift-drift-tail`の必須の経路（検出が1以上）を通らずに失敗したので、その部分だけ必須の経路の確認を外した（回数だけを確かめる）。直した後に通った。生成の手順（`capture_case`）は、毎回実行する関数のまま。
- Task 3（2026-10-20）: 指定したケースの確認は、分け直したtestの中の確認（指定したケースで、`sample_prediction`の表が版5と違う／クロス評価が起き、下りが減る／版8で`merged_model_count`が1以上）として、goldenのファイルの並列の実行（`-n 12 --dist loadfile`、JUnit `../../venv/refactoring-tests/golden-split-file.xml`）で、`fedsda-sine2-events`（版5→6）と`fedsda-sine2-adoption-without-validation`（版7→8、版8→9）が通った。手順の逸脱として、分け直す前の別の確認は行わず、分け直したtestの確認で代えた（指定が外れると、そのtestが失敗する）。配り方: ケースの名前は21。ケースの名前を持つtestは315件で、組ごとの件数は、sine2-events 17、mnist2-decisions 17、sine2-reference-reuse 15、sine2-alternative-reference-reuse 11、sine2-adoption-without-validation 5、mnist2-adoption-without-validation 5、others-0 131、others-1 114。ファイルの組は51件（ケースの名前を持たないtest）で、ケースの名前を持つのにファイルの組へ戻ったtestは0（配り方のtestで確かめる）。idの読み取りは、設計の「最も長く一致する名前」ではなく、先頭の版の番号（`<数字>-`）を外した残りが、ケースの名前と一致するときだけとした（ケースの名前は数字で始まらないので、同じ結果で、より厳しい）。組ごとの時間（JUnitの合計）は、最長がsine2-eventsの3.7分、ほかは3.3・2.5・1.5・0.7・0.6・0.6・0.3分、ファイルの組0.2分。ファイルの合計は13.4分（変更の前は14.0分。使い回しの効果は小さい見込みどおり）、goldenのファイルだけの並列の実行の時間は260秒（このとき、使い回しのtestの1件が上の理由で失敗）。配り方のtestは、`-k`などで一部だけを選んだときも通るように、組の割り当ての全体を、全部のケースの名前の合成のidで確かめる形にした。
- Task 4（2026-10-20 20:47〜20:57、Task 1と同じ条件、commit `03bff70`）: 14640 passed / 8 skipped、exit 0、646.1秒（10分46秒。変更の前の874.7秒から26%短い）。件数の増加60件は、分け直したtest（版5のケース19＋版7・8のケース各21＝61件が、元の3件に代わる＝+58件）と、新しい2件（使い回しのtest、配り方のtest）。失敗0。JUnit（`../../venv/refactoring-tests/golden-parallel-after.xml`）の組ごとの合計は、最長が構成の照合の組1の7.4分、goldenの`fedsda-sine2-events`の組6.6分、構成の照合の組0・2が6.1・6.0分、goldenの`fedsda-sine2-reference-reuse`の組5.6分。goldenのファイルの合計は22.8分（単独の実行の13.4分より大きい。12のworkerが同時に重い実行をするための競合とみられるが、確かめていない）。共通引継ぎ手順（`.kiro/steering/agent-handoff.md`）の所要時間を更新した。
- 実装レビュー1回目（Luna high、session `01a1592d-e872-7712-b966-430e5c228fce`、`ae32e8b..fd6b92c`）: CHANGES REQUIRED、Major 1件（使い回しのtestで、seedを変えた設定を試す前にcacheを空にしていないため、版2はcacheを使い、版3だけが実行される。「違う設定で別に実行する」を確かめていない。Task 2の記録も、testの内容を超えていた）。反映: 試す前にcacheを空にし、版2と版3で2回実行することを確かめる形に直した。cacheの鍵をケースの名前だけに変える変異で、このtestが失敗することを確かめた（元へ戻した）。ほかの確認点（分け直したtestの比べ方、cacheの共有、生成の手順、配り方とidの読み取り、controllerでのimport）は、問題なしとの判定。
