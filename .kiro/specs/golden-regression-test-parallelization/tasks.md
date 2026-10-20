# Implementation Plan

- [ ] 1. 変更の前の所要時間を測る
  - 研究の比較などを同時に動かさずに、全pytestを今の配り方で実行し、全体の時間と、goldenのtestのファイルの時間（JUnitの合計）を記録する。
  - 完了条件: 時間と合否の件数がImplementation Notesにある。
  - _Requirements: 4.2_

- [ ] 2. 実行の使い回しを足す
  - 使い回しのtestを先に書き、REDを確かめる。`capture_case_and_count_divergence_events`を分け、testの側だけがcacheを使う。
  - 完了条件: 使い回しのtest（同じ設定では1回、違う設定では2回実行する）と、goldenのtestの全部が通る。
  - _Requirements: 3.1, 3.2_

- [ ] 3. 版の差の説明のtestを分け直し、配り方を足す
  - 指定するケースが経路を通ることを確かめてから、3つのtestをケースごとに分け直す。配り方のtestを先に書き、`tests/conftest.py`へgoldenのtestのファイルの組を足す。
  - 完了条件: 指定した3つのケースが経路を通ることの確認の結果（ケースごとの、`sample_prediction`の差の有無、クロス評価の有無と下りの増減、統合の数）と、配り方のtestの結果（収集したケースの名前を持つtestの数と、組ごとの数、ファイルの組へ戻ったtestが0であること）がImplementation Notesにあり、配り方のtestと、goldenのtestの全部が、並列の全pytestで通る。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_

- [ ] 4. 変更の後の所要時間を測り、文書を合わせる
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
