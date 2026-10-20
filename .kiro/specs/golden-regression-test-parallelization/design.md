# Design Document

## Overview
goldenのtestのファイル（306件、1 processで45.4分）を、ケースの名前の組へ分けて配る。版の差の説明のtestのうち全ケースを回すものを、ケースごとのtestに分け直す。同じprocessの中では、同じrun設定のrunを使い回す。

## Boundary Commitments
### This Spec Owns
`tests/conftest.py`（配り方）、`tests/refactoring/test_important_behavior_regression_goldens.py`（testの分け直し）、`tests/refactoring/behavior_golden_cases.py`（実行の使い回し）、`.kiro/steering/agent-handoff.md`の全pytestの所要時間の記述。
### Out of Boundary
`src/`、goldenのファイル、照合の内容、goldenの生成の手順、ほかのtestのファイル。
### Allowed Dependencies
新しい依存は無い。
### Revalidation Triggers
全pytest（並列）、goldenのtestの全部、配り方のtest。

## Architecture
### 配り方（Requirement 1。`tests/conftest.py`）
- 今の`select_parallel_test_scope`は、構成の照合のファイルだけを、parametrizeの値の先頭の構成の名前で組へ分ける。これに、goldenのtestのファイルを足す。goldenのtestでは、parametrizeの値からケースの名前を取り出し（ケースの名前は`behavior_golden_cases`の全版のケースの名前の和集合。値のidの中のケースの名前を、最も長く一致するもので取る。idの形は`<版>-<ケース>`、`<ケース>`、`<前の版>-<次の版>-<ケース>`）、ケースの名前から組を決める。
- 組: 重いケース（大きい条件のFedSDAの6ケース: `fedsda-sine2-events`、`fedsda-mnist2-decisions`、`fedsda-sine2-reference-reuse`、`fedsda-sine2-alternative-reference-reuse`、`fedsda-sine2-adoption-without-validation`、`fedsda-mnist2-adoption-without-validation`）は1ケース1組。ほかのケース（小さい条件の構成のケースとFedDrift・FedAvg）は、名前の一覧の位置を2で割った余りで2組。合わせて8組。ケースの名前を持たないtestは、今と同じファイルの組。
- 組の数と割り当ては、`tests/conftest.py`の定数に置き、配り方のtest（既存の`test_parallel_scope_keeps_tests_of_one_configuration_in_one_process`と同じ形の、新しいtest）で、収集したgoldenのtestのうち、ケースの名前を持つ全部のtestが、そのケースの組（ファイルの組ではない）に割り当てられること、同じケースのtestが同じ組になること、重い6ケースが別々の組になることを確かめる（idの読み取りの失敗でファイルの組へ戻ったtestを見逃さない）。

### 版の差の説明のtestの分け直し（Requirement 2）
- `test_baseline_6_differs_from_5_only_on_the_prediction_side`: ケース（版5のケース）でparametrizeする。各ケースの比べ方は今と同じ。「`sample_prediction`の表が版5と違うケースが少なくとも1つ」は、`fedsda-sine2-events`のケースのtestで、そのケースの`sample_prediction`の表が版5と違うことを確かめる形にする（版6の設計の下調べで、sine2・mnist2の4ケースで違った。その1つ）。
- `test_baseline_8_differs_from_7_only_in_communication_volume`: ケースでparametrizeする。「下りが減るケースが少なくとも1つ」は、`fedsda-sine2-adoption-without-validation`のケースで、クロス評価が起き、下りが減ることを確かめる形にする。
- `test_baseline_9_differs_from_8_only_where_consolidation_occurred`: ケースでparametrizeする。「統合が起きたケースが少なくとも1つ」は、`fedsda-sine2-adoption-without-validation`のケースで、統合が起きる（`merged_model_count`が1以上。版7からの必須の経路で、統合を求めている）ことを確かめる形にする。
- すでにケースでparametrizeしているtest（`test_baseline_7_keeps_baseline_6_outside_the_new_default`、`test_next_baseline_differs_only_where_the_fixed_behavior_occurred`、数値の照合）は、そのまま（idにケースの名前を持つので、組へ配られる）。
- 実装の前に、ケースの名前を指定して、指定したケースが今の結果で上の経路を通ることを、今の全ケースのtestの結果と照らして確かめる（指定が外れると、確認が失敗する）。

### 実行の使い回し（Requirement 3。`tests/refactoring/behavior_golden_cases.py`）
- `capture_case_and_count_divergence_events`を、(a) run設定から実行してrun・記録の表・パラメータ・事象の数を返す部分と、(b) 版ごとの経路の記録・必須の経路の確認・比べる形の設定を組み立てる部分に分ける。
- (a)の結果を、完全なrun設定の保存の形（`_plain(build_case_settings(...))`のJSON、キーの順をそろえる）を鍵に、process内のcacheへ置き、testの側の`capture_case_once`が使う関数（新しい`capture_case_reusing_runs`）だけがcacheを使う。生成の手順（`generate_golden`と、その中の`capture_case`）は、cacheを使わない。
- FedDriftの実行では、ケースの名前で確かめる項目（`feddrift-drift-tail`の局所学習の回数）があるので、(a)はケースの名前も受け取り、確かめる項目は今と同じにする。cacheの鍵は設定だけ（設定が同じなら、確かめる項目の結果も同じ）。
- 版の差の事象の数は、runの間の包み（警報・登録・確認の数え）から数えるので、設定だけで決まる。使い回してよい。

## Validation Design
- 配り方のtest（Requirement 1.3）。
- 分け直したtestが、今と同じ項目を比べること（実装の後の全pytestで、goldenのtestの全部が成功する）。分け直す前に、指定したケースが経路を通ることを確かめる。
- 使い回しのtest: 実行の関数を包んで回数を数え、同じ完全な設定の2つの「ケース×版」では2回目を実行しないこと、完全な設定が違う2つ（1つの項目だけが違う組を含む）では2回実行すること、経路の記録は版ごとに計算されること（版で項目が違う組で確かめる）。
- 全pytestを変更の前と後で実行し、合否と所要時間を比べる（変更の前の値は、直近の基準版9の全pytest: 14580 passed / 8 skipped、2761秒——手元の研究の比較と同時の実行だったので、変更の前の値を、同じ条件で取り直す）。

## File Structure Plan
- `tests/conftest.py`、`tests/refactoring/test_important_behavior_regression_goldens.py`、`tests/refactoring/behavior_golden_cases.py`、`.kiro/steering/agent-handoff.md`。

## Requirements Traceability
| 要求 | 設計 |
|---|---|
| 1.1〜1.3 | 配り方、配り方のtest |
| 2.1〜2.2 | 版の差の説明のtestの分け直し |
| 3.1〜3.2 | 実行の使い回し |
| 4.1〜4.2 | 全pytestの前後の比較 |
