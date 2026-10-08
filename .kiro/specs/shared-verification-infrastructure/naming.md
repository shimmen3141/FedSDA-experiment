# 検証基盤の整理 — 命名 revision1

追加する名前の実装前一覧。範囲は共通引継ぎ手順のとおり、testはmodule直下の名前だけ（sourceの変更はない）。リポジトリ外の下書きを作業ツリーの複製へ置いて`spec_checks.py names --base HEAD`で照合した。承認状態はspec.json。

## `tests/refactoring/fresh_process_smoke.py`（新規。test moduleではないscript）

| 名前 | 役割 |
| --- | --- |
| fresh_process_smoke.py | 新実装だけを別processで動かす共用script。「fresh process」は、pytestや旧実装を読み込んでいない新しいprocessのこと。 |
| `HELD_MODEL_IDS` | 流れで保有させるモデルID（正式ID、他の正式でないID、現行）。 |
| `CURRENT_MODEL_ID` / `OTHER_MODEL_ID` | 最初の現行モデルのIDと、再利用の相手にする保有モデルのID。 |
| `SAMPLE_COUNT` / `VALIDATION_SAMPLE_COUNT` | 最初の警報までの標本数と、候補検証に要する標本数。 |
| `DETECTOR_NAME` / `EPISODE_ID` | 記録へ渡す検出器名とepisode ID（値が記録まで届くことを確かめるための固定値）。 |
| `ALARM_OUTCOMES_WITHOUT_ACTIVE_VALIDATION` | 候補検証中でない警報がとりうる結果の値。 |
| `MODEL_SWITCHING_OUTCOMES` | 学習帰属が変わる（診断が再始動する）適応結果の値。 |
| `SMOKE_SCENARIOS` | 流れの一覧。（名前、最初の警報の推定区間長、警報時に適合させるモデル、期待する最初の結果、候補検証の進め方）。 |
| `run_smoke_scenario` | 1つの流れを実行し、観測した適応結果の列を返す。食い違いはAssertionError。 |
| `main` | 全流れを実行し、必要な結果の観測と、旧実装・test moduleを読み込んでいないことを確かめて、目印の行を出力する。 |

## `tests/refactoring/test_fresh_process_smoke.py`（新規）

| 名前 | 役割 |
| --- | --- |
| test_fresh_process_smoke.py | 上のscriptを別processで実行するtest module。 |
| `SMOKE_SCRIPT_PATH` | scriptの場所。 |
| `test_fresh_process_smoke_runs_without_legacy_or_test_modules` | 別processで実行し、成功と目印の行を確かめる。 |

## `tests/refactoring/test_single_run_dependency_boundaries.py`（追加）

| 名前 | 役割 |
| --- | --- |
| `collect_module_allowed_dependency_names` | `dependency_is_allowed`のうち、moduleごとに名前の組で書いた許可集合を、test file自身のASTから読む。既存`collect_dependency_boundary_violations`（違反を集める）とは別で、登録そのものを読む。 |
| `test_module_allowed_dependencies_are_all_imported_by_the_module` | 許可集合の各名前が、そのmoduleの実際のsourceでimportされていること。 |

## `tests/refactoring/test_held_candidate_validation_progress.py`（追加）

| 名前 | 役割 |
| --- | --- |
| `make_subclass_copy` | 同じ属性を持つ、派生型の値を作る（exact型の検査が拒否するべき値）。alarm-occurrence-handlingのtestの`make_uninitialized_subclass_instance`は属性を持たない値を作る。こちらは、型検査が抜けたときに後続の処理が正常に進む値にするため、属性を写す。 |

既存のtest関数・helperの名前は変えない。
