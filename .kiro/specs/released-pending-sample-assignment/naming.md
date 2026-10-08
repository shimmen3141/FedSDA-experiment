# 警報のない標本での帰属確定 — 命名 revision1

sourceの名前と、testのmodule直下の名前の実装前一覧（範囲は共通引継ぎ手順）。リポジトリ外の下書きを作業ツリーの複製へ置いて`spec_checks.py names --base HEAD`で照合した。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| released_pending_sample_assignment.py | runtimeの新module。警報のない標本で、保留から解放する標本の帰属を確定する。既存`assigned_training_sample_absorption.py`（帰属が決まった標本をモデルへ吸収する）を呼ぶ側で、どの標本を・どのモデルへ・いつ解放するかを決める。 |
| `assign_released_pending_samples_to_current_training_model` | 容量を超えた保留標本を現在の学習帰属のモデルへ確定し、保留から解放する。「released」は保留の容量を超えて保留から外れる標本、「current training model」は既存`CurrentTrainingModelAssignment`が持つ現在の学習帰属。警報のときの既存`respond_to_alarm_with_buffered_samples`（保留標本を区間に分けて再利用評価などを行う）とは別で、評価や切替を行わない。 |
| `released_sample_observations` | 局所名と戻り値。解放する（した）保留標本の、古い順のtuple。 |
| `get_sample_indices_exceeding_capacity` | 既存`PendingTrainingAssignmentBuffer`へ足す読取りの操作。容量を超える最古の位置を、解放せずに返す。既存`release_sample_indices_exceeding_capacity`（同じ位置を解放して返す）の読取り版。 |
| `exceeding_sample_count` | 上の操作の局所名。容量を超えている件数。 |

引数`pending_training_assignment_buffer`、`pending_sample_observations`、`current_training_model_assignment`、`held_model_training_state_registry`、`training_sample_store`、`model_training_and_assignment_counts_store`、`loss_statistics_store`と、局所名`indexed_observation`は、警報のときの既存の応答・吸収の同名と同じ役割。importする既存symbol（設計5節）も定義元と同じ役割。

## test（module直下の名前）

| 名前 | 役割 |
| --- | --- |
| test_released_pending_sample_assignment.py | 新test module。確定を、実旧の標本処理（警報のない経路）と照合する。 |
| `released_assignment_module` | 新moduleのimport別名（test専用）。吸収を差し替えるために使う。 |
| `FIRST_PENDING_SAMPLE_INDEX` | 保留標本の最初の位置。 |
| `INVALID_RELEASED_ASSIGNMENT_INPUT_CASES` | 拒否条件名から（不正にする引数名、正常な引数から不正な値を作る操作、期待する例外）への対応。 |
| `build_released_assignment_oracle` | 吸収のoracleの標本を保留標本にし、最後の1件を観測した直後の新ownerと実旧clientを作る。 |
| `run_legacy_sample_processing_without_alarm` | 実旧の標本処理へ最後の標本を渡す。 |
| `snapshot_assignment_state` / `assert_assignment_state_unchanged` | 吸収のownerに保留と現在の学習帰属を加えた状態の読取りと、不変の確認。 |
| `replace_observation` | 保留標本のtupleの1件を差し替えたtupleを作る。 |
| `test_released_assignment_matches_real_legacy_sample_processing` | 実旧との対照。 |
| `test_released_samples_are_absorbed_before_pending_positions_are_released` | 吸収が解放より前であることと、吸収へ渡る値。 |
| `test_released_assignment_rejects_invalid_input_before_any_update` | 全拒否条件で全状態が不変。 |
| `test_own_checks_reject_before_absorption_is_called` | 本処理自身の検査が、吸収の呼出しより前に拒否すること。 |
| `test_invalid_sample_that_is_not_released_is_not_inspected` | 解放しない保留標本の中身を読まないこと。 |
| `test_pending_assignment_reports_exceeding_indices_without_releasing` | 既存の保留位置のownerのtest moduleへ足すtest。読取りの操作。 |

派生型の値は、保持と進行のtestの既存helper `make_subclass_copy`をimportして使う（同じ役割）。
