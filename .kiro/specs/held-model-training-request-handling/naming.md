# 学習要求の記録と保有モデルの共同学習 — 命名 revision2

sourceの名前と、testのmodule直下の名前の実装前一覧（範囲は共通引継ぎ手順）。リポジトリ外の下書きを作業ツリーの複製へ置いて`spec_checks.py names`で照合した。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| held_model_training_request_handling.py | runtimeの新module。学習要求の件数管理・共同学習の反復・学習量の計数をつなぐ。既存`local_training_request_schedule.py`（要求の件数と回数の算出だけ）と`held_model_joint_training_iterations.py`（渡された回数の共同学習だけ）を呼ぶ側。「handling」は既存`alarm_occurrence_handling.py`と同じく、1つの出来事に対する一連の処理をつなぐmoduleの意味。 |
| `record_training_request_and_train_held_models_when_due` | 学習要求を1件記録し、保留が更新間隔に達していれば保留中の全要求のぶんを学習する（旧`train_step`）。「when due」は、間隔に達したときだけ学習することを表す。既存`LocalTrainingRequestSchedule.record_training_request`（件数を足して回数を返すだけ）とは別で、学習と計数まで行う。 |
| `train_held_models_for_pending_training_requests` | 保留中の全要求のぶんを、間隔に関係なく学習し、共同更新の完了ごとに計数へ足して、最後に保留を消化する（旧`flush_pending_updates`）。既存`perform_held_model_joint_training_iterations`（回数を受け取って学習するだけ）とは別で、回数を件数管理から求め、計数と消化まで行う。 |
| `_validate_training_request_owners` | 上の2関数が共用する非公開の検査。本処理が読むowner 4つがexact型であることを確かめる。 |
| `has_pending_requests_reaching_update_interval` | 既存`LocalTrainingRequestSchedule`へ足す読取りの操作。保留件数が実行間隔以上かを返す。既存`calculate_pending_joint_update_iteration_count`（回数を返す。一要求あたりの回数が0なら常に0）とは別で、件数だけで決まる。 |
| `joint_update_losses` | 局所名。共同学習の反復を1回ぶん呼んだ戻り値（完了すれば損失1つ、完了しなければ空）。既存の局所名`completed_joint_update_losses`（全回の損失を実行順に集めたもの）の1回ぶん。 |
| `held_model_ids` | 局所名。保有モデルのIDの集合。共同学習の反復の中の同名の引数（抽出へ渡す保有IDの集合）と同じ役割。 |

引数`local_training_request_schedule`、`held_model_training_state_registry`、`training_sample_store`、`model_training_and_assignment_counts_store`と、共同学習の反復へそのまま渡す`batch_sample_count`、`python_random_generator`、`local_training_settings`、`shared_feature_extractor`、`shared_parameter_optimizer`は、吸収・共同学習の反復の同名と同じ役割。局所名`pending_training_request_count`（件数管理のpropertyと同じ値）、`held_model_training_bindings`、`ordered_model_training_samples`、`completed_joint_update_losses`、`_joint_update_iteration_index`、`training_binding`、`model_training_samples`は、共同学習の反復・抽出の同名と同じ役割。importする既存symbol（設計5節）も定義元と同じ役割。

## test（module直下の名前）

| 名前 | 役割 |
| --- | --- |
| test_held_model_training_request_handling.py | 新test module。学習要求の処理を、実旧clientの学習stepと照合する。 |
| `training_request_module` | 新moduleのimport別名（test専用）。共同学習の反復を差し替えるために使う。 |
| `joint_training_iteration_module` | 既存の共同学習の反復のmoduleのimport別名（test専用）。反復の中の共同更新を途中で失敗させるために使う。revision2で追加。 |
| `TRAINING_EVENTS` | 対照に使う、要求（"request"）と明示の消化（"flush"）の列。 |
| `TRAINING_FUNCTIONS` | 上の2種から、新の関数への対応。 |
| `UNHELD_MODEL_ID` | 標本だけを持ち、保有していないモデルのID。 |
| `OWNER_ARGUMENT_NAMES` | 本処理が型を確かめるowner 4つの引数名。 |
| `build_training_request_oracle` | 吸収のoracleの新ownerと実旧clientへ、学習要求の管理と学習の設定を加える。 |
| `run_legacy_training_event` | 実旧の学習stepまたは明示の消化を、指定の乱数状態から実行し、実行後の乱数状態を返す。 |
| `snapshot_training_request_state` / `assert_training_request_state_unchanged` | 吸収のownerの状態に乱数生成器の状態を加えた読取りと、不変の確認。 |
| `spy_on_joint_training_iterations` | 共同学習の反復の呼出しを、引数と呼出し時点の状態つきで記録する（実物をそのまま実行する）。 |
| `test_training_request_handling_matches_real_legacy_training_steps` | 実旧との対照。 |
| `test_training_request_handling_matches_real_legacy_for_other_optimizers` | optimizerの種類を変えた対照。 |
| `test_training_runs_before_counts_are_recorded_and_requests_are_acknowledged` | 各回の共同学習→その回の計数、全回の後に保留の消化、の順と、反復へ渡る値。 |
| `test_training_request_handling_rejects_invalid_owner_before_any_update` | ownerの型の拒否で全状態が不変。 |
| `test_failed_training_keeps_pending_requests_and_counts` | 最初の共同更新より前の失敗で、保留と計数が変わらないこと。 |
| `test_training_failure_after_completed_updates_matches_real_legacy` | 共同更新の合間の失敗で、完了した回の更新・計数と保留件数が実旧と一致すること。revision2で追加（仕様レビューの指摘）。 |
| `test_counts_are_not_recorded_for_iterations_without_completed_update` | 反復が損失を返さなかった回を計数しないこと。revision2で追加。 |
| `test_training_inputs_are_not_read_when_no_training_is_due` | 間隔に達していない・保留が0件のとき、学習へ渡すだけの値を読まないこと。 |
| `test_training_inputs_are_not_read_when_iteration_budget_is_zero` | 一要求あたりの回数が0のとき、学習へ渡すだけの値を読まずに保留を消化すること。revision2で追加（仕様レビューの指摘）。 |
| `test_training_request_schedule_reports_whether_update_interval_is_reached` | 既存の件数管理のtest moduleへ足すtest。読取りの操作。 |

派生型の値は、保持と進行のtestの既存helper `make_subclass_copy`をimportして使う（同じ役割）。
