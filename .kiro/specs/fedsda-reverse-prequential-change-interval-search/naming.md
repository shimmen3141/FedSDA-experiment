# 命名と役割

命名の方針: 研究で一般的な語（予測的符号長＝prequential code length、前向き＝forward・逆向き＝reverse、途中までの和＝prefix sums）を使い、会話で出ただけの語（「R1」「1本の流れ」「刻み」）を公開名にしない。研究ツールの変種の名前（`default-reverse-prequential-grid1`、`every-sample-change-interval-search-configuration`）は研究ツールの中の名前で、sourceの公開名ではない。

## 新しい公開名

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME`、`EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME`、`CHANGE_INTERVAL_SEARCH_NAMES` | `training_data_assignment_settings`の定数 | 設定`change_interval_search`の2つの値（最小変化区間の倍数のkを、kごとに前向きに測る／最小変化区間以上の全部のkを、逆向きの1本の流れで測る）と、その組 | `NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES`は学習者の組。こちらはkの候補と測り方 |
| `validate_change_interval_search` | 同じmoduleの関数 | 設定の値が、builtin strの2つの値のどちらかであることを検査する | `validate_new_model_hypothesis_learners`は学習者の組の検査 |
| `compute_reinitialized_learner_prequential_code_length_prefixes` | `learning/training/prequential_code_length`の関数 | 初期化し直した学習者で、渡された順の各標本までの予測的符号長（途中までの和）の列 | `compute_reinitialized_learner_prequential_code_length`は全部の標本の和だけ。途中までの和の最後の要素はその値と同じ |
| `compute_warm_started_learner_prequential_code_length_prefixes` | 同じmoduleの関数 | warm startの学習者で、同じ途中までの和の列 | `compute_warm_started_learner_prequential_code_length`は全部の標本の和だけ |
| `derive_reinitialized_learner_single_pass_initialization_seed` | `runtime/alarm_interval_minimum_description_length_selection`の関数 | 1本の流れで全部のkを測るときの、初期化し直す学習者の初期化のseed（clientの番号と警報の標本の番号から） | `derive_reinitialized_learner_initialization_seed`はkごとに測るときのseed（kも含む） |
| `EVERY_SAMPLE_CHANGE_INTERVAL_SEARCH_CONFIGURATION_NAME`（値`fedsda-every-sample-change-interval-search`） | `sweep/experiment_configurations`の定数 | 最終構成の既定から、変化区間の候補の探し方だけを新しい値に替えた構成 | |

設定の項目`TrainingDataAssignmentSettings.change_interval_search`と、関数の引数（`select_alarm_interval_assignment`の`change_interval_search`、`select_assignment_by_minimum_description_length`の`change_interval_step_sample_count`）は、設計に書く。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `select_alarm_interval_assignment` | kは最小変化区間の倍数、kごとに前向きに測る | 設定により、最小変化区間以上の全部のkを、逆向きの1本の流れで測る |
| `select_assignment_by_minimum_description_length` | kの候補は最小変化区間の倍数 | kの候補は、最小変化区間から引数の刻みで（今の扱いは刻み＝最小変化区間で同じ） |
