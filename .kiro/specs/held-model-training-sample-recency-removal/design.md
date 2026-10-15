# Design Document

## Overview
学習・設定の5つのmoduleは、追加のcommit（be9396b）と、取り下げた警報の形の追加と削除（01112c9・2d352cf）でしか変わっていないので、追加の前（9420d4d）の内容へ戻す。掃引の3つのmoduleは、後のspec（予測の時機、FedAvgの構成、帰属の保留の容量の軸）でも変わっているので、抜き出し方の部分だけを外す。testは、設定の3項目を明示している構築から、その引数を外す（ASTで呼出しを特定して外す）。

## Boundary Commitments
### This Spec Owns
- 追加の前（9420d4d）の内容へ戻すmodule: `core/settings_field_validation.py`、`learning/training/local_training_settings.py`、`learning/training/held_model_training_batch_sampling.py`、`learning/training/held_model_joint_training_iterations.py`、`learning/training/joint_model_parameter_update.py`。戻す前に、`git log 9420d4d..HEAD`で、ほかのcommitの変更が無いことを確かめる。
- `runtime/fedsda_run_client_settings.py`: Kとbatchの件数の組合せの検査を外す（後のcommitの変更も在りうるので、その部分だけを外す）。
- `runtime/fedsda_final_configuration_run_settings.py`、`learning/training/candidate_epoch_training.py`、`runtime/initial_model_pretraining.py`: `LocalTrainingSettings`の構築から3つの引数を外す。
- `sweep/experiment_sweep_plan.py`: 計画の項目`fedsda_training_sample_selections`と検査、`parse_fedsda_training_sample_selection`・`describe_fedsda_training_sample_selection`、`_build_fedsda_settings_variants`の抜き出し方の部分、分岐の条件と最後の検査の、抜き出し方の部分。
- `sweep/experiment_sweep_command.py`: 引数。`sweep/experiment_sweep_records.py`: 列と、`describe_…`のimport、`SWEEP_RECORD_FORMAT_VERSION`を6へ（版の注釈へ、版5——帰属の保留の容量の軸——と版6——抜き出し方の軸を外した——を足す）。
- `tests/refactoring/behavior_golden_cases.py`: `_TRAINING_SAMPLE_SELECTION_DEFAULTS`と、その版1〜5への登録。
- test: `test_held_model_training_sample_recency.py`を削除。設定の3項目を明示しているtestの構築から引数を外す（共用の`tests/refactoring/fresh_process_smoke.py`を含む、約35ファイル。ファイルを列挙せず、完了の条件で確かめる: 3つの項目の名前と`fedsda_training_sample_selections`・`--fedsda-training-sample-selections`・`training_sample_selection`の列の名前が、src・testsで、保存済みのgoldenのJSON以外に残らない）。設定の検査のtestの、項目の型と許す値の表、`int | None`の未対応の例を、追加の前の形へ戻す。掃引の記録の版（5→6）と計画の項目の期待（`test_experiment_sweep.py`の2か所と、`test_fedsda_pending_assignment_fifo_capacity_axis.py`の版5の期待）。コマンドの引数の一覧のtest。帰属の保留の容量の軸のtestの、2つの軸の組のtestを外す。依存境界のtestの、抜き出しのmoduleと計画のmoduleの許可を、使わなくなった名前の分だけ外す（追加のspecで足した`math.floor`・`math.log`・`bisect.insort`・`LocalTrainingSettings`ほか）。FedSDAのrunの識別子の期待値（`test_fedavg_experiment_sweep.py`・`test_feddrift_experiment_sweep.py`の、混合の計画のFedSDAのrunの識別子のリテラル。設定の保存の形が変わるので、新しい値へ合わせる。予測の時機のspecで項目が増えているので、追加の前の値へは戻らない）。
- 文書: `docs/experiments/experiment-sweep.md`（節・引数の表・記録の形式の版の説明）、研究バックログの7（とまとめの表）、再開案内、完了specの一覧。
### Out of Boundary
全履歴からの抜き出しの動作、帰属の保留の容量の軸の動作、予測の時機、結果のディレクトリのscript。
### Allowed Dependencies
新しい依存は無い。依存境界の許可は、使わなくなった名前を外すだけ。
### Revalidation Triggers
FedSDAのrun設定の保存の形（識別子が全部変わる）、goldenの設定の比較、掃引の計画のキー、一覧の表の列、コマンドの引数。

## Architecture
- 追加の前へ戻す5つのmoduleは、`git checkout 9420d4d -- <path>`で戻し、差分が抜き出し方の部分だけであることを`git diff`で確かめる。仕様を書いた時点の確認（`git log --oneline 9420d4d..HEAD -- <5つのmodule>`）では、be9396b・01112c9・2d352cfの3つだけ（01112c9と2d352cfは、取り下げた警報の形の追加と削除）。実装の直前にも、もう一度確かめる。
- 変更は、testが通る1つのまとまり（task 1）で行う。設定の項目を消すと、掃引の計画の`replace`・依存境界の許可（使われない許可名を拒否するtestがある）・設定を作るtestが同時に壊れるため、途中で分けない。
- `_build_fedsda_settings_variants`は、容量の軸だけを当てる。分岐の条件は「FedSDAのrun設定で、容量の軸がある」。最後の検査は、容量の軸だけ。
- 一覧の表の列`pending_assignment_fifo_capacity`の読み方（`fedsda_client_settings`）は、そのまま残す。
- goldenのhelper: 保存したgoldenは3項目を持たず、今の設定も3項目を持たなくなるので、「後に足された項目」の登録を外せば、比較の前の除き方が要らなくなる。

## Validation Design
- 既存のtestが、外した部分を除いて通る（goldenの版1〜5、実旧との照合、抜き出しの関数のtest、共同学習のtest、帰属の保留の容量の軸のtest、掃引のtest）。
- 外したことの確認: `LocalTrainingSettings`に3項目が無い（`fields`）、計画に項目が無い、コマンドが`--fedsda-training-sample-selections`を引数の誤り（終了コード2）にする、表に列が無い。これは、既存のtestの期待（設定の項目の表、コマンドの引数の一覧、記録の計画の完全一致）で確かめる。
- 全pytest、Ruff、Pyright。WSLで、goldenの版1〜5と、抜き出し・共同学習のtest。

## 旧と違う点
FedSDAのrun設定の保存の形から3項目が減るので、FedSDAのrunの識別子が全部変わる。動作と値は、全履歴から一様に抜き出す今の既定と同じ。
