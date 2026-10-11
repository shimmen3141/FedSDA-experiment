# Design Document: fedsda-without-candidate-validation

## 変更

- 設定: `CandidateModelTrainingAndAcceptanceSettings.candidate_model_acceptance_policy`へ、値`current_model_first_reuse_then_candidate_adoption_without_validation`を足す。この方針では、検証の標本件数（`candidate_post_alarm_validation_sample_count`）は使わない。
- 警報の区間の解決（`runtime/alarm_change_interval_resolution.py`の`resolve_alarm_change_interval`）: 合う保有モデルが無く、方針が上の値のとき、候補を学習し、`adopt_candidate_as_current_training_model`（候補検証での採用と同じ関数）で採用して、結果種別`alarm_interval_candidate_adopted_as_new_model`を返す。sessionは作らない。
- 候補の学習の共通化（`runtime/post_alarm_candidate_validation_session_start.py`）: 入力の検査・独立候補の生成・区間での学習を、`train_candidate_on_alarm_change_interval`（戻り値`TrainedAlarmIntervalCandidate`）として切り出し、候補検証の開始と、上の採用の、両方から呼ぶ。候補検証の開始は、その後に、参照モデルを固定する。
- 採用に使う3つ（`temporary_model_id_allocator`、`pending_model_upload_state`、`upload_delay_round_count`）を、標本1件の処理から、警報1回の処理（`handle_alarm_occurrence`）、警報時の応答（`respond_to_alarm_with_buffered_samples`）、区間の解決へ、渡す。
- 結果種別: `ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES`と、適応の結果種別（`AdaptationOutcome`）へ足す。学習帰属が変わる結果（`TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES`）に含める。警報応答の完了（`AlarmResponseCompletion`）は、学習帰属が変わるのは「保有モデルの再利用」か「候補の採用」のときだけ、と検査する。適応の粗い区分は、`adopt_candidate_model`。
- 掃引: 構成`fedsda-without-candidate-validation`を足す。

## 参照モデルを固定しない理由

旧は、候補検証を始めるときだけ、警報の時点の保有モデルの写しを作る（`_snapshot_reference_models`）。写しを作るとき、モデルの初期化で、乱数を消費する。`immediate`では写しを作らないので、乱数を消費しない。新も、この方針では、参照モデルを固定しない（固定すると、乱数の列がずれて、以後の結果が、旧と合わなくなる。最初の実装で、3つの条件が合わなかった原因）。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、`NEW_MODEL_CREATION_POLICY`を`immediate`）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestを足す。
- 既存の単体test: 結果種別の一覧の期待、引数の組立て（採用に使う3つ）、依存境界を、直す。提案構成の照合と、goldenは、そのまま通る。
