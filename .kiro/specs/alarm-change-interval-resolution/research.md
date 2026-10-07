# 調査根拠

旧基準748c3aa。主担当がコードを読んだ結果。testは未実施。

- 旧`clients/fedsda.py::_resolve_drift`（評価ループの後）: 適合候補があれば`_select_reuse_candidate`でIDを選ぶ。現行と違えば`reuse_selection_counts["alternative_fit"]`を加算、`local_switch_positions`へ位置を追加、`_set_local_current_model`（現行IDを変更し`_on_local_model_change`を呼ぶ）、drift_type=1、action="reuse"。同じなら`current_fit`を加算、drift_type=0、action="maintain"。どちらも続けて`_absorb_into_store(self.current_model_id, drift_data)`で変化区間を選択後の現行IDへ吸収する。
- 適合候補がなければ`_select_initialization_params(evaluated_candidates)`で初期値を選び、最終構成の方針では`_begin_forward_validation(bx, by, drift_data, initialization_params, sample_idx, estimated_start, episode_id)`を呼ぶ。区間は吸収しない（session.held_dataとして保持）。drift_type=0、action="create_pending"。
- bx/byは変化区間の標本（1件ずつ）を`torch.cat`で連結したTensor。評価と候補の学習の両方に使う。
- 旧の順序は「帰属切替→吸収」。吸収先は切替後の現行ID。新の既存確定処理`apply_post_alarm_candidate_validation_resolution`は再利用で「吸収→切替」の順にしており、最終状態は同じ。通知（旧`_on_local_model_change`）は呼出側の責務で、この順序の差は通知の時点にだけ現れる。
- その後の`_record_adaptation_event`、`_reset_drift_detectors`、`buffer.clear()`、戻り値drift_typeは呼出側。結果種別から導ける: 再利用=1/"reuse"、維持=0/"maintain"、候補検証開始=0/"create_pending"。
- 新の既存部品: `runtime/alarm_interval_model_reuse_assessment.py::evaluate_held_models_for_alarm_interval_reuse`（読取り専用）、`runtime/assigned_training_sample_absorption.py::absorb_assigned_training_samples_into_held_model`（全標本の検証と損失評価の後に更新）、`learning/training/current_training_model_assignment.py::CurrentTrainingModelAssignment.assign_model_for_training`、`methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py::select_candidate_initial_parameter_snapshot`、`learning/models/classifier_parameter_snapshot.py::snapshot_classifier_parameters`、`runtime/post_alarm_candidate_validation_session_start.py::start_post_alarm_candidate_validation_session`（全入力検査後に候補生成→学習→参照固定。乱数消費は実旧と一致）。
- 前specのtestで、評価情報→初期値選択→session開始の接続は実旧と照合済み（test内だけの接続）。productionの組立は今回。
- 旧の他の作成方針（immediate/validated）は最終構成で使わず、新の設定型は警報後検証の方針だけを持つ。移植しない。
- 新しい旧不具合は観測していない。旧の採用分岐の非対称（LEGACY-014）は警報後検証の確定側の話で、今回の再利用・維持の吸収は割当概念計数と損失統計を更新する。
