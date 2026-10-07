# 命名と役割 revision1

## Source/API

| 名前 | 役割/型/単位/所有 |
| --- | --- |
| post_alarm_candidate_validation_session_start.py | runtimeの開始順序の組立。観測/確定とは別 |
| PostAlarmCandidateValidationSession | frozen/kw_only binding record。学習部品/参照/collectionを保持しactive ownerではない |
| start_post_alarm_candidate_validation_session | 入力事前検査後に生成→学習→参照固定→空collection、recordを返す |
| _validate_post_alarm_candidate_validation_start_inputs | metadata/settings/owners/保有/区間/保留の乱数なしpreflight |
| _validate_session_training_tensor_pair | 特徴/ラベルのCPU float32/shape/finite/class検査 |
| proposal_sample_index | 0始まり提案標本位置、後続観測は+1から |
| estimated_change_point_sample_index | 推定変化点の標本位置またはNone。提案位置以下 |
| detection_episode_id | 検出episode IDまたは未採番None、非負整数 |
| detector_name | 空白のみでない検出器の名称str |
| initial_training_model_id | 開始時に読み取った現在学習帰属ID。後続帰属変更から独立 |
| candidate_training_state | 既存IndependentCandidateTrainingState、候補/2optimizerのlive状態 |
| candidate_epoch_training_result | 呼出し内の既存4計数record、所有して保持 |
| training_input_features / training_observed_class_labels | sessionが借用する学習区間、[N,F]/[N,1] |
| pending_assignment_training_samples | exact tuple[ObservedTrainingSample,...]、container固定、tensor borrow、空可 |
| fixed_reference_models | 既存FixedPostAlarmReferenceModels、独立参照/履歴平均 |
| post_alarm_candidate_loss_collection | 既存collection、規定数まで後続で可変 |
| required_sample_count | Tensor組helperの要求件数intまたはNone。保留は1、区間は任意正数 |

引数architecture_reference_classifier/initial_candidate_parameter_snapshot/parameter_optimizer_settings/candidate_epoch_training_settings/candidate_model_training_and_acceptance_settings/held_model_training_state_registry/loss_statistics_store/current_training_model_assignment/input_features/observed_class_labels、既存snapshot/type名は同義で再利用する。小変数held_model_training_states/held_model_training_state/parameter_snapshot/loss_statistics/training_sample/input_feature_count/class_count/model_id/sample_countも既存の同義だけを再利用。

## Tests

- test_post_alarm_candidate_validation_session_start.py: 実旧開始と比較、拒否・接続。
- build_session_start_oracle: build_fixation_oracleを利用し実旧の学習を有効にした開始引数とclientを揃える。
- begin_candidate_validation_session_in_legacy_client: actual _begin_forward_validationを呼ぶ（数値式を複製しない）。
- assert_started_session_matches_legacy: 全candidate/optimizer/参照/metadata/borrow/計数を照合。
- test_session_start_matches_legacy: 方式/class/optimizer/epoch matrixと最終構成。
- test_session_start_rejects_before_mutation: 各preflight条件、不変/RNG照合。
- test_session_start_connects_to_validation_observation: 生成→実学習→規定数観測、候補継続更新でも参照不変。
- test_session_start_dependency_contract: exact AST注入契約。
- session_start_arguments: 公開入口のkeyword dict、旧adoption_argumentsとは区別。
- started_session / legacy_session: 新session record / 旧ForwardValidationSession。
- candidate_parameter_snapshot / reference_parameter_snapshots_by_model_id: 開始後の候補/参照値のcopy、後続不変比較。
- held_parameter_values_and_gradients / held_optimizer_state_snapshots: 借用保有parameter+gradとoptimizerの開始前copy。
- loss_statistics_snapshot / initial_training_model_id: 拒否前統計copy / 帰属ID値。
- legacy_candidate_losses / legacy_reference_losses_by_model_id: actual旧評価の損失列。
- session_input_tensor_values: 区間と保留tensorのclone列。

class_count/optimizer_variant/candidate_training_strategy/maximum_epoch_count/invalid_case/initial_rng_state/expected_rng_state/source_module_path/source_text/expected_acceptance/monkeypatch等は既存testの同義。追加名が必要なら実装前にここへ追記してLunaへ戻す。
