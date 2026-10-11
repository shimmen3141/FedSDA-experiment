# 命名表: fedsda-without-candidate-validation

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `current_model_first_reuse_then_candidate_adoption_without_validation` | `CandidateModelTrainingAndAcceptanceSettings.candidate_model_acceptance_policy`の値 | 保有モデルの再利用を先に試し、合うモデルが無ければ、学習した候補を、検証せずに採用する |
| `alarm_interval_candidate_adopted_as_new_model` | 警報の区間の解決の結果種別、適応の結果種別 | 警報の位置で、学習した候補を、新しいモデルとして採用した |
| `train_candidate_on_alarm_change_interval` | `runtime/post_alarm_candidate_validation_session_start.py` | 入力を検査して、独立候補を作り、警報の変化区間で学習する |
| `TrainedAlarmIntervalCandidate` | 同上 | 学習した独立候補と、その学習の結果 |
| `CANDIDATE_VALIDATION_ABLATION_CONFIGURATION_NAME`（値`fedsda-without-candidate-validation`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
