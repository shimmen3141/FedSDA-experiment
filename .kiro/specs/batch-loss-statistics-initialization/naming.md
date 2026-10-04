# 命名と役割
revision: 1

| 名前 | 役割・単位・更新 |
|---|---|
| batch_loss_statistics_initialization.py | batch損失からモデル全体/class初期統計を作るpure処理 |
| initialize_model_and_class_loss_statistics_from_batch | 外部batch損失から既存immutableモデル集計を新規作成、storeを更新しない |
| _validate_batch_loss_statistics_inputs | reduction前にloss/label/class_count契約を検査、返却None |
| per_sample_bounded_losses / observed_class_labels / class_count | CPUfloat32[N]の有界loss / [N,1]正解class / 宣言クラス数>=2 |
| batch_sample_count / overall_mean_loss / overall_sample_variance | batch件数 / 全体平均loss / 不偏標本分散loss²、n1は旧fallback.1 |
| class_id / class_bounded_losses / class_sample_count / class_mean_loss / class_sample_variance | 昇順class番号 / 対象loss vector / 件数 / 平均 / 不偏分散(n1=0) |
| class_loss_moments_by_class_id / flat_class_labels | classIDと既存集計のpair列 / mask用の一次元ラベル |
| loss_values / label_values / per_sample_bounded_losses / observed_class_labels / class_count | test入力列とtensor、クラス数 |
| legacy_client / legacy_model / legacy_stats / bx / by / model / temp_id / pending_ready | test旧register oracleのstubと旧引数（実装の新APIには旧名を持ち込まない） |
| loss_statistics / seed / store / result / state_before_call / snapshot / invalid_value / field_name / operation | test集計・所有者・結果・入力保存・異常ケース |
| global_python_random_state / global_numpy_random_state / global_torch_random_state / global_default_dtype / global_default_device / global_grad_enabled | test共有状態診断 |
| test_batch_loss_statistics_initialization.py / test_batch_initial_statistics_ | testfile/接頭辞。suffix matches_legacy_registration, rejects_invalid_input_without_mutation, preserves_independence_and_shared_state, connects_to_store_and_baseline, uses_keyword_arguments |

torch/pytest/stdlib APIと既存upstream型/field/関数の命名は既存承認を参照。初期化は生成だけを指し、学習・登録・所有更新を行わない。
