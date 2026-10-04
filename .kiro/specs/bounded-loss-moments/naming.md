# 命名と役割

revision: 1

| 名前 | 役割・入出力・単位 |
|---|---|
| loss_statistics / bounded_loss_moments.py | 有界損失の数値集計。モデル所属は管理しない |
| BoundedLossMoments | frozen件数・平均・偏差平方和、seedと更新結果 |
| LossMeanAndSampleVariance | 2件以上で生成するfrozen平均・不偏標本分散 |
| observed_loss_count | 実際に集計された件数、sample/series |
| mean_loss | 保存平均、loss |
| sum_squared_loss_deviations | 旧M2相当の偏差平方和、loss²。分散そのものではない |
| sample_variance | M2/(n−1)の不偏標本分散、loss² |
| accumulate_bounded_loss_observation | 明示損失を一件追加し新値を返す、input不変 |
| estimate_loss_mean_and_sample_variance | n<2はNone、保存平均と標本分散を返す |
| loss_moments / observed_loss | 入力値型 / 今回の有限[0,1]損失 |
| _validate_finite_nonnegative_number / specified_value / parameter_name / maximum_value | 明示値・項目名・任意上限を検査しfloat返却 |
| _validate_loss_moment_fields / _validate_loss_moments | field契約 / exact値型を更新せず検査 |
| first_mean_difference / second_mean_difference | 旧平均からの差 / 更新平均からの差 |
| updated_mean_loss / updated_sum_squared_loss_deviations | 今回更新後の局所値 |
| legacy_stats / legacy_client / legacy_class_stats / state_before_call | 旧oracle辞書・stub・class別辞書・入力保存 |
| result / estimate / class_loss_moments / overall_loss_moments | 戻り値・推定・別系列の局所値 |
| invalid_value / field_name / sample_index / class_id / model_id / operation / loss_sequence | test契約値・帰属ID/順序/損失列 |
| global_python_random_state / global_numpy_random_state / global_torch_random_state / global_default_dtype / global_default_device | test共有状態診断、一時torch.device scopeで復元 |
| monitor / observation / reference_losses_by_model_id / reference_historical_mean_losses_by_model_id / available_reference_model_ids / current_training_model_id | 上流public接続のtest値、数値部品は所有しない |

## 検証名
test_bounded_loss_moments.pyと接頭辞test_loss_moments_:
match_legacy_updates / reject_invalid_fields / reject_invalid_observations / distinguish_unsupported_from_zero / match_legacy_class_updates / connect_to_monitor_and_reference_evaluation / preserve_shared_numeric_state / use_keyword_arguments。
旧n/mean/M2/model_stats/class_stats/_update_running_stats等はtest-only oracle参照。既存上流API/AST helperの命名は上流承認を参照。

