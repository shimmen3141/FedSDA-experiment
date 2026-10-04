# 命名と役割
revision: 2

| 名前 | 役割・単位・更新 |
|---|---|
| model_and_class_loss_statistics.py | 一モデル全体とクラス別の集計値およびその所有 |
| ModelAndClassLossStatistics | 一モデルのfrozen全体/class統計、model IDは外側のkey |
| ModelAndClassLossStatisticsStore | 一所有者のmodel ID別統計を管理 |
| initial_loss_statistics_by_model_id / _model_loss_statistics_by_model_id | 明示初期map / 内部唯一の所有状態 |
| overall_loss_moments / class_loss_moments_by_class_id | 全体集計 / class IDと集計のimmutable pair tuple、到着順 |
| set_model_loss_statistics / loss_statistics | 全体/classの一括置換 / コピー検査する一モデル値 |
| record_assigned_loss / observed_loss / observed_class_id | 帰属時の一損失追加 / loss単位 / 正解classまたはNone |
| get_model_loss_statistics / get_state_snapshot | 欠落Noneか一モデル独立参照 / 全model独立順序付き参照 |
| model_id / class_id | signedモデル識別子 / 非負正解class識別子、どちらも件数ではない |
| _validate_identifier / identifier / parameter_name / minimum_value | ID型/下限検査、項目名と任意最小値 |
| _copy_loss_moments / loss_moments / _copy_model_and_class_loss_statistics | exact入力型と全fieldをpublic constructorで検査して独立コピー |
| copied_overall_loss_moments / copied_class_loss_moments / class_loss_moments / seen_class_ids | 全体コピー / class pairコピー列 / 一class集計 / 重複検査集合 |
| class_loss_moments_pair | 入力tupleの未検査の一要素。shape検査後にclass_id/class_loss_momentsへ展開し、検証済みpair列と区別 |
| validated_model_loss_statistics_by_model_id | 全初期要素検査後の内部map候補 |
| updated_overall_loss_moments / updated_class_loss_moments_by_class_id / updated_class_loss_moments | commit前の全体/全class候補dict/指定class候補 |
| store / other_store / result / state_before_call / snapshot / loss_sequence / seed | test所有者・独立所有者・結果・更新前・参照・帰属列・明示初期値 |
| legacy_client / legacy_stats / legacy_class_stats / legacy_model_stats | test旧oracle stub/全体/class/全model辞書 |
| invalid_value / field_name / operation / observed_loss_count / mean_loss / sum_squared_loss_deviations | test契約値とupstreamfield名 |
| monitor / observation / baseline_mean_loss / reference_historical_mean_losses_by_model_id / reference_losses_by_model_id | 上位public明示接続のtest局所値 |
| global_python_random_state / global_numpy_random_state / global_torch_random_state / global_default_dtype / global_default_device | test共有状態診断 |
| test_model_and_class_loss_statistics.py / test_model_class_statistics_ | testfile/接頭辞。suffix matches_legacy_updates, accepts_and_replaces_seeds, preserves_snapshot_independence, rejects_invalid_input_atomically, connects_to_baseline_and_monitor, preserves_shared_state, uses_keyword_arguments |

dataclassの__post_init__/self、pytestの標準fixture・既存upstream型/引数と旧oracle名は既存承認を参照。新aliasは作らない。storeはクライアント実体を知らず、ModelAndClassにより単一loss数値型との差を表す。
