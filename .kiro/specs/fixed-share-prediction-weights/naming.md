# 命名正本

revision: 1。正式名は本書、承認状態・LF SHA256はspec.json。旧router/probabilities/share_horizonはテストoracleだけ、新srcにaliasを残さない。

## ファイル・公開名

| 名前 | 役割・入出力 |
|---|---|
| fixed_share_prediction_weights.py | モデル別予測重みの状態・更新 |
| FixedSharePredictionWeightController | 一実体の状態所有者 |
| prediction_combination_settings | 明示必須の既承認不変設定 |
| get_prediction_weights_before_label_observation | model_ids: Iterable[int] → 独立dict[int,float] |
| update_weights_after_loss_observation | observed_losses_by_model_id、prediction_weights_by_model_id: Mapping[int,float] → 次回重み更新 |
| select_maximum_weight_model_id | prediction_weights_by_model_id、preferred_model_id: intまたはNone → int。状態不変 |
| reset_weights_after_aggregation | 証拠消去、再較正回数+1 |
| replay_observed_losses | observed_loss_sequence: Iterable[Mapping[int,float]]を検証・列順再生 |
| replay_observed_losses_after_aggregation | 非空列の再較正計数と再生 |

## 状態・補助

以下の状態表の名前は公開読取propertyであり、各backing fieldはその名前に先頭_を付けたprivate属性とする。公開mutable属性は作らない。固定設定は_prediction_combination_settingsだけに保持する。

| 名前 | 意味・単位 |
|---|---|
| weights_by_model_id | dict[int,float]、混合確率、初期空 |
| cumulative_observed_loss_variance | 累積分散、無次元 |
| model_pool_reset_count | 初回を除く集合変更、回/run |
| prediction_weight_leader_switch_count | 最小ID同点leader交代、回/run |
| aggregation_recalibration_count | 集約後再生/明示reset、回/run |
| aggregation_recalibration_sample_count | 集約後再生標本、sample/run |
| _validated_model_ids | 非空・重複なしint集合を昇順tuple化 |
| _validated_observed_losses | 有限損失Mappingを昇順独立dict化 |
| _validated_prediction_weights | 確率の値域・総和を検証コピー |
| _validated_observed_loss_sequence | 全行を先行検証コピー |
| _synchronize_model_pool | 集合差だけ一様重み・ゼロ分散へ |
| _clear_prediction_weight_evidence | 重み・分散消去、計数保持 |

補助はcontroller内のprivate static/method。__init__などPython規約名・既承認設定名を再利用する。

## 引数・局所名

| 名前 | 役割 |
|---|---|
| model_ids / validated_model_ids / model_id | 元集合 / 検証済昇順tuple / 個別ID |
| observed_losses_by_model_id / observed_loss | 観測損失Mapping / 有限値 |
| prediction_weights_by_model_id / prediction_weight | 予測時確率Mapping / 個別値 |
| observed_loss_sequence / validated_observed_loss_sequence | 元列 / 検証済列 |
| total_prediction_weight | 確率総和検査 |
| bounded_losses_by_model_id | 0～1損失 |
| previous_leader_model_id | 更新前最小IDleader |
| expected_observed_loss / observed_loss_variance | 期待損失 / 当該分散 |
| learning_rate | 指数更新係数 |
| unnormalized_weights_by_model_id / total_unnormalized_weight | 指数更新 / 総和 |
| posterior_weights_by_model_id | 正規化後の重み |
| share_probability / uniform_model_weight | 1/時間尺度 / 1/モデル数 |
| maximum_prediction_weight / maximum_weight_model_ids | 最大値 / 同率集合 |

数値は無次元、IDはモデル識別子。時間尺度の元値は既承認fixed_share_weight_redistribution_time_scale_samples。

## 検証

ファイル`test_fixed_share_prediction_weights.py`。テスト名は`test_fixed_share_`+以下の役割句。

- controller_requires_explicit_valid_settings
- prediction_weights_match_reference_before_and_after_each_observation
- model_pool_changes_preserve_order_and_reset_only_when_needed
- leader_selection_resolves_ties_without_changing_state
- updates_use_the_supplied_pre_observation_weight_snapshot
- invalid_model_ids_leave_all_state_unchanged
- invalid_observation_inputs_leave_all_state_unchanged
- replay_matches_reference_for_ordered_and_changing_model_pools
- empty_replay_preserves_evidence_and_all_counters
- invalid_later_replay_rows_leave_all_state_unchanged
- aggregation_reset_and_replay_counters_match_reference
- returned_weights_and_diagnostics_do_not_expose_owned_state
- controller_instances_and_caller_random_state_are_independent
- prediction_weight_calls_require_keyword_arguments

補助`assert_fixed_share_state_matches_reference`（全状態照合）、`capture_fixed_share_controller_state`（診断copy）。
局所`controller`、`repeated_controller`（別実体）、`reference_router`（旧oracle）、`controller_state_before_call`、`returned_prediction_weights`、`reference_prediction_weights`、`invalid_model_ids`、`invalid_observed_losses`、`invalid_prediction_weights`、`invalid_observed_loss_sequence`、`expected_model_id`、`replay_after_aggregation`、`observation_index`（0始まり標本位置）。既存境界検査と共通pytest変数は再利用。追加の意味ある名前は本書へ追記しLunaレビュー後に実装する。
