# 警報時の学習区間の準備 — 命名案 revision3

ファイル/公開契約の初期一覧。承認状態はspec.jsonを参照する。test/局所名はtask開始前に外部下書きASTで抽出し、追加承認する。

revision2では外部runtime/test下書きの関数・class・引数・代入先・内包targetとimport別名を機械抽出した。追加の役割表は下段。外部分割test名は作業用で、本番の検証はdesignの単一testファイルへ結合する。

| 追加名 | 役割 |
| --- | --- |
| np | 既存と同じNumPy module別名。乱数状態の読取り比較に使う |
| preparation_module | 新runtimeのtest用module別名。記録wrapperで公開保存/吸収をpatchする窓口。判断を再実装しない |
| test_alarm_training_interval_preparation_dependency_contract | 新3moduleのexact依存許可/拒否を検証するtest |
| source_module_path, source_text, expected_acceptance, dependency_boundary_violations | 既存AST検証と同じ役割の引数/検査結果。新しい依存判定ownerを設けない |

| 名前 | 型・役割・違い |
| --- | --- |
| indexed_observed_training_sample.py | 学習層の位置付き観測標本record。標本の所有者ではない |
| IndexedObservedTrainingSample | frozen入力。位置と観測済み標本と診断概念IDを結ぶ。ObservedTrainingSample自体には位置がない |
| sample_index | 非負builtin int。クライアント観測列の位置。roundやモデル内の位置ではない |
| training_sample | ObservedTrainingSampleの借用参照。Tensorのcopyはしない |
| observed_concept_id | intまたはNone。評価用の真概念ID、分割/モデル選択に使用しない |
| prepared_alarm_training_intervals.py | 前区間の処理が済んだ結果recordを定義 |
| PreparedAlarmTrainingIntervals | frozen返却型。BufferedChangeIntervalPartitionは位置だけの分割、本型はpayloadと前区間処理済みの情報も持つ |
| earlier_observations | tuple[IndexedObservedTrainingSample,...]。変化区間より前、処理順を保持 |
| change_interval_observations | 同tuple。変化区間、未吸収。後続の区間解決へ渡す |
| change_interval_start_sample_index | intまたはNone。FIFO切詰め後の変化区間先頭。検出器の推定開始位置と区別 |
| earlier_interval_absorbed_model_id | intまたはNone。前区間を吸収済みの帰属ID。前区間空ならNone |
| alarm_training_interval_preparation.py | runtimeの区間準備。警報全体の解決ではない |
| prepare_alarm_training_intervals | 位置照合/区間分割/前区間保存と吸収を行いPreparedAlarmTrainingIntervalsを返す。学習optimizer更新はしない |
| pending_training_assignment_buffer | PendingTrainingAssignmentBuffer。保留位置owner、読取りのみ |
| pending_sample_observations | tuple[IndexedObservedTrainingSample,...]。保留順で供給する位置付きpayload列 |
| estimated_change_span_sample_count | 正のbuiltin int。推定変化区間の標本数。実件数とは限らない |
| current_training_model_assignment | CurrentTrainingModelAssignment。現在帰属owner、ID読取りのみ |
| model_evaluation_sample_store | ModelEvaluationSampleStore。前区間の評価標本を保存、学習storeと区別 |
| python_random_generator | exact random.Random。評価抽出用の明示乱数、torch生成器と区別 |
| held_model_training_state_registry | HeldModelTrainingStateRegistry。現行分類器を取得、一覧を更新しない |
| training_sample_store | ModelTrainingSampleStore。前区間学習標本の追加先 |
| model_training_and_assignment_counts_store | ModelTrainingAndAssignmentCountsStore。既存吸収で診断概念の割当計数を更新 |
| loss_statistics_store | ModelAndClassLossStatisticsStore。前区間の全体/クラス別損失を更新 |

既存公開名は元specと同じ意味で再利用する。新永続状態を設けない。private検査helper、test名と引数/局所名は実装前の追加表で役割を承認する。

# 外部下書きから抽出した追加命名案

全関数定義・引数・代入先・内包targetをASTで抽出し、初期naming表の名前を除外。test case dict keyは個別登録しない。最終結合版を含む。既存helperや局所名の同役割再利用も出現箇所を記録する。

| 名前 | 役割案 | 出現箇所 |
| --- | --- | --- |
| `_` | 使用しないoracle戻り値または反復値。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `actual_joint_loss` | 新共同更新戻り損失。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `added_batch_sample_count` | 評価追加抽出の上限。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `alarm_change_interval_resolution` | 実解決結果record。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `alarm_interval_resolution_case` | 再利用・維持・候補開始の既存oracle case。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_integration.py（argument） |
| `assert_evaluation_samples_match_legacy` | 評価保存のモデル順・抽出順・payload identityを実旧stored_dataへ照合する。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `assert_preparation_checkpoint` | 標本・計数・統計・評価保存・Python最終Randomを旧前区間処理後へ照合。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `buffered_change_interval_partition` | 位置だけを持つ公開分割結果。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `build_alarm_preparation_oracle` | 既存実NN解決oracleに同一payloadの前区間、位置FIFO、評価store、明示Randomを加える。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `change_interval_input_features` | 変化区間モデル状態比較用の特徴batch。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `change_interval_training_samples` | 既存解決oracle変化区間の標本列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `change_sample_count` | 実保留件数と推定spanの小さい方。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `class_count` | 分類数2/4。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_integration.py（argument） |
| `classifier` | 現行IDの実保有分類器。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `collection` | モデル別標本collection。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `collections` | 新評価保存の公開collection列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `counts_store` | 訓練と概念割当計数owner。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `current_training_model_id` | 帰属ownerから読み取る実吸収先ID。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `detector_reset_calls` | 新準備対象外の旧検出器reset記録。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `earlier_sample_count` | 前区間の件数。公開分割またはoracle追加件数。 | alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `earlier_training_samples` | 実旧へ同じpayloadで供給する前区間標本。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `estimated_span` | 旧警報に渡す推定変化区間件数。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `evaluation_sample` | 新評価record1件。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `evaluation_snapshot` | 拒否/空時比較用の公開評価保存snapshot。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `event_fields` | 実旧通知callbackのkeyword集合。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `existing_evaluation_samples` | 事前保存あり/なしのcase。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `expected_field_names` | recordの期待field順列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `expected_joint_loss` | 実旧共同更新戻り損失。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `expected_owner_type` | 引数ごとのexact受理型。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `explicit_python_random_state` | 拒否前の新明示Random状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `field` | dataclass field反復値。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `field_name` | 凍結代入の対象field名。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `first_model_id` | 区間評価checkpointを記録する保有順先頭の旧モデルID。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `first_sample_index` | proposal末尾に合わせる保留列先頭位置。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `global_python_random_state` | 旧実行前に退避しfinallyで復元するglobal Python状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `indexed_observation` | 検査・分割・保存対象の位置付き観測1件。 | alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `initial_python_random_state` | 旧global Pythonと新明示Randomの開始状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `initial_torch_random_state` | 旧候補開始前Torch状態。新を同じ状態で再実行。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `initial_training_model_id` | 警報解決前の帰属ID。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `input_features` | 特徴Tensor。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `invalid_case` | owner/record/位置/span/未保有の拒否case。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `invalid_payload` | shape/dtype/device/有限性/クラス/特徴数の拒否case。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `invalid_sample_offset` | 前区間先頭/後半の不正標本offset。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `legacy_adaptation_events` | 実旧記録event列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_client` | 実NNを持つ実旧client。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_drift_type` | 実旧_resolve_driftの戻り値。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_epoch_training_calls` | 実旧候補epoch呼出しをwrapした記録。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_evaluation_sample` | 実旧保存評価payload1件。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `legacy_evaluation_samples` | 実旧モデル別保存評価標本列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `legacy_training_batches` | 実旧標本storeから作る固定batch列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_training_sample` | 実旧訓練payload1件。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `legacy_training_samples` | 実旧モデル別訓練標本列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `local_training_settings` | 既存共同更新方式設定。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `maximum_stored_sample_count_per_model` | 評価storeのモデル別容量。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `minimum_change_sample_count` | 旧MIN_DRIFT_DATAへ設定する不足/十分閾値。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `model_id` | 対象モデルID。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `model_training_sample_collections` | モデル別訓練標本公開snapshot。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `monkeypatch` | 一時差替pytest fixture。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_integration.py（argument） |
| `negative_current_model` | 負ID現行モデルのcase。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `numpy_state` | 最終NumPy乱数状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `observation` | 位置付き観測1件。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `observations` | 位置付き保留観測列または不正注入用list。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `observed_class_labels` | 観測ラベルTensor。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `operation_calls` | 保存と吸収の実呼出し順列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `operation_keyword_arguments` | wrapperが実処理へそのまま渡すkeyword。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `optimizer_variant` | 共同更新oracleのoptimizer方式。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `original_absorption` | 実公開吸収関数。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `original_per_sample_error` | wrapperが保持する実旧損失評価method。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `original_save` | 実評価保存method。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `owner_name` | owner取得・拒否理由の引数名。 | alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `participating_training_batches` | 共同更新に参加する保有順batch。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `pending_assignment_buffer_capacity_samples` | 保留FIFOの平時容量。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `pending_assignment_snapshot` | 呼出し前に取得する公開FIFO位置snapshot。 | alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `pending_buffer` | 各件数caseの実FIFO。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `pending_sample_concept_ids` | 保留列の診断概念ID列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `pending_sample_count` | 空・1件・capacity+1の保留件数。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `pending_snapshot` | FIFO非消費の比較snapshot。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `pending_training_samples` | 前区間と変化区間の連結標本列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `per_sample_bounded_losses` | 公開評価の1標本有界損失。吸収時の再評価と共有しない。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `preparation_arguments` | 新準備のkeyword集合。旧開始Random keyは新呼出し後にだけ追加。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `preparation_checkpoint_calls` | 旧前区間後の照合が1回通った記録。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `prepared_intervals` | 前区間吸収済みの新結果record。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `previous_snapshot` | owner/parameter/gradient/optimizer/global RNGの事前snapshot。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `python_random_state` | 新明示Randomの事前比較状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `random_states` | 旧完了時のTorch/Python/NumPy状態。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `record` | 入力または結果record。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_records.py（assignment/comprehension） |
| `record_interval_evaluation` | 実旧最初の区間batch評価入口でcheckpointを記録し、実評価を呼ぶ。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `record_legacy_adaptation_event` | 不足event前にcheckpointを確認し実eventを記録。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `recorded_absorption` | 保存後の順序を確認して実吸収関数を呼ぶ。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `recorded_save` | 保存順を記録して実保存methodを呼ぶ。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `registry` | 保有モデル訓練状態registry。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `resolution_arguments` | 既存変化区間解決のkeyword集合。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（argument）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `run_joint_update_in_both_implementations` | 各標本storeから固定batchを作り両実装の共同更新・計数・モデル全状態を照合。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_integration.py（function） |
| `run_legacy_alarm_with_preparation_checkpoint` | 実旧_resolve_driftを実行し前区間保存吸収後の状態を比較。旧保存・吸収をstubしない。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `sample_offset` | 標本列の0始まり反復位置。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |
| `self` | 実評価保存methodのowner。 | test_alarm_training_interval_preparation.py（argument）、test_alarm_training_interval_preparation_core.py（argument） |
| `shared_optimizer_owners` | 既存oracleの共有optimizer所有者列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `source_module_path` | 既存dependency guardと同じsource moduleのpath。新sourceを列挙する用途で再利用。 | test_single_run_dependency_boundaries.py（既存同役割） |
| `state_owner` | exact型検査対象owner。 | alarm_training_interval_preparation.py（assignment/comprehension） |
| `test_alarm_preparation_matches_real_legacy_before_minimum_count_decision` | 2/4class、正規/負ID、不足/十分、追加0/部分/全数、容量超過、事前保存あり/なしを実旧_resolve_driftの前区間checkpointへ照合。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `test_alarm_preparation_preserves_empty_and_capacity_plus_one_buffers` | 空/1/capacity+1とspan短い/等しい/長い、正規/負ID、返却identity、FIFO/明示Random非変更を確認。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `test_alarm_preparation_records_are_immutable_and_borrow_payloads` | 入力/結果recordのfield順、keyword専用、frozen、Tensor借用、constructor非検査を確認。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_records.py（function） |
| `test_alarm_preparation_rejects_contract_errors_without_state_changes` | owner/Random/tuple/record/span/位置完全対応/未保有の拒否と全状態不変を確認。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `test_alarm_preparation_rejects_every_earlier_sample_before_any_state_update` | 先頭/後半の不正shape/dtype/device/有限値/クラス/特徴数を、保存や全RNG更新前に拒否し全状態不変を確認。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `test_alarm_preparation_saves_evaluation_before_real_absorption` | 実保存・実吸収のwrapper記録により保存→吸収の順を確認。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_core.py（function） |
| `test_alarm_training_interval_preparation_dependency_contract` | 新3moduleのexact leaf依存と禁止import注入を確認する境界test（主担当で実装）。 | test_single_run_dependency_boundaries.py（function（予定）） |
| `test_prepared_alarm_interval_continues_resolution_and_joint_training` | 2/4classと再利用/維持/候補開始で実準備→実解決→共同更新を実旧へ全状態照合。 | test_alarm_training_interval_preparation.py（function）、test_alarm_training_interval_preparation_integration.py（function） |
| `training_batch` | 1モデル共同更新batch。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `training_binding` | 1モデル分類器と概念optimizerのbinding。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `training_bindings` | 保有順の訓練binding列。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `update_shared_features` | 共同更新で共有特徴更新を有効にする指定。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_integration.py（assignment/comprehension） |
| `valid_preparation_arguments` | 不正引数注入後も正常ownerへアクセスする浅いcopy。 | test_alarm_training_interval_preparation.py（assignment/comprehension）、test_alarm_training_interval_preparation_core.py（assignment/comprehension） |

## importする既存の型・操作と定義元

各import名は表の定義元と同じ責務で再利用する。runtimeは既存owner・公開操作を組み立て、旧APIと共有oracle helperはtestだけで用いる。標準ライブラリ等はsnapshot、不変性検査、記録wrapper、乱数照合に用いる。型・操作の詳細な契約は定義元の完了specとdesignのAllowed Dependenciesを参照する。

| 名前 | 定義元と維持する役割 |
| --- | --- |
| `ALARM_INTERVAL_RESOLUTION_CASES` | `test_alarm_change_interval_resolution.ALARM_INTERVAL_RESOLUTION_CASES`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `Counter` | `collections.Counter`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `CurrentTrainingModelAssignment` | `federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `FrozenInstanceError` | `dataclasses.FrozenInstanceError`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `HeldModelTrainingStateRegistry` | `federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `IndexedObservedTrainingSample` | `federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `LocalTrainingSettings` | `federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ModelAndClassLossStatisticsStore` | `federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ModelEvaluationSampleStore` | `federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ModelTrainingAndAssignmentCountsStore` | `federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ModelTrainingSampleStore` | `federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ObservedEvaluationSample` | `federated_learning_experiments.evaluation.model_evaluation_sample_records.ObservedEvaluationSample`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ObservedTrainingSample` | `federated_learning_experiments.learning.training.model_training_sample_records.ObservedTrainingSample`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `ParticipatingModelTrainingBatch` | `federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `PendingTrainingAssignmentBuffer` | `federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `PreparedAlarmTrainingIntervals` | `federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `Random` | `random.Random`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `Tensor` | `torch.Tensor`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `TrainingDataAssignmentSettings` | `federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings.TrainingDataAssignmentSettings`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `absorb_assigned_training_samples_into_held_model` | `federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `assert_absorption_matches_legacy` | `test_assigned_training_sample_absorption.assert_absorption_matches_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_alarm_change_interval_resolution_matches_legacy` | `test_alarm_change_interval_resolution.assert_alarm_change_interval_resolution_matches_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_alarm_change_interval_resolution_state_unchanged` | `test_alarm_change_interval_resolution.assert_alarm_change_interval_resolution_state_unchanged`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_held_model_states_match_legacy` | `test_adopted_candidate_initial_local_registration.assert_held_model_states_match_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_model_counts_match_legacy` | `test_model_training_and_assignment_counts.assert_model_counts_match_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_store_statistics_match_legacy` | `test_loss_statistics_model_id_reassignment.assert_store_statistics_match_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `assert_training_samples_match_legacy` | `test_adopted_candidate_local_adoption.assert_training_samples_match_legacy`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `build_alarm_change_interval_resolution_oracle` | `test_alarm_change_interval_resolution.build_alarm_change_interval_resolution_oracle`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `config` | `federated_drift_experiment.config`; 固定旧oracleの設定。testからのみ使用 |
| `defaultdict` | `collections.defaultdict`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `deque` | `collections.deque`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `evaluate_classifier_per_sample_bounded_losses` | `federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation.evaluate_classifier_per_sample_bounded_losses`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `fields` | `dataclasses.fields`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `np` | `numpy`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `patch` | `unittest.mock.patch`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `perform_joint_model_parameter_update` | `federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `preparation_module` | `federated_learning_experiments.runtime.alarm_training_interval_preparation`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `prepare_alarm_training_intervals` | `federated_learning_experiments.runtime.alarm_training_interval_preparation.prepare_alarm_training_intervals`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `pytest` | `pytest`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `random` | `random`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `replace` | `dataclasses.replace`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
| `resolve_alarm_change_interval` | `federated_learning_experiments.runtime.alarm_change_interval_resolution.resolve_alarm_change_interval`; 既存公開owner/型/操作。定義元と同じ意味で再利用し、runtimeの依存は設計の許可集合に限定 |
| `run_legacy_joint_update` | `test_joint_model_parameter_update.run_legacy_joint_update`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `snapshot_alarm_change_interval_resolution_state` | `test_alarm_change_interval_resolution.snapshot_alarm_change_interval_resolution_state`; 既存の実旧oracle/全状態assert/共同更新helper。同じ入力・比較責務でtestだけから再利用 |
| `torch` | `torch`; 標準/検証/数値ライブラリ。testのsnapshot・不変性assert・乱数比較、またはruntimeのTensor/Random型として使用 |
