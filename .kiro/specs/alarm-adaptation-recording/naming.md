# 命名 revision3

| 名前 | 役割 |
| --- | --- |
| AdaptationOutcome | 適応の5結果のLiteral型 |
| AdaptationRecord | 一回の適応の不変記録 |
| AdaptationRecordSnapshot | 記録ownerの不変snapshot |
| AdaptationRecordStore | 適応記録・切替位置・再利用計数の唯一のowner |
| append_adaptation_record | 検査済み記録を追加し位置/件数を反映 |
| record_completed_alarm_response | 警報完了情報を適応記録にして追加 |
| adaptation_outcome | 適応結果の明確な値 |
| adaptation_record | 一件の適応記録 |
| adaptation_records | 入力順の適応記録 |
| adaptation_record_store | 記録owner |
| validated_adaptation_record | 追加前に再検査した記録 |
| training_model_switch_sample_indices | 学習帰属が実際に変わった標本位置 |
| alternative_model_reuse_count | 他保有モデルへの再利用件数 |
| current_model_fit_count | 現行モデルが適合した件数 |
| _adaptation_records | owner内部の記録list |
| _training_model_switch_sample_indices | owner内部の切替位置list |
| _alternative_model_reuse_count | owner内部の他保有モデル再利用件数 |
| _current_model_fit_count | owner内部の現行モデル適合件数 |

既存名alarm_sample_index、detector_name、previous_training_model_id、current_training_model_id、estimated_change_point_sample_index、detection_episode_id、alarm_response_completion、get_state_snapshot、parameter_name、specified_value、model_id、sample_index、selfは上流と同じ役割で再利用する。ファイル名adaptation_record_store.pyは適応記録の保持、alarm_adaptation_recording.pyは警報完了記録の組立を表す。testの全束縛名は下書きからASTで洗い出して承認前に追記する。adaptation_outcomeとresponse_outcomeの区別はdesign.md 3節。

## testの下書きから抽出した名前

| 名前 | 役割 |
| --- | --- |
| RECORDING_ORACLE_CASES | 旧対照の5応答条件 |
| INVALID_ADAPTATION_RECORD_FIELDS | recordの不正field条件と例外 |
| build_completed_recording_oracle | 実旧と新上流で記録対象の完了情報を作る |
| make_adaptation_record | 単体検査用recordの生成helper |
| assert_recording_preserves_upstream_state | 記録前後で上流状態と乱数の不変を確認 |
| record_and_compare_legacy_event | 新記録を追加して実旧eventと比較するtest内操作 |
| reject_recording_before_append | 不正入力をappend前に拒否したことを確認する操作 |
| test_completed_alarm_record_matches_all_real_legacy_event_fields | 実旧の全fieldと計数/位置の対照 |
| test_record_constructor_and_store_reject_invalid_fields_before_any_update | constructor/store検査と状態不変 |
| test_record_store_keeps_input_order_equal_positions_and_old_immutable_snapshot | 順序・重複位置・snapshot不変 |
| test_recording_rejects_invalid_input_before_updating_populated_store | 記録済みownerへの不正入力拒否 |
| recording_oracle_case | 選んだ旧対照条件 |
| expected_response_outcome | 対照条件の期待応答結果 |
| recording_arguments | 記録関数へ渡す引数 |
| recording_operation | 状態不変を確認する対象操作 |
| record_field_overrides | 単体recordのfield置換値 |
| upstream_state_snapshot | 上流のモデル・帰属等の記録前snapshot |
| session_bindings | 同じsession内のfield参照一覧 |
| binding_name | session fieldの名前 |
| binding_value | session fieldの参照値 |
| candidate_optimizer_snapshots | 候補optimizerの参照と全状態 |
| candidate_loss_collection_state | 候補損失収集の記録前snapshot |
| monitor_reset_calls | resetを呼ばないことを確認するspy |
| pending_index_drain_calls | drainを呼ばないことを確認するspy |
| append_record_calls | 不正入力時にappendを呼ばないことを確認するspy |
| adaptation_record_snapshot | 記録後のowner snapshot |
| empty_snapshot | 記録開始前のsnapshot |
| first_adaptation_record | 最初のrecord |
| first_snapshot | 一件目の記録後snapshot |
| final_snapshot | 複数件を記録した後のsnapshot |
| invalid_adaptation_record | frozen fieldを破壊した検査用入力 |
| invalid_recording_case | 記録関数の拒否条件 |
| valid_adaptation_record | constructorが受理した正常record |

import別名valid_run_settings_mappingは既存fixtureと同じ役割。その他の既存test名・局所名は元helperと同義で再利用する。sourceの局所名は上記のrecord/ID/位置を再利用し、追加の名前が出たら命名レビューへ戻す。

## 型・単位・状態と区別

AdaptationOutcomeはdesign.md 4節の5文字列のLiteral、adaptation_outcomeはその履歴の値で、警報応答のresponse_outcomeを今回写す。後続の候補確定履歴も扱うため履歴fieldは別の責務名にしている。
位置fieldは標本番号のbuiltin int（bool不可・非負）、episode IDは位置ではなくbuiltin intの非負識別子、モデルIDは負の一時IDを許すbuiltin int、detector_nameは非空builtin str。件数はintの累積件数。
append_adaptation_recordはrecordを受けNoneを返し、4つの内部状態だけを更新する。get_state_snapshotは入力なしで不変snapshotを返し更新しない。record_completed_alarm_responseは完了情報/検出器名/storeを受けrecordを返し、storeだけを更新する。
training_model_switch_sample_indexは上流completionの一位置またはNone、training_model_switch_sample_indicesはownerの累積位置tuple。adaptation_recordは入力/生成記録、validated_adaptation_recordは再検査済みの独立copyでownerが保存する。castは型注釈の橋渡しで実検査はconstructor、get_argsはtestでLiteralの5値を取り出す。
record_and_compare_legacy_eventは実際にrecordの追加と全fieldのassert比較を行うtest内操作である。判定をしない記録専用wrapperに判定名を付けたものではない。

| 再利用する名前 | 元と同じ役割 |
| --- | --- |
| previous_snapshot | 拒否時不変を比べるownerの操作前snapshot |
| expected_exception | 不正条件の期待例外型 |
| candidate_parameter_snapshot | 候補分類器の値/gradの操作前snapshot |
| optimizer_owner | optimizerを所有する既存owner |
| parameter_optimizer | ownerが保持するoptimizerの参照 |
| optimizer_state | optimizer state_dictの操作前copy |
| monitoring_state | 損失監視の操作前snapshot |
| pending_assignment_state | 保留位置bufferの操作前snapshot |
| python_random_state | Python乱数の操作前状態 |
| candidate_training_state | 開始session内の候補学習状態 |
| initial_torch_random_state | 新旧実行の開始条件を合わせるtorch RNG状態 |
| legacy_result | 実旧メソッドの返却値 |
| response_arguments | 警報応答へ渡す実上流の引数 |
| completion_arguments | 警報完了へ渡す実上流の引数 |
| resolution_arguments | 変化区間解決のownerを含む引数 |
| shared_optimizer_owners | 既存の共有optimizer ownerの一覧 |
| legacy_client | 実旧メソッドのoracle用client |
| alarm_interval_resolution_case | 既存の解決条件の選択 |
| preparation_arguments | 区間準備へ渡す既存の引数 |
| active_validation_session | 応答が返した開始済みsessionの同一参照 |
| class_count | 分類対象のクラス件数 |
| estimated_change_span_sample_count | 推定変化区間の標本件数 |
| monkeypatch | pytestの一時置換fixture |
| valid_run_settings_mapping | 既存の正常設定fixture |

他のimportは既存公開symbolを別名なしでそのまま使用する。__init__と__post_init__は既存dataclass/ownerと同じ初期化・値検査の特殊method。
