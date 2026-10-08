# 設計 revision3

## 1. Boundary Commitments

- 適応記録・切替位置・再利用計数の状態をAdaptationRecordStoreが所有する。
- evaluation/model_evaluation_sample_store.pyと同様に評価へ渡す履歴を保持するownerで、適応の判断や学習状態は所有しない。
- runtimeはAlarmResponseCompletionをAdaptationRecordへ変換してstoreへ一件追加する。evaluationはruntimeをimportしない。
- Out of Boundary: 上流の応答/reset/drainの再実行、候補検証の到達/終端変換、session保持/解除、予測通知、episode操作、保存schema、全client/runの接続。
- 旧の即時作成/分割holdout検証のcreate・create_rejectedは既存AlarmBufferResponseの5結果に含まれず本specの対象外。最終の将来標本検証構成を扱う。
- Allowed Dependencies: evaluationはdataclasses/typingのみ。runtimeは本evaluation moduleとAlarmResponseCompletionのみ。
- Revalidation Triggers: 応答結果種別、帰属IDの意味、完了recordのfield、記録の順序、保存側の対応を変更したら再照合する。

## 2. File Structure Plan

| ファイル | 責務 |
| --- | --- |
| src/federated_learning_experiments/evaluation/adaptation_record_store.py | 不変AdaptationRecord/AdaptationRecordSnapshot、可変AdaptationRecordStore |
| src/federated_learning_experiments/runtime/alarm_adaptation_recording.py | record_completed_alarm_response関数 |
| tests/refactoring/test_alarm_adaptation_recording.py | 既存の完了oracleを使う旧対照、状態不変、順序、拒否、snapshot |
| tests/refactoring/test_single_run_dependency_boundaries.py と既存の両resolver | ASTで検出した実importのexactな依存境界登録 |

## 3. 値とAPI

AdaptationOutcomeは旧actionと互換を持たず、AlarmBufferResponseの5種と同じ明確な文字列を持つLiteralとする。
AdaptationRecord.adaptation_outcomeの型はAdaptationOutcome。上流のstrをruntimeでtyping.cast(AdaptationOutcome, response_outcome)として渡す。cast自体に検査能力はなく、record constructorのbuiltin strと5値検査が更新前に必ず実行される。

AdaptationRecordはfrozen/kw_onlyのdataclass。fieldはalarm_sample_index、detector_name、adaptation_outcome、previous_training_model_id、current_training_model_id、estimated_change_point_sample_index、detection_episode_id。位置はboolでないbuiltin intかつ非負、任意位置は同条件またはNone、モデルIDは負の一時IDも許すbuiltin int、検出器は空白だけではないbuiltin str（内容はtrimせず保存）。結果はbuiltin strかつ5種の一つ。「IDが異なる」と「保有モデル再利用」は必要十分条件で、等しいIDで再利用結果を指定しても拒否する。全fieldをconstructorで検査する。

adaptation_outcomeは適応履歴の結果、response_outcomeは警報応答の返却結果を表す。今回は同じ5値を写すが、後続で候補検証の確定も履歴に加えるため、履歴fieldを警報応答の責務名に固定しない。推定変化点を警報位置以下へ制限する追加検査は上流に合わせて設けない。

AdaptationRecordSnapshotはfrozen/kw_onlyのdataclass。adaptation_records: tuple[AdaptationRecord, ...]、training_model_switch_sample_indices: tuple[int, ...]、alternative_model_reuse_count: int、current_model_fit_count: int。store以外の手動構築を下流入力として受けるAPIは今回なく、snapshot constructorの追加検査は設けない。

AdaptationRecordStoreは引数なしで空状態から開始。append_adaptation_record(*, adaptation_record: AdaptationRecord) -> None、get_state_snapshot() -> AdaptationRecordSnapshot。内部は_adaptation_records・_training_model_switch_sample_indicesのlistと、_alternative_model_reuse_count・_current_model_fit_countの件数を所有し、外部へ可変listを返さない。appendはexact record型を検査し、dataclasses.replaceでfield全検査を再実行した独立copyをvalidated_adaptation_recordへ準備し、このcopyを保存してから位置・件数を更新する。入力recordを後でobject.__setattr__で破壊しても取得済み履歴を変えない。検査を最初の更新より前に完了させる。tuple/listの通常appendに伴うMemoryErrorのロールバックは契約外。時系列の非減少や同じ位置の重複を独自に拒否しない（旧も拒否しない）。呼出側が一完了につき一度記録する。

record_completed_alarm_response(*, alarm_response_completion: AlarmResponseCompletion, detector_name: str, adaptation_record_store: AdaptationRecordStore) -> AdaptationRecord。exact completionとstoreを確認し、completionの既存__post_init__を読取りだけで再実行してから、値をrecordへ写す。recordを完成させ、store.appendへ渡して返す。警報応答record内の手動不正fieldも、completion再検査とrecord検査を経て履歴更新前に拒否される。通知・reset・drain・session更新は呼ばない。

## 4. 旧対応と順序

| 新結果 | 旧action | 切替位置 | 他モデル件数 | 現行適合件数 |
| --- | --- | --- | --- | --- |
| alarm_during_candidate_validation | forward_validation_pending | なし | 0 | 0 |
| alarm_change_interval_too_short | insufficient_data | なし | 0 | 0 |
| alarm_interval_held_model_reused | reuse | 警報位置 | +1 | 0 |
| alarm_interval_current_model_maintained | maintain | なし | 0 | +1 |
| alarm_interval_candidate_validation_started | create_pending | なし | 0 | 0 |

他のfieldは旧AdaptationEventのposition/detector/old_model_id/new_model_id/estimated_change_point/episode_idと一対一。旧は切替位置/計数を解決中に更新し、イベントはreset前に追加する。本部品の入力時には上流のresetが完了済みだが、成功時の記録値・位置順・計数は変わらない。1.1/1.2→3節の値と4節の対応、1.3→storeの条件更新、2.1→全検査をappendより前、2.2→tuple snapshot、2.3→記録のみの依存、3.1→実旧oracle、3.2→全回帰と境界外。

## 5. 検証

既存build_response_completion_oracle、run_legacy_alarm_with_real_completionと実上流respond/completeを使い、2/4classと5結果で全field/件数/位置を実旧へ対応付ける。候補検証中は開始後の同sessionを再入力する既存経路を使う。複数イベントの順・snapshot保持、bool ID/空検出器/不明結果/不整合ID/不正completion/storeの拒否と拒否前後の全snapshotを確認する。検査を更新の後へ移す変異、切替位置・件数・イベントfieldの破壊、省略、乱数消費を検出する。fresh CPUは旧/testをimportせず、新上流の成功完了recordを記録する。全pytestは主担当、対象と境界test/品質は独立担当。新runのgolden接続完了とは区別する。

typing.get_args(AdaptationOutcome)と上流ALARM_BUFFER_RESPONSE_OUTCOMESの集合一致をtestで確認する。再利用かつ同じIDの拒否、入力recordの後続破壊から保存copyの独立性も確認する。両resolverはtests/refactoring/test_single_run_dependency_boundaries.pyのresolve_imported_module_namesとcollect_dependency_boundary_violations内のexact symbol解決を指す。dependency_is_allowedの許可集合は実importだけ（evaluation: dataclasses.dataclass/replace、typing.Literal。runtime: AdaptationRecord/AdaptationRecordStore/AlarmResponseCompletion）とし、注入契約testを先に追加して拒否/受理のREDを取る。

runtimeは上記にtyping.castとAdaptationOutcomeを追加する。collect_dependency_boundary_violationsの2つのexact解決対象tupleへ新しい2module pathを登録し、dependency_is_allowedへexact集合の分岐を追加する。MemoryErrorは不正入力ではなく実行資源の不足であり、要求2.1の入力拒否の例外とはしない。記録関数の拒否時不変は呼出し時点以降の状態を対象とする。新client組立の前検査には検出器名の検査を含め、上流が更新した後で記録名が不正だと判明しないよう後続specへ申し送る。episode IDは位置ではなく非負の識別子で、任意位置と同じ数値検査を持つ。
