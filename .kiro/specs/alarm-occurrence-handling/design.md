# 警報1回ぶんの処理の接続 — 設計 revision1

## 1. Boundary Commitments

- runtimeの関数`handle_alarm_occurrence`が、警報が起きた標本での処理を、完了済みの5つの部品の呼出しとして並べる。新しいownerは作らない。
- 5つの部品（以下「段」）: (1)`respond_to_alarm_with_buffered_samples`、(2)`complete_alarm_buffer_response`、(3)`record_completed_alarm_response`、(4)`apply_alarm_response_to_validation_session_holder`、(5)`notify_diagnostics_of_training_assignment_change`。
- 結果は不変record`AlarmOccurrenceHandling`（完了した応答の情報と、追加した適応記録）。

Out of Boundary: 警報の検出、推定変化点・推定区間長の算出、保留標本の組立、検出位置の記録、保留中の学習更新の消化、予測側の警報hook、検出episodeの制御、候補検証の進行・確定とその診断通知、警報のない標本の処理、標本1件の処理全体、設定登録、新client・全体run。

## 2. 旧処理との対応（`federated_drift_experiment/clients/fedsda.py`）

| 旧 | 本spec |
| --- | --- |
| `process_one_step`の警報分岐（483〜509行）のうち`_resolve_drift(sample_idx=idx, estimated_start=..., episode_id=...)`の呼出し | `handle_alarm_occurrence`の1回の呼出し |
| `_resolve_drift`（737〜907行）: 候補検証中なら全件吸収、そうでなければ区間の分割・旧側の吸収・不足の判定・再利用評価・切替または候補検証の開始 | 段(1)（既存）。進行中のsessionは引数でなく保持から与える（旧は`self._forward_validation`を読む） |
| `_resolve_drift`内の`_set_local_current_model`→`_on_local_model_change`（84〜89行、最終構成では2052行のAdaHedge再始動） | 段(5)（既存）。段(1)が返す帰属変更を渡す |
| `_begin_forward_validation`の最後の`self._forward_validation = ...` | 段(4)（既存） |
| `_resolve_drift`末尾の`_record_adaptation_event`、`local_switch_positions`、`reuse_selection_counts` | 段(3)（既存） |
| `_resolve_drift`末尾の`_reset_drift_detectors`→`buffer.clear()`（不足のときはclearしない） | 段(2)（既存） |
| `flush_pending_updates`、`detected_event_positions`ほかの位置の記録、`_on_drift_alarm`、`_on_drift_resolution`、`detection_episodes` | 範囲外 |

旧と順序が違う点: 旧は、帰属の切替の直後（吸収・イベント記録・検出器resetより前）に再始動hookを呼び、sessionの代入をイベント記録より前に行う。本specは、応答→完了→記録→保持→通知の順にする。記録は完了処理の結果を入力とするので完了より後になる。保持と診断証拠は、他の段が読まず、他の段の更新も読まない（保持が読まれるのは段(1)の入力として1回だけで、段(4)より前）。成功時の最終状態が旧と一致することを、実旧との対照testで示す。

## 3. 契約

`handle_alarm_occurrence(*, validation_session_holder, adaptation_record_store, diagnostic_evidence_collection, loss_change_monitor, alarm_sample_index, <段(1)の引数からactive_validation_sessionとproposal_sample_indexを除いたもの>) -> AlarmOccurrenceHandling`

全引数keyword-only・既定値なし。段へ渡す値:

| 段 | 渡す値 |
| --- | --- |
| (1) 応答 | `active_validation_session` = `validation_session_holder.held_validation_session`、`proposal_sample_index` = `alarm_sample_index`、他は受け取った引数そのもの |
| (2) 完了 | 段(1)の結果、`alarm_sample_index`、`estimated_change_point_sample_index`、`detection_episode_id`、`current_training_model_assignment`、`loss_statistics_store`、`loss_change_monitor`、`pending_training_assignment_buffer` |
| (3) 記録 | 段(2)の結果、`detector_name`、`adaptation_record_store` |
| (4) 保持 | 段(1)の結果、`validation_session_holder` |
| (5) 通知 | 段(1)の結果が区間解決を持てばその`training_model_assignment_change`、持たなければNone。`diagnostic_evidence_collection` |

`AlarmOccurrenceHandling`はfrozen/kw_only。field `alarm_response_completion`（段(2)の結果。応答はその中の`alarm_buffer_response`）と`adaptation_record`（段(3)の結果）。本moduleの関数だけが作り、下流の入力として受け取るAPIは今回ないので、constructorの追加検査は設けない。

提案位置を警報位置と同じにする根拠: 旧は`_resolve_drift`の`sample_idx`を、イベントの位置と`_begin_forward_validation`の提案位置の両方に使う。

## 4. 検査と処理順

| 検査（例外） | 位置 | 値の出所と、段(1)より前に置く理由 |
| --- | --- | --- |
| 5つのownerがexact型（TypeError）: `validation_session_holder`、`adaptation_record_store`、`diagnostic_evidence_collection`、`loss_change_monitor`、`pending_training_assignment_buffer` | 段(1)より前 | 引数。先の4つは段(2)〜(5)だけが検査する。保留位置は段(1)も検査するが、下の最終観測位置の検査で本処理が読むので、読む前に確かめる。段(2)が検査する残りのowner（現在の学習帰属、損失統計）は、段(1)がどの経路でも自分の更新より前にexact型を検査する（応答自身と、その中の吸収・区間の準備の冒頭）ので、重ねて検査しない |
| 警報位置・検出器名・推定変化点・episode IDが適応記録の規則に合う（TypeError/ValueError） | 段(1)より前 | 引数。段(1)は、候補検証中と不足の経路でこれらを検査しない。段(2)が位置・推定変化点・episode IDを、段(3)が検出器名を、段(1)の更新の後で検査する。規則を複製せず、同じ値で適応記録（`AdaptationRecord`）を1つ組み立てて検査させ、保存しない（結果種別は帰属を変えない値、モデルIDは同じ値を仮に入れる） |
| 警報位置が保留位置の最終観測位置と一致（ValueError） | 段(1)より前 | 引数とownerの現在の状態。段(2)が段(1)の更新の後で検査する。段(1)は最終観測位置を変えない |
| 段(1)の引数の検査 | 段(1)の中（その更新より前） | 既存の部品の契約 |
| 段(2)〜(5)の残りの検査（応答と保留位置の対応、応答と帰属の対応、基準平均の範囲、記録・保持・通知の入力） | 各段の中 | 入力は直前の段が作った値と、上で確かめたowner。上流の契約が守られている限り拒否されない |

保持の前提（段(4)）が満たされる理由: 段(1)へ渡す進行中のsessionは保持から読んだ値で、段(1)は、sessionを受け取れば候補検証中の応答（同じsession）を、受け取らなければそれ以外の応答を返す。保持は段(1)〜(3)の間に変わらない。

処理順: (a)ownerの型検査、(b)記録の規則による値の検査、(c)最終観測位置の検査、(d)段(1)〜(5)、(e)結果を返す。(a)〜(c)は読取りだけ。

部分更新について: 段(1)が更新を始めた後に失敗した場合の巻き戻しは、既存契約のとおり行わない。段(2)以降が拒否するのは上流の出力が上流自身の契約に反するときだけで、その場合は先行する段の更新が残り、後の段は実行しない（要求2.4）。

既知の限界（引き継ぐもの）: 候補検証の進行は、標本位置が提案位置より後であることを検査しない（held-candidate-validation-progressの設計4節）。本specは提案位置を「保留位置の最終観測位置と一致する警報位置」に固定するので、以後の標本位置が保留位置の規則（最終観測位置より後だけを受け入れる）に従う限り、提案位置より前の位置で進行することはない。その保証は標本1件の処理全体のspecで扱う。

## 5. Allowed Dependencies

`runtime/alarm_occurrence_handling.py`: `dataclasses.dataclass`、`random.Random`（型注釈）、`AdaHedgeDiagnosticEvidenceCollection`、`AdaptationRecord`、`AdaptationRecordStore`、`ModelEvaluationSampleStore`、`ModelAndClassLossStatisticsStore`、`ResidualAdapterClassifier`、`CandidateEpochTrainingSettings`、`CurrentTrainingModelAssignment`、`HeldModelTrainingStateRegistry`、`IndexedObservedTrainingSample`、`ModelTrainingAndAssignmentCountsStore`、`ModelTrainingSampleStore`、`AdamParameterOptimizerSettings`、`SgdParameterOptimizerSettings`、`CandidateModelTrainingAndAcceptanceSettings`、`CandidateParameterInitializationSettings`、`OverallAndTrueClassLossMonitor`、`PendingTrainingAssignmentBuffer`、`record_completed_alarm_response`、`respond_to_alarm_with_buffered_samples`、`AlarmResponseCompletion`、`complete_alarm_buffer_response`、`CandidateValidationSessionHolder`、`apply_alarm_response_to_validation_session_holder`、`notify_diagnostics_of_training_assignment_change`の27 symbol。

exact集合をAST guardへ登録する（注入契約testのREDの後）。旧実装・private・module全体・star・子moduleのimport、および候補検証の進行・確定・終端回収、応答の型（`AlarmBufferResponse`）、帰属変更の型のimportは拒否する。

## 6. Revalidation Triggers

5つの段の引数・戻り値・検査の位置、応答の結果5値、適応記録のfieldの規則、保持の規則、保留位置の最終観測位置の意味が変わったら、本specの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/runtime/alarm_occurrence_handling.py | 新規。関数1つと結果record 1つ |
| tests/refactoring/test_alarm_occurrence_handling.py | 新規 |
| tests/refactoring/test_single_run_dependency_boundaries.py | exact集合と注入契約test |

## 8. Testing Strategy / Requirements Traceability

oracleは実旧clientの`_resolve_drift`（イベント記録・検出器reset・FIFO clearも差し替えない）と、実旧の帰属変更hook。完了済みspecのhelperを再利用する: `test_alarm_response_completion.py`の`build_response_completion_oracle`・`run_legacy_alarm_with_real_completion`・`assert_response_completion_matches_legacy`、`test_alarm_buffer_response.py`の`assert_buffer_response_matches_legacy`、`test_adahedge_diagnostic_evidence.py`の`assert_adahedge_matches_legacy`。上流のoracleのclientは帰属変更を記録するだけのhookを持つので、その記録を残したまま、実旧`RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change`（最終構成の再始動）を同じclientに対して実行させる。新しく使う旧メソッドはこの1つ。

| 要求 | 証拠 |
| --- | --- |
| 1.1, 1.3 | 5つの段を記録用wrapperで包み、呼出しの順が5段の順で各1回、段の間で渡る値が直前の段の結果そのもの（同一object）、他の引数が受け取った引数そのものであること |
| 1.2 | 同じtestで、応答へ渡る進行中のsessionがNone（保持が空）、提案位置が警報位置。候補検証中の条件（下）で、保持中のsessionが応答へ渡ること |
| 1.4 | 戻り値のfieldが段(2)・(3)の結果そのもの。recordがfrozen・kw_only・2 field |
| 1.5 | 依存のexact集合に候補検証の進行・確定とepisodeの制御がない。対照testでepisode IDが実旧イベントと一致 |
| 2.1 | 5つのowner×（別の型、派生型）の10条件で、5段のどれも呼ばれず（呼ばれたら失敗するmock）、監視・保留位置・記録・保持・診断が不変 |
| 2.2 | 12条件（警報位置がbool・float・負・最終観測位置の前後、推定変化点が負・bool、episode IDが負・float、検出器名がNone・str派生型・空白）で同じ |
| 2.3 | 設計4節の範囲のとおり、段(1)の引数の検査は上流の既存testに任せ、重ねて網羅しない。引数の集合が「応答の引数−2＋5」で全てkeyword-onlyであること |
| 2.4 | 5つの段のそれぞれを失敗させ、後の段が呼ばれないこと、記録の件数が失敗した段の位置に対応すること |
| 3.1 | 2/4class×警報応答5種類: 応答と完了を既存helperで実旧と照合、記録を実旧イベント・切替位置・再利用件数と照合、保持を実旧のsession属性と照合、診断証拠を実旧のAdaHedgeと照合（再始動は他モデルの再利用のときだけ1回）、torch乱数が実旧と一致。候補検証中は、1回目の警報で保持させたsessionを2回目の警報が保持から読む |
| 3.2 | 実source変異（検査を応答の後へ移す変異を含む）、exact AST、新実装だけのfresh CPU、全pytest、Ruff・Pyright |

未検証として残す: 候補検証の確定に伴う診断通知、標本1件の処理全体（応答の前後に標本を観測しない保証、標本位置の連続性を含む）、予測側の警報hook、検出位置の記録、新全体runのgolden一致。
