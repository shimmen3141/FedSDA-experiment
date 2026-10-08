# 候補検証の適応記録 — 設計 revision2

## 1. Boundary Commitments

- 既存`AdaptationRecordStore`（evaluation）が履歴・切替位置・2種類の件数を所有し続ける。本specは適応結果を5値増やし、学習帰属が変わる結果の集合を1つの定数にまとめる。
- runtimeの新moduleが、`PostAlarmCandidateValidationCompletion`と`IncompletePostAlarmCandidateValidationFinalization`を`AdaptationRecord`へ変換してstoreへ1件追加する。evaluationはruntimeをimportしない。
- 既存`record_completed_alarm_response`の変換規則は変えない（下のfield名の変更だけ）。

Out of Boundary: 確定・回収の再実行、session保持/解除、学習帰属変更の通知と保存診断、episode操作、判定記録の一覧、保存schema、新client・全体run。

## 2. 既存recordのfield名の変更

`AdaptationRecord.alarm_sample_index`を`adaptation_sample_index`へ改める。旧`AdaptationEvent.position`は、警報応答では警報位置だが、到達時の確定では確定位置、終端回収では終端位置である。候補検証の記録を`alarm_sample_index`へ入れると、警報位置（提案位置）と取り違える名前になる。alarm-adaptation-recordingの設計は結果fieldを警報応答の責務名に固定しない方針だったが、位置fieldは警報の名前のままだった。

影響: `evaluation/adaptation_record_store.py`、`runtime/alarm_adaptation_recording.py`（keyword 1箇所）、`tests/refactoring/test_alarm_adaptation_recording.py`（11箇所の機械的な置換と、適応結果の集合一致のassertを「先頭5値が警報応答の5結果」へ改める）。値・検査・拒否の規則は変えない。旧名のaliasは残さない（worktreeの方針）。完了済みspecの文書は書き換えず、そのREADMEへ本specへの参照を1行足す。

## 3. 値と規則

`AdaptationOutcome`（Literal）へ次を追加する。既存5値の後に、この順で並べる。

| 新しい適応結果 | 元の値 | 旧action | 帰属IDの変更 | 切替位置 | 2種類の件数 |
| --- | --- | --- | --- | --- | --- |
| post_alarm_validation_candidate_adopted | 確定結果 candidate_adopted_as_new_model | create | あり | 確定位置 | 変えない |
| post_alarm_validation_held_model_reused | 確定結果 held_reference_model_reused | reuse | あり | 確定位置 | 変えない |
| post_alarm_validation_current_model_maintained | 確定結果 current_model_maintained | maintain | なし | なし | 変えない |
| post_alarm_validation_candidate_rejected | 確定結果 candidate_rejected | create_rejected | なし | なし | 変えない |
| post_alarm_validation_incomplete_candidate_rejected | 終端回収 | create_rejected | なし | なし | 変えない |

- 公開定数`TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES`（tuple）: `alarm_interval_held_model_reused`と上の「あり」の2値。recordの規則「変更前後のIDが異なる ⇔ 結果がこの集合に属する」と、storeの切替位置の追加が同じ定数を使う。既存の規則（警報時の再利用だけIDが異なる）の拡張で、警報応答5結果に対する判定は変わらない。
- recordの結果値の検査は`typing.get_args(AdaptationOutcome)`との照合にする（Literalと検査用の列挙の二重管理をやめる）。
- storeの2種類の件数は従来どおり`alarm_interval_held_model_reused`と`alarm_interval_current_model_maintained`だけで増える。旧`reuse_selection_counts`は`_resolve_drift`の中でだけ加算され、`_finalize_forward_validation`と`finalize_incomplete_forward_validation`では加算されない（research.md）。
- 旧は到達時の棄却と未完了の棄却をどちらも`create_rejected`と記録する。新は判定記録の型も別なので結果値を分け、旧actionへは多対一で対応させる（警報応答の結果値も旧actionと別の語彙である）。

## 4. 契約と処理順

`record_completed_candidate_validation(*, validation_completion, adaptation_record_store) -> AdaptationRecord`
`record_incomplete_candidate_validation_finalization(*, incomplete_validation_finalization, adaptation_record_store) -> AdaptationRecord`

どちらも、(1)型と対応の検査、(2)`AdaptationRecord`の組立（constructorが値を検査）、(3)`append_adaptation_record`（exact型の確認と、全fieldを再検査したcopyの保存）、(4)組み立てたrecordを返す、の順。更新は(3)だけで、(1)(2)の例外はすべてその前に出る。storeの拒否も内部の最初の更新より前（既存契約）。

上流の完了情報・判定記録・確定結果・変更記録は、constructorでfieldをほとんど検査しない素のdataclassである。記録へ写す値と、写す値を決める比較に使う値を次のとおり確かめる。

| 検査（到達時の確定） | 例外 | 位置 | 値の出所と上流の検査 |
| --- | --- | --- | --- |
| 完了情報・storeがexact型 | TypeError | 更新前 | 引数 |
| 判定記録がexact `PostAlarmCandidateValidationDecisionRecord` | TypeError | 更新前 | 完了情報のfield。上流は検査しない |
| 確定結果がexact `PostAlarmCandidateValidationResolution` | TypeError | 更新前 | 同上 |
| 結果種別がbuiltin strで、対応表の4値のいずれか | TypeError / ValueError | 更新前 | 確定結果のconstructorは集合への所属だけを見る。手で差し替えた値は通りうる |
| 変更前IDと保留標本の帰属先IDがbuiltin int | TypeError | 更新前 | 上流は検査しない（`True == 1`で以降の比較を通過しうるので、比較より前に型を見る） |
| 変更記録がNoneまたはexact `TrainingModelAssignmentChange`で、両IDがbuiltin int | TypeError | 更新前 | 上流は検査しない |
| 変更記録の変更前IDが完了情報の変更前IDと等しい | ValueError | 更新前 | — |
| 変更記録の変更後IDが変更前IDと異なる | ValueError | 更新前 | 上流の帰属ownerは実際に変わったときだけ変更記録を返すが、型は前後が同じ記録の構築を妨げない |
| 保留標本の帰属先IDが確定後の学習帰属ID（変更記録があればその変更後ID、なければ変更前ID）と等しい | ValueError | 更新前 | 既存の確定は4結果とも保留標本を確定後の学習帰属モデルへ帰属させる |
| 確定位置・検出器名・推定変化点・episode IDの型と値、「IDが異なる ⇔ 帰属が変わる結果」 | TypeError / ValueError | 更新前 | `AdaptationRecord`のconstructor。採用・再利用なのに変更記録がない、維持・棄却なのに変更記録がある入力はここで拒否される |

上の検査により、結果種別ごとに受理する組合せは次のとおりに決まる（revision1は「変更後IDが変更前IDと異なる」検査がなく、維持・棄却と前後が同じ変更記録の組合せが通過した。設計レビューの指摘で追加した）。

| 確定結果 | 変更記録 | 学習帰属ID | 保留標本の帰属先ID |
| --- | --- | --- | --- |
| candidate_adopted_as_new_model、held_reference_model_reused | 必須。変更前IDは完了情報の変更前IDと同じ、変更後IDはそれと異なる | 変更前→変更後 | 変更後ID |
| current_model_maintained、candidate_rejected | Noneだけ | 変更前のまま | 変更前ID |

- 変更記録があれば前後のIDは必ず異なる（上の検査）ので、recordの規則「IDが異なる ⇔ 帰属が変わる結果」により、維持・棄却では変更記録を持つ入力がすべて拒否される。
- 変更記録がなければ前後のIDは同じなので、同じ規則により、採用・再利用では変更記録を持たない入力がすべて拒否される。

終端回収は、結果とstoreのexact型、判定記録がexact `IncompletePostAlarmCandidateValidationDecisionRecord`であることを確かめ、残り（終端位置、検出器名、学習帰属ID、推定変化点、episode ID）はrecordのconstructorが検査する。変更前後のIDには同じ値（回収時点の学習帰属ID）を入れるので、比較に使う値はない。

確定後の学習帰属IDは完了情報から導く（学習帰属ownerを引数に取らない）。到達時の完了情報は確定直後の値を持ち、記録は確定時点の事実を写すためである。

## 5. Allowed Dependencies

- evaluation/adaptation_record_store.py: `dataclasses.dataclass`、`dataclasses.replace`、`typing.Literal`、`typing.get_args`（追加）。
- runtime/candidate_validation_adaptation_recording.py: `AdaptationOutcome`、`AdaptationRecord`、`AdaptationRecordStore`、`TrainingModelAssignmentChange`、`PostAlarmCandidateValidationDecisionRecord`、`IncompletePostAlarmCandidateValidationDecisionRecord`、`IncompletePostAlarmCandidateValidationFinalization`、`PostAlarmCandidateValidationCompletion`、`PostAlarmCandidateValidationResolution`の9 symbol。
- runtime/alarm_adaptation_recording.pyの依存は変えない。
- exact集合をAST guardへ登録する（注入契約testのREDの後）。旧実装・private・module全体・star・子moduleのimportは拒否する。

## 6. Revalidation Triggers

確定結果の4値、完了情報・終端回収の結果・判定記録・変更記録のfield、storeの切替位置と件数の規則、`AdaptationOutcome`の値と順序が変わったら、本specとalarm-adaptation-recordingの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/evaluation/adaptation_record_store.py | 適応結果5値、切替結果の定数、field名、結果値の検査 |
| src/federated_learning_experiments/runtime/alarm_adaptation_recording.py | keyword 1箇所 |
| src/federated_learning_experiments/runtime/candidate_validation_adaptation_recording.py | 新規。2つの記録関数と結果値の対応表 |
| tests/refactoring/test_candidate_validation_adaptation_recording.py | 新規 |
| tests/refactoring/test_alarm_adaptation_recording.py | field名と結果集合のassert |
| tests/refactoring/test_single_run_dependency_boundaries.py | 新moduleのexact集合、evaluationの`typing.get_args` |

## 8. Testing Strategy / Requirements Traceability

oracleは実旧。到達時は既存`build_validation_progress_oracle`と`set_scripted_validation_losses`で、実旧`_observe_forward_validation`（到達時に`_finalize_forward_validation`を実行し、実`AdaptationEvent`と`local_switch_positions`を残す）と新`advance_post_alarm_candidate_validation`を同じ状態から実行する。終端は既存`build_incomplete_validation_finalization_oracle`で、実旧`finalize_incomplete_forward_validation`と新の終端回収を実行する。どちらも完了済みspecで実行できることを確認済みのoracleで、新しい旧メソッドは使わない。

| 要求 | 証拠 |
| --- | --- |
| 1.1, 3.1 | 2/4class×旧4確定条件×保留0/3件の16条件で、記録の全fieldを実旧イベントと、storeの切替位置の増分を実旧`local_switch_positions`と照合 |
| 1.2, 3.1 | 2/4class×観測0/3件×処理済み件数2種（終端位置が提案位置になる場合を含む）の8条件で同じ照合 |
| 1.3 | 全10結果について、切替位置・2種類の件数の増分と「IDが異なる ⇔ 帰属が変わる結果」の双方向の拒否。警報時の再利用1件を先に持つstoreへの追加順 |
| 1.4 | 既存`test_alarm_adaptation_recording.py`の全件（field名だけ置換）。適応結果の先頭5値が警報応答の5結果と一致 |
| 2.1 | 4節の表の各検査を破る入力（到達時20条件、終端8条件、storeの型）で、警報時の記録と正常な記録を先に持つstoreのsnapshotが不変 |
| 2.2 | 記録の前後で3種の乱数の状態全体が不変。記録の後に既存の照合helperで完了情報と上流の全状態を実旧と再照合。依存のexact集合にsession・通知・学習帰属ownerがない |
| 3.2 | 実source変異（検査を更新の後へ移す変異を含む）、exact AST、新実装だけのfresh CPU、旧11・最終3goldenを含む全pytest、Ruff・Pyright |

未検証として残す: session保持/解除、通知と保存診断、episode、新全体runのgolden一致。
