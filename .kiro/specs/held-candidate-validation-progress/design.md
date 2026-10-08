# 候補検証sessionの保持と進行 — 設計 revision3

## 1. Boundary Commitments

- runtimeの`CandidateValidationSessionHolder`が、進行中の`PostAlarmCandidateValidationSession`を高々1つ所有する。旧`FedSDAClient._forward_validation`属性に対応する。sessionの中身（候補、固定参照、損失収集）は借用で、holderは読まない。
- runtimeの3つの関数が、既存部品と保持・適応記録をつなぐ: 警報応答の反映、標本1件の進行、終端回収。
- 適応記録の変換は既存`record_completed_candidate_validation`・`record_incomplete_candidate_validation_finalization`へ委譲する。

Out of Boundary: 警報応答の実行・完了処理・警報応答の適応記録、通知と保存診断、検出episode、判定記録の一覧、標本ごとのclient進行、設定登録、新client・全体run。

## 2. 旧処理との対応（`federated_drift_experiment/clients/fedsda.py`）

| 旧 | 本spec |
| --- | --- |
| `_forward_validation = None`（初期化） | holderの初期状態は空 |
| `_begin_forward_validation`の最後で`_forward_validation = ForwardValidationSession(...)`（`_resolve_drift`の新規作成の経路だけ） | `apply_alarm_response_to_validation_session_holder`: 結果が候補検証の開始の応答で、応答のsessionを保持 |
| `_resolve_drift`冒頭の`if self._forward_validation is not None:`（全件吸収して終了、sessionはそのまま） | 同関数: 候補検証中の応答では保持を変えない。応答のsessionが保持中のものと同一であることを確かめる |
| `_observe_forward_validation`: sessionがなければ0を返す。観測し、`session.ready`でなければ0、readyなら`_finalize_forward_validation` | `advance_held_candidate_validation`: 既存`advance_post_alarm_candidate_validation`へ保持中のsession（またはNone）を渡す |
| `_finalize_forward_validation`の最後: `_record_adaptation_event(...)`→`_forward_validation = None` | 確定したら`record_completed_candidate_validation`→`release_validation_session`の順 |
| `finalize_incomplete_forward_validation`: sessionがなければ何もしない。回収→イベント記録→`_forward_validation = None` | `finalize_held_incomplete_candidate_validation`: 既存の終端回収→`record_incomplete_candidate_validation_finalization`→解除 |
| `_finalize_forward_validation`最後の`_on_drift_resolution`、`mark_operation`、`_on_local_model_change` | 範囲外（通知・episode） |

## 3. 契約

### holder（`runtime/candidate_validation_session_holder.py`）

- `held_validation_session`（readonly property）: 保持中のsessionまたはNone。
- `hold_validation_session(*, validation_session)`: exact `PostAlarmCandidateValidationSession`でなければTypeError。すでに保持していればValueError（同じsessionでも拒否する。進行中のsessionを黙って置き換えない）。どちらも状態を変えない。
- `release_validation_session()`: 保持中のsessionを外して返す。空ならLookupError。

### 警報応答の反映

`apply_alarm_response_to_validation_session_holder(*, alarm_buffer_response, validation_session_holder) -> None`

| 応答の結果 | 保持の前提 | 保持の変化 |
| --- | --- | --- |
| alarm_during_candidate_validation | 応答のsessionと同一のsessionを保持中 | なし |
| alarm_interval_candidate_validation_started | 空 | 応答のsessionを保持 |
| alarm_change_interval_too_short、alarm_interval_held_model_reused、alarm_interval_current_model_maintained | 空 | なし |

前提に合わなければValueError。入力は完了した応答（`AlarmBufferResponse`）で、完了処理の結果（`AlarmResponseCompletion`）ではない。保持の更新に必要なのは結果種別とsessionだけで、完了処理・記録との前後を本関数は決めない（組立側が決める）。

### 進行と終端回収

`advance_held_candidate_validation(*, validation_session_holder, adaptation_record_store, <既存の進行の引数からvalidation_sessionを除いたもの>) -> HeldCandidateValidationAdvance`
`finalize_held_incomplete_candidate_validation(*, validation_session_holder, adaptation_record_store, <既存の終端回収の引数からvalidation_sessionを除いたもの>) -> HeldIncompleteCandidateValidationFinalization | None`

結果recordはfrozen/kw_only。`HeldCandidateValidationAdvance`は`validation_progress`（既存の進行結果）と`adaptation_record`（確定時だけ）。`HeldIncompleteCandidateValidationFinalization`は`incomplete_validation_finalization`と`adaptation_record`。どちらも本moduleの関数だけが作り、下流の入力として受け取るAPIは今回ないので、constructorの追加検査は設けない。

## 4. 検査と処理順

| 関数 | 検査（例外） | 位置 | 値の出所と上流の検査 |
| --- | --- | --- | --- |
| 反映 | 応答がexact `AlarmBufferResponse`、holderがexact型（TypeError） | 更新前 | 引数 |
| 反映 | 応答の`__post_init__`を再実行（結果種別5値、結果種別とsessionの組。ValueError） | 更新前 | 応答はfrozenだが、手でfieldを差し替えた入力がありうる。再実行で、応答だけを差し替えた入力（開始の応答からsessionを外す、開始でない応答へsessionを足す）を拒否する。区間解決も合わせて差し替えた入力は次の行の検査で拒否する |
| 反映 | sessionを持つのが候補検証中と開始の応答だけであること（ValueError） | 更新前 | 応答の`__post_init__`は、区間解決を持つ応答のsessionを「区間解決が開始したsessionと同一」としか見ず、区間解決の`__post_init__`は結果種別とsessionの組を見ない。応答と区間解決の両方を手で差し替えると、再利用の応答へsessionを足した入力、開始の応答からsessionを外した入力が再実行を通る（Task 1の独立レビューの指摘で追加した検査） |
| 反映 | 上の表の前提（ValueError） | 更新前 | holderの現在の状態 |
| 反映 | `hold_validation_session`の検査（exact session型、空であること） | 状態を変える前 | 前提の検査が空を保証済み。sessionの型は、応答の`__post_init__`が保証するのは候補検証中の応答だけで、開始の応答では区間解決が持つ値のままなので、exact型でなければここでTypeErrorになる（保持は変わらない） |
| 進行・終端 | holderと適応記録のownerがexact型（TypeError） | 上流の呼出しより前 | 引数。上流の進行・終端回収は多くのownerを更新するので、その前に確かめる |
| 進行・終端 | 上流へ渡す引数の検査 | 上流の中（上流の更新より前） | 既存の部品の契約 |
| 進行・終端 | 完了情報・終端回収の結果の検査 | 上流の更新の後、適応記録の追加より前 | 既存の記録処理の契約。入力は直前に上流が作った値なので、上流の契約が守られている限り拒否されない |

処理順（進行）: (1)ownerの型検査、(2)既存の進行、(3)未確定なら結果を返す、(4)確定なら適応記録を追加、(5)保持を解除、(6)結果を返す。終端回収も同じ形で、(2)がNoneを返したら（保持なし）Noneを返す。

部分更新について: (2)の上流が更新を始めた後に失敗した場合の巻き戻しは、上流の既存契約のとおり行わない（保持は変わらないので、sessionは保持されたままになる）。(4)の記録が拒否されるのは上流の出力が上流自身の契約に反するときだけで、通常の入力では起こらない。その場合は上流の更新が残り、保持は解除されない。(5)の解除は、(1)でholderの型を確かめ、(2)へ渡したsessionが保持中のものなので、拒否されない。

既知の限界: 上流の進行は、渡された標本位置が提案位置より後であることを検査しない。提案位置より前の標本位置で要求件数に達すると、上流が確定を適用した後で、記録処理が位置の順序を理由に拒否する（上流の更新が残り、保持は解除されない）。標本位置は本処理が上流へ渡すだけの引数（要求3.3）で、位置の連続性は標本ごとのclient進行が保証する。上流の進行に位置の検査を足すかどうかは、client進行のspecで判断する。

保持がないとき: 既存の進行・終端回収は、sessionがNoneなら他の引数を読まずに何もしない。本処理も(1)の後、そのまま上流へNoneを渡す。

## 5. Allowed Dependencies

- `runtime/candidate_validation_session_holder.py`: `PostAlarmCandidateValidationSession`の1 symbol。
- `runtime/held_candidate_validation_progress.py`: `dataclasses.dataclass`、`torch.Tensor`（引数の型注釈）、`AdaptationRecord`、`AdaptationRecordStore`、`ModelAndClassLossStatisticsStore`、`CurrentTrainingModelAssignment`、`HeldModelTrainingStateRegistry`、`ModelTrainingAndAssignmentCountsStore`、`ModelTrainingSampleStore`、`TemporaryModelIdAllocator`、`CandidateModelTrainingAndAcceptanceSettings`、`PendingModelUploadState`、`AlarmBufferResponse`、`record_completed_candidate_validation`、`record_incomplete_candidate_validation_finalization`、`CandidateValidationSessionHolder`、`IncompletePostAlarmCandidateValidationFinalization`、`finalize_incomplete_post_alarm_candidate_validation`、`PostAlarmCandidateValidationProgress`、`advance_post_alarm_candidate_validation`の20 symbol。
- exact集合をAST guardへ登録する（注入契約testのREDの後）。旧実装・private・module全体・star・子moduleのimportは拒否する。

## 6. Revalidation Triggers

応答の結果5値とsessionの規則、進行・終端回収の引数と戻り値、記録関数の契約、sessionの型が変わったら、本specの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/runtime/candidate_validation_session_holder.py | 新規。holder |
| src/federated_learning_experiments/runtime/held_candidate_validation_progress.py | 新規。3つの関数と2つの結果record |
| tests/refactoring/test_held_candidate_validation_progress.py | 新規 |
| tests/refactoring/test_single_run_dependency_boundaries.py | 2 moduleのexact集合と注入契約test |

## 8. Testing Strategy / Requirements Traceability

oracleは実旧clientの`_forward_validation`属性と適応イベント。完了済みspecのoracleを再利用する: 警報応答5種類は`test_alarm_adaptation_recording.py`の`build_completed_recording_oracle`（実旧`_resolve_drift`を実行済みのclientが得られる）、到達時は`build_validation_progress_oracle`＋`set_scripted_validation_losses`（実旧`_observe_forward_validation`）、終端は`build_incomplete_validation_finalization_oracle`（実旧`finalize_incomplete_forward_validation`）。新しい旧メソッドは使わない。

| 要求 | 証拠 |
| --- | --- |
| 1.1 | holderの保持・解除・拒否（空で解除、保持中に同じ/別のsession、sessionでない値、位置引数、propertyへの代入）と、拒否後の状態不変 |
| 1.2, 4.1 | 2/4class×警報応答5種類で、反映後の保持の有無が実旧`_forward_validation`の有無と一致。保持されるのは応答のsessionそのもの |
| 1.3, 3.2 | 12条件（正式な5値でない結果種別、応答と区間解決の両方を差し替えて、再利用の応答へsessionを足した入力・開始の応答からsessionを外した入力を含む。他は応答が別の型、候補検証中の応答で保持が空/別のsession、開始の応答で別の/同じsessionを保持中、維持・不足の応答でsessionを保持中、開始の応答からsessionを外した入力、維持の応答へsessionを足した入力）で保持が不変。holderの型 |
| 2.1, 4.1 | 2/4class×旧4確定条件: 記録が実旧イベントと一致、保持が空、実旧のsessionもNone、解除の時点で記録が追加済み、進行結果を既存helperで実旧と照合。未到達3標本: 保持と実旧のsessionが維持され、記録は増えない |
| 2.2, 4.1 | 2/4class×観測0/3件: 記録が実旧イベントと一致、保持が空。解除後の再回収は何も変えない（旧も同じ） |
| 2.3 | 保持が空のとき、owner以外の引数を読まずに何も更新しない（記録・乱数が不変） |
| 2.4 | 依存のexact集合に通知・episode・警報応答の記録がない |
| 3.1 | 2関数×2ownerで、型の拒否が上流の呼出しより前（上流を呼んだら失敗するmock）、保持と記録が不変 |
| 3.3 | 設計4節の範囲のとおり、上流の引数の検査は上流の既存testに任せ、重ねて網羅しない |
| 4.2 | 実source変異（検査を上流の呼出しの後へ移す変異を含む）、exact AST、新実装だけのfresh CPU（警報→開始→観測→確定、および終端回収までを1つのholderと記録ownerで実行）、全pytest、Ruff・Pyright |

未検証として残す: 通知と保存診断、episode、標本ごとのclient進行（応答と完了の間に標本を観測しない保証を含む）、新全体runのgolden一致。
