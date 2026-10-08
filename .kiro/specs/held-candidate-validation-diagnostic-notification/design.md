# 候補検証の確定に伴う診断通知 — 設計 revision2

## 1. Boundary Commitments

- 既存`runtime/held_candidate_validation_progress.py`の`advance_held_candidate_validation`へ、引数`diagnostic_evidence_collection`と、確定時の既存`notify_diagnostics_of_training_assignment_change`の呼出しを足す。新しいmodule・owner・結果recordは作らない。結果record `HeldCandidateValidationAdvance`は変えない。
- 同じmoduleの`apply_alarm_response_to_validation_session_holder`の、保持が空でないときの`ValueError`の文言を直す（[NEW-001](../../../docs/research/implementation-findings/new-001-holder-rejection-message-wording.md)）。条件・例外の型・処理は変えない。
- `finalize_held_incomplete_candidate_validation`は変えない。

Out of Boundary: 警報のときの通知（`handle_alarm_occurrence`）、予測側のhook、検出episode、標本1件の処理全体、新client・全体run。

別のmoduleで包む案を採らない理由: 進行の関数はすでに「上流の進行→記録→保持の解除」をつなぐ接続で、通知はその続きの1段である。包む関数を足すと、通知のない進行とある進行の2つの入口ができ、呼出し側が通知を落とせる。

## 2. 旧処理との対応（`federated_drift_experiment/clients/fedsda.py`）

| 旧 | 本spec |
| --- | --- |
| `_finalize_forward_validation`の採用の分岐: `_set_local_current_model(temp_id)`（312行）→`_on_local_model_change`（最終構成では2052行のAdaHedge再始動） | 確定の結果の`training_model_assignment_change`（採用では変更あり）を通知へ渡す |
| 同、再適合した参照の分岐: `_set_local_current_model(requalified_model_id)`（332行）。IDが変わるときだけhook（84〜89行） | 同じ。他モデルの再利用では変更あり、現行の維持では変更なし（Noneまたは同じID）で、通知は再始動しない |
| 同、棄却の分岐: `_set_local_current_model`を呼ばない | 帰属変更はNoneで、通知は何もしない |
| `finalize_incomplete_forward_validation`: `_set_local_current_model`を呼ばない | 終端回収は通知しない（変更なし） |
| 旧の順: 切替とhook→保留標本の扱い→イベント記録→session解除 | 新の順: 上流の進行（切替を含む）→記録→解除→通知 |

順序の違い: 旧は切替の直後に再始動する。新は記録と解除の後に通知する。診断証拠は進行・記録・保持のどれも読まず、通知は診断証拠だけを更新するので、成功時の最終状態は同じ。実旧との対照testで示す。通知を最後にするのは、学習と記録の状態を先に確定させ、警報のとき（`handle_alarm_occurrence`）と同じ「記録・保持の後に通知」に揃えるため。

## 3. 契約

`advance_held_candidate_validation(*, validation_session_holder, adaptation_record_store, diagnostic_evidence_collection, <既存の残りの引数>) -> HeldCandidateValidationAdvance`

- `diagnostic_evidence_collection`: exact `AdaHedgeDiagnosticEvidenceCollection`。
- 確定したとき: 既存の記録→解除の後、`notify_diagnostics_of_training_assignment_change(assignment_change=<確定の結果のtraining_model_assignment_change>, diagnostic_evidence_collection=...)`を1回呼ぶ。
- 未到達・保持なし: 通知を呼ばない（既存の早期return）。

## 4. 検査と処理順

| 検査（例外） | 位置 | 値の出所 |
| --- | --- | --- |
| 保持と記録のownerがexact型（既存。TypeError） | 上流の進行より前 | 引数 |
| 診断証拠のownerがexact型（TypeError。追加） | 上流の進行より前 | 引数。通知も同じ検査を持つが、それは上流の更新の後に呼ばれるので、先に確かめる |
| 上流の進行・記録の検査（既存） | 各部品の中 | 既存の契約 |
| 通知の検査（帰属変更がexact型またはNone、IDがbuiltin int） | 通知の中。上流の更新・記録・解除の後 | 入力は直前に上流の確定が作った値。上流の契約が守られている限り拒否されない |

処理順: (1)ownerの型検査（保持、記録、診断）、(2)既存の進行、(3)未確定なら結果を返す、(4)適応記録、(5)保持の解除、(6)診断通知、(7)結果を返す。

保持がないとき: (1)の後、上流はNoneを受けて何もしない。診断のownerの型は(1)で確かめるので、保持がなくても不正な型は拒否する（要求2.1）。それまで「保持と記録のowner以外の引数を読まない」だった挙動は、「保持・記録・診断のowner以外の引数を読まない」になる。

部分更新: (6)が拒否するのは上流の出力が上流自身の契約に反するときだけで、その場合は(2)(4)(5)の更新が残る。通常の入力では起こらない。

## 5. Allowed Dependencies

`runtime/held_candidate_validation_progress.py`の許可依存を20 symbolから22 symbolへ増やす: 追加は`AdaHedgeDiagnosticEvidenceCollection`と`notify_diagnostics_of_training_assignment_change`。exact集合のAST guardへ登録する（注入契約testのREDの後）。診断証拠の中身（`AdaHedgeDiagnosticEvidence`）、帰属変更の型、`handle_alarm_occurrence`のimportは拒否する。

## 6. Revalidation Triggers

通知の引数と再始動の条件、確定の結果の`training_model_assignment_change`の意味、進行の引数と戻り値が変わったら、本specの対照testを再検証する。

## 7. File Structure Plan

| ファイル | 変更 |
| --- | --- |
| src/federated_learning_experiments/runtime/held_candidate_validation_progress.py | 引数1つ、型検査1つ、通知の呼出し1つ、文言1件、import 2つ |
| tests/refactoring/test_held_candidate_validation_progress.py | 進行の呼出しへ診断のownerを渡す。診断の照合、通知の時点、未到達・保持なしで不変、ownerの型の拒否（派生型を含む）を追加 |
| tests/refactoring/test_single_run_dependency_boundaries.py | exact集合へ2 symbol、注入契約test |

進行の関数を呼ぶ箇所は、srcでは他になく、testでは上の1ファイルだけ（2026-10-09にgrepで確認）。過去specのfresh CPU script（Git管理外）は当時のcommitの証拠で、更新しない。

## 8. Testing Strategy / Requirements Traceability

oracleは、既存の`build_validation_progress_oracle`の実旧client（実旧`_observe_forward_validation`を実行）に、実旧`RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change`を、上流oracleの記録用hookを残したまま実行させるもの（alarm-occurrence-handlingと同じ方法）。新しく使う旧メソッドはない。

| 要求 | 証拠 |
| --- | --- |
| 1.1 | 2/4class×確定4条件: 通知を記録用wrapperで包み、呼出しが1回、渡る帰属変更が確定の結果のものそのもの、通知の時点で記録が追加済み・保持が空 |
| 1.2 | 未到達3標本で診断証拠が不変。保持なしで診断証拠が不変 |
| 1.3 | 終端回収のtestは変更なしで成功（引数に診断のownerがない）。依存の変更は2 symbolの追加だけ |
| 2.1 | 進行×（保持、記録、診断）と終端回収×（保持、記録）の5組×（別の型、派生型）×（保持あり、保持なし）の20条件で、上流を呼ぶ前に拒否（上流を呼んだら失敗するmock）、例外の文言で対象の引数を確認、保持と記録が不変 |
| 2.2 | 既存の拒否12条件が、例外の型と保持の不変のまま成功する。文言の変更はsourceの差分で確認する（文言を照合するtestは足さない） |
| 3.1 | 同じ2/4class×確定4条件: 診断証拠が実旧の再始動hookの後のAdaHedgeと一致、再始動は採用と他モデルの再利用で1回、他は0回 |
| 3.2 | 既存の照合（進行、記録、保持、記録→解除の順）がそのまま成功。実source変異（診断の型検査を上流の後へ移す・保持があるときだけ行う、通知の省略・二重実行・解除や記録の前への移動、帰属変更を渡さない、未到達でも通知する）、exact AST、新実装だけのfresh CPU、全pytest、Ruff・Pyright |

未検証として残す: 標本1件の処理全体、予測側のhook、新全体runのgolden一致（保存する診断の同一性を含む）。
