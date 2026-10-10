# Research & Design Decisions

## Summary

- **Feature**: `candidate-validation-decision-record-retention`
- **Discovery Scope**: Extension（clientへ、記録のownerを1つ足す）
- **Key Findings**:
  - 判定記録の2つの型と、その生成は、移植済み。足りないのは、保持だけ。
  - 判定記録は、clientの2つの操作の戻り値から読める（標本1件の処理の結果の`held_validation_advance.validation_progress.completed_validation.decision_record`、終端の回収の結果の`incomplete_validation_finalization.decision_record`）。下位の関数の引数を変えずに、clientが保持へ足せる。
  - 判定記録と、旧の候補の判定の項目の対応は、上流のtest（test_post_alarm_candidate_loss_evaluation.py、test_incomplete_post_alarm_candidate_validation_finalization.py）にある。

## Research Log

### 旧の候補の判定の記録

- **Sources Consulted**: federated_drift_experiment/clients/fedsda.py（`provisional_model_decisions`へ足す4か所: 270行・367行・619行・681行）、provisional_model.py（`ProvisionalModelDecision`、`disjoint_validation_rejection_reason`）、metrics.py（`compute_metrics`の候補の判定の指標）、experiment.py（`_save_raw_run`の`provisional_*`の列）、tests/test_proposed_regression.py（goldenの比較項目）。
- **Findings**:
  - 最終構成（`forward_persistent`）で通るのは、候補検証の確定（270行。`validation_source="forward"`、`resolution_position`は確定した標本の位置）と、終端の回収（367行。理由`insufficient_forward_data`、損失はNaN、参照モデルなし）。619行・681行は、時系列holdoutの方式（`_spawn_validated_provisional_model`）で、最終構成では通らない。
  - 確定の判定の理由は、保有モデルの再利用・現行の維持（`alternative_reference_refit`・`current_reference_refit`）、2つの区間の判定（`first_interval`・`second_interval`・`first_and_second`・`accepted`）。
- **Implications**: 新の2つの判定記録の型が、この2か所に対応する。保持の順は、旧のlistの順（起きた順）。

### 新の判定記録の流れ

- **Sources Consulted**: runtime/post_alarm_candidate_validation_progress.py（`PostAlarmCandidateValidationCompletion.decision_record`）、runtime/incomplete_post_alarm_candidate_validation_finalization.py、runtime/held_candidate_validation_progress.py（適応記録を足して、sessionを外す）、runtime/observed_sample_processing.py（`ObservedSampleProcessing.held_validation_advance`）、runtime/fedsda_run_client.py（`process_observed_sample`・`finalize_incomplete_candidate_validation`、`FedsdaRunClientOwners`、`assemble_fedsda_run_client`）。
- **Findings**: 判定記録は、適応記録へ写された後、戻り値として、clientの操作まで返る。

## Design Decisions

### Decision: clientの操作が、戻り値の判定記録を、保持へ足す

- **Alternatives Considered**: (1)保持中の候補検証の進行（`advance_held_candidate_validation`ほか）へ、ownerの引数を足して、適応記録と同じ場所で足す——標本1件の処理と、その下の関数の引数が変わり、それらのtest（引数の一覧、拒否、原子性）を、広く直す必要がある。(2)適応記録へ、理由の項目を足す——適応記録は、手法に依存しない記録で、候補検証の理由は、その一部の結果にしかない。
- **Selected Approach**: `FedsdaRunClient`の2つの操作が、下位の処理が成功して返った後に、戻り値の判定記録を、ownerへ足す。
- **Rationale**: 下位の関数と、そのtestを変えない。判定記録は、すでに検査済みの不変の値。
- **Trade-offs**: 標本1件の処理が、候補検証の確定の後の段（損失の監視、警報の処理、学習）で失敗すると、適応記録は足されているが、判定記録は足されない。失敗したrunは、実行の枠が中断し、結果を使わないので、許容する（設計に書く）。

### Decision: ownerは、手法の層（候補の選択）に置く

- **Rationale**: 保持する型（2つの判定記録）が、FedSDAの候補の選択のmoduleにある。手法に依存しない記録の層（evaluation）から、手法の型へ依存させない。

### Synthesis

- **Build vs. Adopt**: 新しいmoduleは、ownerの1つ。
- **Simplification**: 判定記録の型は、変えない。旧の項目への写しは、次のspec（列の導出）で行う。本specのtestは、対応を、testの中で照合する。

## Risks & Mitigations

- 保持の追加が、既存の状態の読取り（testの読取りの関数）に入らず、比較から漏れる — clientの全状態の新旧照合のhelperへ、実旧との照合を足す（全条件で照合される）。

## References

- [post-alarm-candidate-validation関連のspec](../)（判定記録の生成）、[fedsda-run-client-assembly](../fedsda-run-client-assembly/)（clientの組立て）、[evaluation-concept-diagnostic-delivery](../evaluation-concept-diagnostic-delivery/)（不干渉のtest）。
