# 命名表: candidate-validation-decision-record-retention

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module・class

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `candidate_validation_decision_record_store` | `methods/fedsda/candidate_model_selection/` | 下のclassのmodule |
| `CandidateValidationDecisionRecordStore` | 同上 | 候補検証の判定記録（確定と、未完了の終端回収）を、起きた順に保持する。適応記録の`AdaptationRecordStore`とは、保持するものが違う（判定の根拠。手法に固有） |

## メソッド

| 名前 | 役割 |
|---|---|
| `append_candidate_validation_decision_record` | 判定記録を1件、末尾へ足す |
| `snapshot_candidate_validation_decision_records` | 足した順の一覧を返す（既存の`snapshot_…`と同じく、後からの追加で変わらない読取り） |

## 判断が必要な点

- 「candidate validation decision record」は、既存の2つの型（`PostAlarmCandidateValidationDecisionRecord`、`IncompletePostAlarmCandidateValidationDecisionRecord`）の共通の語。ownerは両方を持つので、`PostAlarm`を付けない。
