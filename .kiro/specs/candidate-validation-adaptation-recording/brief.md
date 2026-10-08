# 候補検証の適応記録

再開案内の候補「候補検証の適応記録とsession保持、診断通知」のうち、適応記録だけを先に切り出す。

- 本spec: 候補検証の到達時の確定（`PostAlarmCandidateValidationCompletion`）と未完了の終端回収（`IncompletePostAlarmCandidateValidationFinalization`）を、既存の`AdaptationRecordStore`へ記録する。適応結果の値を5つ増やし、切替位置の規則を候補の採用と検証後の再利用へ広げる。
- 後続spec: 候補検証sessionの保持と解除、学習帰属変更の通知と保存診断、検出episode。いずれも新しいownerが要り、client進行の組立と一緒に決める方が形が定まる。

警報応答の記録（alarm-adaptation-recording）は再実装しない。記録の位置を表すfield名だけを、警報位置に限らない名前へ改める。固定旧`748c3aa`とgoldenは変更しない。アルゴリズムの変更はしない。
