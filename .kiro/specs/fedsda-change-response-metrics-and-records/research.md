# Research & Design Decisions

## Summary

- **Feature**: `fedsda-change-response-metrics-and-records`
- **Discovery Scope**: Extension（指標と、保存する列を足す。保存の形式の名前を直す）
- **Key Findings**:
  - 旧の`compute_metrics`（federated_drift_experiment/metrics.py）の入力は、clientの、切替の位置、警報の位置、推定した変化点、適応の記録、候補の判定の記録、再利用の選択の計数。新の記録（`AdaptationRecordStore`、`LossChangeAlarmRecordStore`、`CandidateValidationDecisionRecordStore`）に、全部ある。
  - 標本ごとの検出器の値（log e）は、`LossChangeAlarmRecordSnapshot.monitored_log_e_values`にある。
  - 旧の判定の記録の、損失の差（`full_margin`・`recent_margin`）、確定までの遅れ（`resolution_delay`）、参照の損失の超過（`reference_excess`）は、記録のfieldからの計算（provisional_model.py）。

## Design Decisions

### Decision: 保存する表・列の名前を、時点と意味が分かる名前へ直す

- **Context**: ユーザーの指摘（2026-10-11）。表`round`は、簡潔すぎる。短さより、分かりやすさと、実態との一致を優先する。
- **Selected Approach**: design.mdの「名前の見直し」の表。直すのは、どのラウンドか・どの時点か・どの標本かが、名前から分からなかったものだけ。記録の型とfieldの名前を、そのまま使っている列は、変えない。
- **Rationale**: 保存した結果を使う既存のfileが、まだないので、いま直すのが、いちばん安い。形式の版は、1のまま。

### Decision: 応答の指標は、1つのまとまりとして、指標へ足す

- **Selected Approach**: `FedsdaChangeResponseMetrics`（切替の時機、警報の検出、変化点の推定、適応の区分、候補の判定）を、`FedsdaRunMetrics`の1つのfieldにする。
- **Rationale**: 既存の指標の型の、平らなfieldを、数十個増やさない。まとまりごとに、意味が分かる。

### Decision: 旧の指標のうち、ほかの値から計算できるものは、持たない

- **Selected Approach**: 見逃し率（1−再現率）、誤検出率（1−適合率。検出が0のときは0）、真の変更の数・対応した数・対応しなかった数（検出の指標の計数から）、棄却の理由のうち「参照の再適合」の合計（現行と別モデルの和）は、新の指標に持たない。旧の値との照合は、testの対応で行う。
- **Rationale**: 同じ情報を、2つ持たない。

### Synthesis

- **Build vs. Adopt**: 対応づけの規則は、既存の計算（`count_events_matched_to_concept_changes`）と同じ。対応づいた位置と遅れを返す関数を足し、既存の関数は、変えない。

## Risks & Mitigations

- 指標の項目が多く、対応の写し間違いが起きる — 実旧の`compute_metrics`の戻り値の、全部のキーについて、新→旧の対応があることを、testで確かめる（対応のないキーを、明示の一覧にする）。
