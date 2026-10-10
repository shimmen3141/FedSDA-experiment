# Research & Design Decisions

## Summary

- **Feature**: `fedsda-prediction-diagnostic-metrics-and-records`
- **Discovery Scope**: Extension（指標と、保存する列を足す）
- **Key Findings**:
  - 旧の予測の診断の計数と列は、新の`SamplePredictionRecord`の、標本ごとのfieldから、全部、数えられる。
  - 最終構成（文脈`switching`）では、旧の`mixture`・`switching`・`accuracy`は、同じ予測。旧の`history_routing_effective_experts`と`history_routing_switching_effective_experts`も、同じ値。
  - 旧の回復の区間の正解の割合（`_routing_window_accuracies`）は、既存の定常精度と、同じ定常の区間を使うが、標本がないときの値が違う（0とNaN）。

## Design Decisions

### Decision: 指標は、数（正解の数、標本数）を持ち、割合・差・比は持たない

- **Selected Approach**: 旧の40の指標のうち、数から計算できるもの（正解の割合、上限との差、上限に対する比）は、新の指標に持たない。持つのは、数と、数からは計算できないもの（区間ごとの正解の割合、クラスの間のまとめ、実効モデル数の平均、重みの更新の回数）。
- **Rationale**: 同じ情報を、2つ持たない（応答の指標のspecと同じ方針）。旧の結果の変換は、数から計算する。

### Decision: clientとクラスごとの表は、作らない

- **Selected Approach**: 標本ごとの表へ、観測したクラスと、正否の列を足す。旧の`routing_class_*`は、そこから数える。
- **Rationale**: 標本ごとの列があれば、クラス別のほか、概念別・時期別の集計も、後から作れる。

### Decision: 区間ごとの正解の割合は、新しい関数にする

- **Selected Approach**: `calculate_stable_and_recovery_period_accuracies`を足す。既存の定常精度の関数は、変えない。
- **Rationale**: 標本がないときの値（0とNaN）が、旧の2つの指標で違う。既存の指標の値を、変えない。

## Risks & Mitigations

- 浮動小数点の和の順（実効モデル数の平均、クラスの間の平均と標準偏差）で、最後の桁が変わりうる — 旧と同じ順・同じ式で計算し、`==`で照合する。
- 旧の指標が多く、対応の写し間違いが起きる — 対応表の取り違えの検査（実行した全部の条件で、旧の値が同じになる対を数える）を、このspecでも行う。
