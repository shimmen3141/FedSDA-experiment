# Research & Design Decisions

## Summary

- **Feature**: `fedsda-cross-evaluation-records-and-metrics`
- **Discovery Scope**: Extension（指標と、保存する表を足す）
- **Key Findings**:
  - 旧のクロス評価の診断の記録（`cross_evaluation_diagnostics`・`cross_evaluation_class_diagnostics`・`pair_prediction_diagnostics`）は、新の`ClientCrossEvaluationRecord`の1件に、まとまっている。
  - 最終構成（クラスタリングの判定が`class_functional_confidence`）は、クラスごとの正誤を集める。

## Design Decisions

### Decision: 旧の`model_pair_*`の配列に当たる表は、作らない

- **Context**: 旧は、同じ評価を、2つの一覧（全部の評価と、正誤を比べた評価）に入れて、別々の配列として保存する。
- **Selected Approach**: 新は、clientの評価ごとの表だけを持つ。旧の`model_pair_*`は、その表の、正誤を比べた行。旧の結果の変換は、`cross_evaluation_*`から作れる。
- **Rationale**: 同じ情報を、2つ持たない。

### Decision: 指標は、評価の層の関数で、記録の列から直接作る

- **Selected Approach**: `evaluation/`の関数が、`ClientCrossEvaluationRecord`の列を受け取る（記録の型と、同じ層）。`runtime/`の導出のmoduleは、足さない。

## Risks & Mitigations

- 旧は、標本数の合計が0のとき、評価の数も0と返す — 同じにして、testで確かめる。
