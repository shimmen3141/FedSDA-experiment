# Research & Design Decisions

## Summary
- Feature: cross-evaluation-download-deduplication
- Discovery Scope: 既存計数の局所修正。
- LEGACY-018の修正自体はユーザー合意済み。修正順は未回答のままだったため、全構成golden作成前に今回扱うことを主担当が選んだ。

## Research Log
### 既存の例外分岐
runtime/model_cross_evaluation.pyは共有部が空のときだけ毎組の送信を数える。共有部ありの集合を共用すれば抽出/評価を動かさず修正できる。旧servers/clustering.pyのbegin/recordはクロス評価ごとの観測境界になる。
### 構成の照合
test_fedsda_run_metric_derivation.pyの小規模run照合が通信量8項目の一括等値を持つ。独立構成だけ旧転送イベントの重複を減算する対応付けに変更し、比較そのものは残す。MNIST全状態照合は既存testを維持する。

## Design Decisions
独立の設定flagや旧互換計数を増やさず既存集合を利用する。固定旧/goldenは更新しない。旧の転送を観測するだけで旧値を修正しない。LEGACYのまとめ修正をこのspecの範囲にしない。

## Risks & Mitigations
旧との通信量の差を無条件に除外する危険は、実params/対象clientから積算する重複値による厳密なassertで防ぐ。集合が呼出しを越える危険は再呼出しのテストで防ぐ。

## References
- docs/research/implementation-findings/legacy-018-cross-evaluation-download-volume-without-shared-part.md
- 利用skill: fable-method、kiro-spec-init/requirements/design/tasks、kiro-impl（manual --review off）、kiro-validate-impl、kiro-verify-completion。共通引継ぎ手順を優先し独立CLIレビューを実施する。
