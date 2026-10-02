# 単一runの実行順序・SINEデータ供給spec

## 正本と読む順序

1. `spec.json`: フェーズ・承認・実装可能状態。
2. `requirements.md`: この段階の受け入れ条件。要件案であり、承認前に実装しない。
3. `reference-inventory.md`: 旧SINE goldenの条件・処理順・不足条件の調査結果。旧名は参照元の説明に限る。
4. `naming.md`: このspecで採用する名前の正本。新しい実装名は設計の具体化後にLunaレビューする。

`design.md`・`tasks.md`は要件承認後に作成する。存在しない文書を承認済みとして扱わない。
上位方針は`../../../docs/research/refactoring-policy.md`と`../../steering/`。
既存設定の契約は`../configuration-foundation/README.md`を参照し、ここへ複製しない。

## 境界

初回はSINEのデータ供給と単一runの実行順序を、学習・判断処理から切り離して検証する。
最終FedSDAの予測・検出・候補・学習・統合は後続の移植単位へ接続する。
処理順の観測テストを、最終FedSDA実験やgolden同値の完成と扱わない。

命名レビュー履歴・仕様レビュー証拠は`review.md`。承認状態はspec.jsonだけへ記録する。
