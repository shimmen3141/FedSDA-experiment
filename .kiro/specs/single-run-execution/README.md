# 単一runの実行順序・SINEデータ供給spec

## 正本と読む順序

1. `spec.json`: フェーズ・承認・実装可能状態。
2. `requirements.md`: 承認済みの受け入れ条件の正本。
3. `design.md`: 責務・依存方向・入出力の設計案。承認状態を確認する。
4. `naming.md`: 名前・役割の正本。revision 2はLuna指摘を反映した命名表。承認状態はspec.jsonを確認する。
5. `tasks.md`: 実装順序・依存・各単位の検証条件。承認状態と進捗を確認してから実行する。

`reference-inventory.md`は旧SINE条件・処理順・不足条件の根拠。旧名は参照元の説明に限る。

要件・設計と命名revision 2は承認済み。`tasks.md`は独立レビューを通過した実装計画案であり、人間の承認待ち。
各実装単位の補助名・テスト名は、開始前にLunaレビューを通してnaming.mdへ追加する。
task承認を経て実装する。現在の承認状態はspec.jsonを参照する。
上位方針は`../../../docs/research/refactoring-policy.md`と`../../steering/`。
既存設定の契約は`../configuration-foundation/README.md`を参照し、ここへ複製しない。

## 境界

初回はSINEのデータ供給と単一runの実行順序を、学習・判断処理から切り離して検証する。
最終FedSDAの予測・検出・候補・学習・統合は後続の移植単位へ接続する。
処理順の観測テストを、最終FedSDA実験やgolden同値の完成と扱わない。

命名レビュー履歴・仕様レビュー証拠は`review.md`。承認状態はspec.jsonだけへ記録する。
`research.md`は設計調査の根拠。実装契約はdesign.mdと承認済みnaming.mdを参照する。
