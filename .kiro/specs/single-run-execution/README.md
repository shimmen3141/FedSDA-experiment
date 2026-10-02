# 単一runの実行順序・SINEデータ供給spec

## 正本と読む順序

1. `spec.json`: フェーズ・承認・実装可能状態。
2. `requirements.md`: 承認済みの受け入れ条件の正本。
3. `design.md`: 責務・依存方向・入出力の設計案。承認状態を確認する。
4. `naming.md`: 名前・役割の正本。revision 1はLunaレビュー前の案であり、実装に使わない。

`reference-inventory.md`は旧SINE条件・処理順・不足条件の根拠。旧名は参照元の説明に限る。

要件は承認済み。`design.md`は責務・依存・入出力の設計案、`naming.md`はrevision 1の未承認命名案。
Luna呼出上限により命名レビューは未実施。設計と命名を承認済みと扱わない。
設計・命名確認後に`tasks.md`を作成する。存在しない文書を承認済みとして扱わない。
上位方針は`../../../docs/research/refactoring-policy.md`と`../../steering/`。
既存設定の契約は`../configuration-foundation/README.md`を参照し、ここへ複製しない。

## 境界

初回はSINEのデータ供給と単一runの実行順序を、学習・判断処理から切り離して検証する。
最終FedSDAの予測・検出・候補・学習・統合は後続の移植単位へ接続する。
処理順の観測テストを、最終FedSDA実験やgolden同値の完成と扱わない。

命名レビュー履歴・仕様レビュー証拠は`review.md`。承認状態はspec.jsonだけへ記録する。
`research.md`は設計調査の根拠。実装契約はdesign.mdと承認済みnaming.mdを参照する。
